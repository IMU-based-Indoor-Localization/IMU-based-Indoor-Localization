import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from model import IMU_ResNet_MTL
from tlio_ekf_manager import TLIO_EKF_Manager

def run_visualizer(csv_path, model_path, max_rows=1500, ekf_params=None):
    """
    Trajectory 시뮬레이션 및 4분할 시각화 보드
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. 데이터 로드
    try:
        df = pd.read_csv(csv_path, nrows=max_rows)
        print(f"Data Load Success: {len(df)} rows")
    except Exception as e:
        print(f"Data Load Failure: {e}")
        return

    # 2. 모델 및 매니저 초기화
    model = IMU_ResNet_MTL(in_channels=12, num_classes=7).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model weights loaded.")
    model.eval()

    manager = TLIO_EKF_Manager(model, device=device)
    
    # --- 3. 경로 복원 및 시뮬레이션 ---
    
    # A. Ground Truth (정답지)
    if 'translation.x' in df.columns:
        gt_path = df[['translation.x', 'translation.y', 'translation.z']].values
    else:
        gt_path = df[['target_delta_x', 'target_delta_y', 'target_delta_z']].cumsum().values
    
    start_pos = gt_path[0]
    
    # B. AI-Only Reference (CSV의 target_delta 누적)
    # 이는 모델의 '목표값'을 누적한 것이므로 모델이 완벽할 때의 이상적인 결과입니다.
    ai_ref_delta = df[['target_delta_x', 'target_delta_y', 'target_delta_z']].values
    ai_ref_path = np.zeros_like(ai_ref_delta)
    ai_ref_path[0] = start_pos
    for i in range(1, len(ai_ref_delta)):
        ai_ref_path[i] = ai_ref_path[i-1] + ai_ref_delta[i]

    # C. Real EKF Fusion & Raw Model Simulation
    ax0 = df['user_acc_x(m/s^2)'].iloc[0] + df['gravity_x(m/s^2)'].iloc[0]
    ay0 = df['user_acc_y(m/s^2)'].iloc[0] + df['gravity_y(m/s^2)'].iloc[0]
    az0 = df['user_acc_z(m/s^2)'].iloc[0] + df['gravity_z(m/s^2)'].iloc[0]
    
    # 파라미터를 적용하여 EKF 초기화
    print(f"Custom EKF Params: {ekf_params}")
    manager.init_ekf(start_pos=start_pos, acc_init=[ax0, ay0, az0], **(ekf_params or {}))
    
    ekf_results = []
    raw_model_results = []
    current_raw_pos = np.array(start_pos, dtype=np.float64)
    
    window_size = 100
    feature_cols = [
        'user_acc_x(m/s^2)', 'user_acc_y(m/s^2)', 'user_acc_z(m/s^2)',
        'rotation_rate_x(rad/s)', 'rotation_rate_y(rad/s)', 'rotation_rate_z(rad/s)',
        'gravity_x(m/s^2)', 'gravity_y(m/s^2)', 'gravity_z(m/s^2)',
        'attitude_roll(rad)', 'attitude_pitch(rad)', 'attitude_yaw(rad)'
    ]
    
    print("Running Simulation (EKF + Raw Model)...")
    for i in range(len(df)):
        ax = df['user_acc_x(m/s^2)'].iloc[i] + df['gravity_x(m/s^2)'].iloc[i]
        ay = df['user_acc_y(m/s^2)'].iloc[i] + df['gravity_y(m/s^2)'].iloc[i]
        az = df['user_acc_z(m/s^2)'].iloc[i] + df['gravity_z(m/s^2)'].iloc[i]
        gx = df['rotation_rate_x(rad/s)'].iloc[i]
        gy = df['rotation_rate_y(rad/s)'].iloc[i]
        gz = df['rotation_rate_z(rad/s)'].iloc[i]
        
        imu_raw = [ax, ay, az, gx, gy, gz]
        dt = 0.01 
        
        ai_window = None
        if i >= window_size and i % 50 == 0:
            window_df = df.iloc[i-window_size : i]
            ai_window = window_df[feature_cols].values
            
        state, obs = manager.step(imu_raw, dt, ai_window=ai_window)
        ekf_results.append(state[0:3].flatten().copy())
        
        # Raw Model Output만 따로 누적 (EKF 보정 없음)
        if obs:
            current_raw_pos += obs['z'][:3].flatten()
        raw_model_results.append(current_raw_pos.copy())

    ekf_path = np.array(ekf_results)
    raw_model_path = np.array(raw_model_results)
    
    print(f"Final EKF Pos: {ekf_path[-1]}")
    print(f"Final Raw Model Pos: {raw_model_path[-1]}")
    print(f"Total Raw Model Movement: {np.linalg.norm(raw_model_path[-1] - start_pos)}")

    # --- 4. 4분할 시각화 ---
    fig = plt.figure(figsize=(22, 12))
    
    mid_x = (gt_path[:,0].max()+gt_path[:,0].min()) * 0.5
    mid_y = (gt_path[:,1].max()+gt_path[:,1].min()) * 0.5
    mid_z = (gt_path[:,2].max()+gt_path[:,2].min()) * 0.5
    
    gt_range = np.array([gt_path[:,0].max()-gt_path[:,0].min(), 
                         gt_path[:,1].max()-gt_path[:,1].min(), 
                         gt_path[:,2].max()-gt_path[:,2].min()]).max() / 2.0
                         
    max_range = max(gt_range * 1.2, 5.0) 
    max_range = min(max_range, 20.0)     

    def setup_ax(ax, title):
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_title(title, fontsize=14, pad=10, fontweight='bold')
        ax.scatter(start_pos[0], start_pos[1], start_pos[2], color='green', s=100, label='Start')
        ax.scatter(gt_path[-1, 0], gt_path[-1, 1], gt_path[-1, 2], color='red', s=100, label='End')

    # Subplot 1: Ground Truth
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(gt_path[:, 0], gt_path[:, 1], gt_path[:, 2], 'g-', label='GT Path', lw=2)
    setup_ax(ax1, "1. Ground Truth (Reference)")
    ax1.legend()

    # Subplot 2: AI-Only Reference (CSV Target)
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.plot(ai_ref_path[:, 0], ai_ref_path[:, 1], ai_ref_path[:, 2], 'r--', label='AI Ref (GT Delta)', lw=1.5)
    setup_ax(ax2, "2. AI-Only Reference (Target Delta)")
    ax2.legend()

    # Subplot 3: Raw Model Prediction (Pure AI)
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.plot(raw_model_path[:, 0], raw_model_path[:, 1], raw_model_path[:, 2], 'orange', label='Raw Model Prediction', lw=2)
    setup_ax(ax3, "3. Raw Model Prediction (Actual AI)")
    ax3.legend()

    # Subplot 4: EKF Fusion Result
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.plot(ekf_path[:, 0], ekf_path[:, 1], ekf_path[:, 2], 'b-', label='EKF Path', lw=2)
    setup_ax(ax4, "4. EKF Fusion Result (15-State)")
    ax4.legend()

    plt.suptitle(f"Multi-View Path Analysis\nParams: {ekf_params}", fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.show()

if __name__ == "__main__":
    dataset_path = r'c:\Users\hs091\Documents\GitHub\IMU-based-Indoor-Localization\Dataset\handbag_13.csv'
    model_path = 'imu_resnet_mtl_best.pth'
    
    # ---------------------------------------------------------
    # EKF TUNING PARAMETERS (여기서 조절하세요)
    # ---------------------------------------------------------
    tuning_params = {
        'sigma_na': 0.1,    # 가속도 노이즈 (클수록 IMU를 적게 믿음)
        'sigma_ng': 0.01,   # 자이로 노이즈
        'ita_ba': 1e-4,     # 가속도 바이어스 변동률
        'ita_bg': 1e-6      # 자이로 바이어스 변동률
    }
    
    MAX_LOAD_COUNT = 1500 
    
    if os.path.exists(dataset_path):
        run_visualizer(dataset_path, model_path, max_rows=MAX_LOAD_COUNT, ekf_params=tuning_params)
    else:
        print(f"File not found: {dataset_path}")