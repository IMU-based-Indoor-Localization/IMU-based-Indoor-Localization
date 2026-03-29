import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 업로드된 모듈 임포트
from model import IMU_ResNet_MTL
from tlio_ekf_manager import TLIO_EKF_Manager
from ekf_processor import TLIO_EKF

def run_visualizer(csv_path, model_path):
    # 0. 환경 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = 0.01  # 100Hz 기준 (0.01초)
    window_size = 100
    step_size = 10  # 시각화 속도를 위해 10개 샘플마다 추론
    
    # 1. 모델 및 클래스 초기화
    model = IMU_ResNet_MTL(in_channels=12, num_classes=7).to(device)
    try:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"✅ 모델 로드 성공: {model_path}")
    except Exception as e:
        print(f"⚠️ 모델 로드 실패: {e}. 무작위 가중치로 진행합니다.")
    
    manager = TLIO_EKF_Manager(model, device=device)
    ekf = TLIO_EKF() # EKF 본체 (pos, vel 초기화됨)
    
    # 2. 데이터 로드
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ CSV 로드 실패: {e}")
        return

    # feature_cols 정의
    feature_cols = [
        'user_acc_x(m/s^2)', 'user_acc_y(m/s^2)', 'user_acc_z(m/s^2)',
        'rotation_rate_x(rad/s)', 'rotation_rate_y(rad/s)', 'rotation_rate_z(rad/s)',
        'gravity_x(m/s^2)', 'gravity_y(m/s^2)', 'gravity_z(m/s^2)',
        'attitude_roll(rad)', 'attitude_pitch(rad)', 'attitude_yaw(rad)'
    ]
    
    traj_ref = []  # 정답 경로 (Ground Truth)
    traj_imu = []  # IMU 전용 경로 (Pure Double Integration)
    traj_ekf = []  # AI-EKF 융합 경로
    
    # IMU Only 변수 초기화
    pos_imu = np.zeros(3)
    vel_imu = np.zeros(3)
    
    print("🔄 궤적 계산 및 필터링 시작...")
    
    # 3. 메인 루프 (Sliding Window 방식)
    for i in range(0, len(df) - window_size, step_size):
        # 현재 윈도우 데이터 추출
        window_df = df.iloc[i : i + window_size]
        window_data = window_df[feature_cols].values
        
        # --- A. 정답 데이터 (Vicon Delta 누적) ---
        # 0번부터 현재 i까지의 누적 변위를 계산하여 위치 복원
        ref_x = df['target_delta_x'].iloc[:i+window_size].sum()
        ref_y = df['target_delta_y'].iloc[:i+window_size].sum()
        ref_z = df['target_delta_z'].iloc[:i+window_size].sum()
        traj_ref.append([ref_x, ref_y, ref_z])
        
        # --- B. IMU Only (가속도 이중 적분) ---
        # 윈도우의 마지막 가속도 값을 사용하여 dt*step_size만큼 적분
        acc_raw = window_data[-1, 0:3]
        total_dt = dt * step_size
        vel_imu += acc_raw * total_dt
        pos_imu += vel_imu * total_dt
        traj_imu.append(pos_imu.copy())
        
        # --- C. AI-EKF 융합 ---
        # 1. AI 관측치 획득 (z: 델타 변위/회전, R: 조정된 공분산)
        obs = manager.get_observation(window_data)
        
        # 2. EKF 예측 (IMU 가속도 기반)
        ekf.predict(total_dt, acc_raw)
        
        # 3. EKF 보정 (AI가 예측한 변위 6차원 중 위치 관련 3차원만 사용)
        # obs['z'] index: 0=dx, 1=dy, 2=dz
        z_ai = obs['z'][:3]
        R_ai = obs['R'][:3, :3]
        
        updated_state, _ = ekf.update(z_ai, R_ai)
        traj_ekf.append(updated_state[0:3].flatten())

    # 4. 데이터 배열 변환
    traj_ref = np.array(traj_ref)
    traj_imu = np.array(traj_imu)
    traj_ekf = np.array(traj_ekf)
    
    # 5. 3D 시각화
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 정답 궤적 (초록 실선)
    ax.plot(traj_ref[:,0], traj_ref[:,1], traj_ref[:,2], 'g-', label='Reference (Ground Truth)', lw=2)
    
    # AI-EKF 융합 궤적 (파란 실선)
    ax.plot(traj_ekf[:,0], traj_ekf[:,1], traj_ekf[:,2], 'b-', label='AI-EKF Fusion', lw=1.5)
    
    # 순수 IMU 적분 궤적 (빨간 점선)
    ax.plot(traj_imu[:,0], traj_imu[:,1], traj_imu[:,2], 'r--', label='Pure IMU (Dead Reckoning)', alpha=0.5)
    
    ax.set_title("3D Trajectory Comparison: IMU vs AI-EKF vs Ground Truth")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    
    # 보기 좋게 스케일 조정
    all_points = np.vstack([traj_ref, traj_ekf])
    max_range = np.array([all_points[:,0].max()-all_points[:,0].min(), 
                          all_points[:,1].max()-all_points[:,1].min(), 
                          all_points[:,2].max()-all_points[:,2].min()]).max() / 2.0
    mid_x = (all_points[:,0].max()+all_points[:,0].min()) * 0.5
    mid_y = (all_points[:,1].max()+all_points[:,1].min()) * 0.5
    mid_z = (all_points[:,2].max()+all_points[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    print("✅ 시각화 완료. 팝업 창을 확인하세요.")
    plt.show()

if __name__ == "__main__":
    # 데이터셋 경로와 가중치 파일명을 확인하세요.
    CSV_PATH = r'c:\Users\hs091\Desktop\대학\2026 1학기\전자공학종합설계\전처리 데이터 셋\handbag_11.csv'
    MODEL_PATH = 'imu_resnet_mtl_best.pth'
    
    run_visualizer(CSV_PATH, MODEL_PATH)