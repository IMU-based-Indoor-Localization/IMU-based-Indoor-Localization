import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

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
    # 모델 정의: num_classes는 학습 시 설정에 따라 6 또는 7일 수 있음
    model = IMU_ResNet_MTL(in_channels=12, num_classes=7).to(device)
    try:
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"✅ 모델 로드 성공: {model_path}")
        else:
            print(f"⚠️ 모델 파일을 찾을 수 없습니다: {model_path}. 무작위 가중치로 진행합니다.")
    except Exception as e:
        print(f"⚠️ 모델 로드 중 오류 발생: {e}. 무작위 가중치로 진행합니다.")
    
    manager = TLIO_EKF_Manager(model, device=device)
    ekf = TLIO_EKF() # EKF 본체 (AI-EKF 융합용)
    
    # 2. 데이터 로드
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ 데이터 로드 성공: {csv_path} (총 {len(df)} 행)")
    except Exception as e:
        print(f"❌ CSV 로드 실패: {e}")
        return

    # 입력 데이터로 사용할 컬럼 정의
    feature_cols = [
        'user_acc_x(m/s^2)', 'user_acc_y(m/s^2)', 'user_acc_z(m/s^2)',
        'rotation_rate_x(rad/s)', 'rotation_rate_y(rad/s)', 'rotation_rate_z(rad/s)',
        'gravity_x(m/s^2)', 'gravity_y(m/s^2)', 'gravity_z(m/s^2)',
        'attitude_roll(rad)', 'attitude_pitch(rad)', 'attitude_yaw(rad)'
    ]

    # 가속도 데이터 추출 (IMU 적분용)
    raw_accs = df[['user_acc_x(m/s^2)', 'user_acc_y(m/s^2)', 'user_acc_z(m/s^2)']].values
    
    # 정답 궤적 계산 (Target Delta x, y, z만 사용)
    target_deltas = df[['target_delta_x', 'target_delta_y', 'target_delta_z']].values
    traj_ref = np.cumsum(target_deltas, axis=0)

    # 결과 저장을 위한 리스트
    traj_ekf = []      # AI + IMU (EKF)
    traj_imu = []      # Pure IMU (Dead Reckoning)
    traj_ai_only = []  # 순수 네트워크 예측값 누적
    
    # 초기 상태 설정
    curr_pos_imu = np.zeros(3)
    curr_vel_imu = np.zeros(3)
    curr_pos_ai = np.zeros(3) # AI 전용 위치 누적기
    
    print("🏃 시뮬레이션 시작...")
    
    # 3. 루프 실행
    for i in range(window_size, len(df), step_size):
        # A. Pure IMU 적분 (비교용)
        for j in range(step_size):
            idx = i - step_size + j
            acc = raw_accs[idx]
            curr_vel_imu += acc * dt
            curr_pos_imu += curr_vel_imu * dt
        traj_imu.append(curr_pos_imu.copy())
        
        # B. EKF 예측 단계 (IMU 입력)
        for j in range(step_size):
            ekf.predict(dt, raw_accs[i - step_size + j])
            
        # C. EKF 보정 단계 (AI 모델 입력)
        window_data = df.iloc[i-window_size:i][feature_cols].values
        
        # AI 추론
        obs_data = manager.get_observation(window_data)
        z_full = obs_data['z']  # 네트워크가 예측한 값 (6차원: [dx, dy, dz, dRoll, dPitch, dYaw])
        R_full = obs_data['R']  # 불확실성 행렬 (6x6)
        
        # 에러 해결: EKF는 위치(3차원)만 관측하므로 앞의 3개만 추출
        # z_full.shape가 (6, 1)이거나 (6,)인 경우를 대비해 슬라이싱 후 리셰이프
        z_pos = z_full.flatten()[:3].reshape(3, 1)
        R_pos = R_full[:3, :3] # 공분산 행렬도 3x3으로 슬라이싱
        
        # EKF Update (위치 정보만 업데이트)
        ekf.update(z_pos, R_pos)
        traj_ekf.append(ekf.x[0:3].flatten())
        
        # D. 순수 AI 출력값 누적
        curr_pos_ai += z_pos.flatten() 
        traj_ai_only.append(curr_pos_ai.copy())

    # 리스트를 넘파이 배열로 변환
    traj_ekf = np.array(traj_ekf)
    traj_imu = np.array(traj_imu)
    traj_ai_only = np.array(traj_ai_only)
    
    # 4. 시각화
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(0, 0, 0, color='black', s=100, label='Start', marker='x', zorder=10)
    ax.plot(traj_ref[:,0], traj_ref[:,1], traj_ref[:,2], 'g-', label='Reference (GT)', lw=2)
    ax.plot(traj_ekf[:,0], traj_ekf[:,1], traj_ekf[:,2], 'b-', label='AI-EKF Fusion', lw=1.5)
    ax.plot(traj_ai_only[:,0], traj_ai_only[:,1], traj_ai_only[:,2], color='orange', label='AI-Only (Network Output)', lw=1.2)
    ax.plot(traj_imu[:,0], traj_imu[:,1], traj_imu[:,2], 'r--', label='Pure IMU (Dead Reckoning)', alpha=0.4)
    
    ax.set_title("3D Trajectory: IMU vs AI vs EKF Fusion")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    
    # 스케일 조정
    all_points = np.vstack([traj_ref, traj_ekf, traj_ai_only])
    max_range = np.array([all_points[:,0].max()-all_points[:,0].min(), 
                          all_points[:,1].max()-all_points[:,1].min(), 
                          all_points[:,2].max()-all_points[:,2].min()]).max() / 2.0
    mid_x = (all_points[:,0].max()+all_points[:,0].min()) * 0.5
    mid_y = (all_points[:,1].max()+all_points[:,1].min()) * 0.5
    mid_z = (all_points[:,2].max()+all_points[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.show()

if __name__ == "__main__":
    CSV_PATH = '/Users/parkhaneul/Documents/IMU-based-Indoor-Localization/Dataset/large_scale_floor4_3.csv'
    MODEL_PATH = 'imu_resnet_mtl_best.pth'
    
    run_visualizer(CSV_PATH, MODEL_PATH)