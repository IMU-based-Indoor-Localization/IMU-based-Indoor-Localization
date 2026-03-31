import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def run_visualizer(csv_path, model_path, max_rows=1000):
    """
    AI 모델 예측값과 EKF 융합 결과를 개별 서브플롯으로 시각화합니다.
    max_rows: 시각화할 최대 데이터 행 수 (눈 피로 방지 및 성능 최적화)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 데이터 로드 (nrows를 사용하여 필요한 만큼만 로드)
    try:
        # 데이터가 너무 많을 경우를 대비해 지정된 행 수만 읽어옵니다.
        df = pd.read_csv(csv_path, nrows=max_rows)
        print(f"✅ 데이터 로드 성공: {len(df)} 행 (최대 제한: {max_rows}행)")
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return

    # --- 2. 경로 복원 로직 ---
    
    # A. Ground Truth (정답지) 경로 복원
    if 'translation.x' in df.columns:
        gt_path = df[['translation.x', 'translation.y', 'translation.z']].values
    else:
        # 절대 좌표가 없을 경우 Delta(변위) 값을 누적 합산하여 경로 생성
        gt_path = df[['target_delta_x', 'target_delta_y', 'target_delta_z']].cumsum().values
    
    # B. AI 예측 시뮬레이션
    ai_delta = df[['target_delta_x', 'target_delta_y', 'target_delta_z']].values
    
    # 초기 위치 (GT의 첫 번째 좌표)
    start_pos = gt_path[0]
    
    # AI-Only 경로 복원 (시작점 + 누적 변위)
    ai_path = np.zeros_like(ai_delta)
    ai_path[0] = start_pos
    for i in range(1, len(ai_delta)):
        ai_path[i] = ai_path[i-1] + ai_delta[i]

    # C. EKF Fusion 경로 (가상 시뮬레이션)
    # 실제 환경에서는 EKF 클래스의 출력값을 사용해야 합니다.
    # 여기서는 시각화 확인을 위해 GT 70%, AI 30% 비율로 섞은 가상 결과물을 생성합니다.
    ekf_path = (gt_path * 0.7 + ai_path * 0.3) 

    # --- 3. 개별 3D 시각화 (Subplots) ---
    fig = plt.figure(figsize=(20, 7))
    
    # 모든 그래프의 축 범위를 통일하기 위한 계산 (가독성 향상)
    all_points = np.vstack([gt_path, ai_path, ekf_path])
    max_range = np.array([all_points[:,0].max()-all_points[:,0].min(), 
                          all_points[:,1].max()-all_points[:,1].min(), 
                          all_points[:,2].max()-all_points[:,2].min()]).max() / 2.0
    mid_x = (all_points[:,0].max()+all_points[:,0].min()) * 0.5
    mid_y = (all_points[:,1].max()+all_points[:,1].min()) * 0.5
    mid_z = (all_points[:,2].max()+all_points[:,2].min()) * 0.5

    def setup_ax(ax, title, color):
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_title(title, fontsize=14, pad=10, fontweight='bold')
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        # 시작점 표시
        ax.scatter(start_pos[0], start_pos[1], start_pos[2], color='green', s=100, label='Start', edgecolors='black', zorder=5)
        # 끝점 표시
        ax.scatter(gt_path[-1, 0], gt_path[-1, 1], gt_path[-1, 2], color='red', s=100, label='End', edgecolors='black', zorder=5)

    # Subplot 1: Ground Truth
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(gt_path[:, 0], gt_path[:, 1], gt_path[:, 2], 'g-', label='GT Path', lw=2.5, alpha=0.7)
    setup_ax(ax1, "1. Ground Truth (Reference)", 'green')
    ax1.legend()

    # Subplot 2: AI-Only
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot(ai_path[:, 0], ai_path[:, 1], ai_path[:, 2], 'r--', label='AI Path', lw=1.5)
    setup_ax(ax2, "2. AI-Only Prediction", 'red')
    ax2.legend()

    # Subplot 3: EKF Fusion
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot(ekf_path[:, 0], ekf_path[:, 1], ekf_path[:, 2], 'b-', label='EKF Path', lw=2)
    setup_ax(ax3, "3. EKF Fusion Result", 'blue')
    ax3.legend()

    # 전체 제목 및 레이아웃 조정
    plt.suptitle(f"Trajectory Comparison: Separate Views (Max {max_rows} rows)\nSource: {os.path.basename(csv_path)}", fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.show()

if __name__ == "__main__":
    # 파일 경로 설정
    dataset_path = '/Users/parkhaneul/Documents/IMU-based-Indoor-Localization/Dataset/handbag_13.csv'
    dummy_model_path = 'best_model.pth'
    
    # ---------------------------------------------------------
    # MAX_LOAD_COUNT: 시각화할 데이터 포인트를 조절하세요 (예: 500, 1000, 3000)
    # ---------------------------------------------------------
    MAX_LOAD_COUNT = 1500 
    
    if os.path.exists(dataset_path):
        run_visualizer(dataset_path, dummy_model_path, max_rows=MAX_LOAD_COUNT)
    else:
        print(f"❌ 파일을 찾을 수 없습니다: {dataset_path}")