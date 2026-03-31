import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def visualize_all_sensors(handbag_file, vi_file, imu_raw_file, max_rows=1500):
    """
    3개의 CSV 파일을 읽어 4개의 3D 그래프를 시각화합니다.
    1. IMU User Acc (from handbag_xx)
    2. Ground Truth / Target Delta (from handbag_xx)
    3. VI Translation Path (from vi1.csv)
    4. Raw IMU User Acc (from imu1.csv)
    """
    try:
        # 1. handbag_xx.csv 로드 (헤더 있음)
        df_handbag = pd.read_csv(handbag_file, nrows=max_rows)
        
        # 2. vi1.csv 로드 (헤더 없음, 수동 지정)
        vi_cols = ['Time', 'Header', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz', 'rw']
        df_vi = pd.read_csv(vi_file, names=vi_cols, nrows=max_rows)
        
        # 3. imu1.csv 로드 (제공된 헤더 순서대로 수동 지정)
        imu_cols = [
            'Time', 'att_roll', 'att_pitch', 'att_yaw', 
            'rot_x', 'rot_y', 'rot_z', 
            'grav_x', 'grav_y', 'grav_z', 
            'user_acc_x', 'user_acc_y', 'user_acc_z', 
            'mag_x', 'mag_y', 'mag_z'
        ]
        df_imu_raw = pd.read_csv(imu_raw_file, names=imu_cols, nrows=max_rows)
        
        print("모든 파일 로드 완료.")
    except Exception as e:
        print(f"파일 로드 중 오류 발생: {e}")
        return

    # 시각화 설정 (2x2 레이아웃)
    fig = plt.figure(figsize=(18, 12))

    # --- 1. IMU User Acc (handbag_xx.csv) ---
    ax1 = fig.add_subplot(221, projection='3d')
    x1, y1, z1 = df_handbag['user_acc_x(m/s^2)'], df_handbag['user_acc_y(m/s^2)'], df_handbag['user_acc_z(m/s^2)']
    ax1.plot(x1, y1, z1, color='steelblue', label='Acc (m/s^2)')
    ax1.scatter(x1.iloc[0], y1.iloc[0], z1.iloc[0], color='green', s=40, label='Start')
    ax1.scatter(x1.iloc[-1], y1.iloc[-1], z1.iloc[-1], color='red', s=40, label='End')
    ax1.set_title('1. IMU User Acc (Handbag CSV)')
    ax1.legend()

    # --- 2. Ground Truth / Target Delta (handbag_xx.csv) ---
    ax2 = fig.add_subplot(222, projection='3d')
    x2, y2, z2 = df_handbag['target_delta_x'], df_handbag['target_delta_y'], df_handbag['target_delta_z']
    ax2.plot(x2, y2, z2, color='darkorange', linewidth=2, label='GT Path')
    ax2.scatter(x2.iloc[0], y2.iloc[0], z2.iloc[0], color='green', s=40, label='Start')
    ax2.scatter(x2.iloc[-1], y2.iloc[-1], z2.iloc[-1], color='red', s=40, label='End')
    ax2.set_title('2. Target Delta (Handbag CSV)')
    ax2.legend()

    # --- 3. VI Translation Path (vi1.csv) ---
    ax3 = fig.add_subplot(223, projection='3d')
    x3, y3, z3 = df_vi['tx'], df_vi['ty'], df_vi['tz']
    ax3.plot(x3, y3, z3, color='forestgreen', linewidth=2, label='VI Path')
    ax3.scatter(x3.iloc[0], y3.iloc[0], z3.iloc[0], color='green', s=40, label='Start')
    ax3.scatter(x3.iloc[-1], y3.iloc[-1], z3.iloc[-1], color='red', s=40, label='End')
    ax3.set_title('3. VI Translation (vi CSV)')
    ax3.legend()

    # --- 4. Raw IMU User Acc (imu1.csv) ---
    ax4 = fig.add_subplot(224, projection='3d')
    x4, y4, z4 = df_imu_raw['user_acc_x'], df_imu_raw['user_acc_y'], df_imu_raw['user_acc_z']
    ax4.plot(x4, y4, z4, color='purple', alpha=0.7, label='Raw Acc (G)')
    ax4.scatter(x4.iloc[0], y4.iloc[0], z4.iloc[0], color='green', s=40, label='Start')
    ax4.scatter(x4.iloc[-1], y4.iloc[-1], z4.iloc[-1], color='red', s=40, label='End')
    ax4.set_title('4. Raw IMU User Acc (imu CSV)')
    ax4.legend()

    # 전체 레이아웃 조정
    plt.suptitle(f'Comprehensive Sensor Data Visualization (First {max_rows} points)', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    # 각 파일의 실제 경로를 입력하세요.
    path_handbag = '/Users/parkhaneul/Desktop/IMU-based-Indoor-Localization/Dataset/handbag_21.csv'
    path_vi = '/Users/parkhaneul/Desktop/Oxford Inertial Odometry Dataset/handbag/data1/syn/vi1.csv'
    path_imu_raw = '/Users/parkhaneul/Desktop/Oxford Inertial Odometry Dataset/handbag/data1/syn/imu1.csv'
    
    # 시각화할 데이터 행 수 (너무 많으면 느려질 수 있으므로 적절히 조절)
    ROWS_TO_VIEW = 1500
    
    visualize_all_sensors(path_handbag, path_vi, path_imu_raw, max_rows=ROWS_TO_VIEW)