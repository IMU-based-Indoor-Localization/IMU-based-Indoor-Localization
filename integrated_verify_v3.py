import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

def quaternion_to_euler(row):
    """Quaternion (x, y, z, w) -> Euler (roll, pitch, yaw) 변환"""
    try:
        quat = [row['rotation.x'], row['rotation.y'], row['rotation.z'], row['rotation.w']]
        # 쿼터니언 정규화 (수치 안정성을 위해 필수)
        quat = quat / np.linalg.norm(quat)
        r = R.from_quat(quat)
        return r.as_euler('xyz')
    except:
        return [0, 0, 0]

def process_and_verify(input_csv, output_csv):
    # 1. 원본 데이터 로드
    col_names = ['Time', 'Header', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
    df = pd.read_csv(input_csv, names=col_names)
    df.columns = [
        'Time', 'Header', 
        'translation.x', 'translation.y', 'translation.z', 
        'rotation.x', 'rotation.y', 'rotation.z', 'rotation.w'
    ]

    # 2. Delta 계산 (이전 좌표와의 차이)
    # 수치적 안정성을 위해 float64 사용
    df['target_delta_x'] = df['translation.x'].diff().fillna(0).astype(np.float64)
    df['target_delta_y'] = df['translation.y'].diff().fillna(0).astype(np.float64)
    df['target_delta_z'] = df['translation.z'].diff().fillna(0).astype(np.float64)

    # 3. Euler 각도 변환
    print("Converting Quaternions to Euler angles...")
    eulers = df.apply(quaternion_to_euler, axis=1, result_type='expand')
    df[['target_roll', 'target_pitch', 'target_yaw']] = eulers

    # 4. 결과 저장
    df.to_csv(output_csv, index=False)
    print(f"File saved: {output_csv}")

    # 5. 시각화 검증
    visualize_final_check(df)

def visualize_final_check(df, max_points=3000):
    df_sub = df.iloc[:max_points].copy()
    
    fig = plt.figure(figsize=(16, 8))
    
    # --- 좌측: 원본 vs 재구성 경로 (모양 일치 확인) ---
    ax1 = fig.add_subplot(121, projection='3d')
    
    # 원본
    ox, oy, oz = df_sub['translation.x'], df_sub['translation.y'], df_sub['translation.z']
    ax1.plot(ox, oy, oz, color='blue', alpha=0.5, linewidth=4, label='Original (Absolute)')
    
    # Delta 누적합 (재구성)
    # 시작점 Offset을 정확히 더해줌
    rx = df_sub['target_delta_x'].cumsum() + ox.iloc[0]
    ry = df_sub['target_delta_y'].cumsum() + oy.iloc[0]
    rz = df_sub['target_delta_z'].cumsum() + oz.iloc[0]
    
    ax1.plot(rx, ry, rz, color='red', linestyle='--', linewidth=2, label='Reconstructed (Cumsum Delta)')
    
    ax1.set_title("Path Consistency Check\n(Red dashed line should overlap Blue line)")
    ax1.legend()

    # 축 범위 균등화 (이게 없으면 모양이 다르게 보임)
    all_data = np.array([ox, oy, oz])
    max_range = (all_data.max(axis=1) - all_data.min(axis=1)).max() / 2.0
    mid_points = (all_data.max(axis=1) + all_data.min(axis=1)) / 2.0
    ax1.set_xlim(mid_points[0] - max_range, mid_points[0] + max_range)
    ax1.set_ylim(mid_points[1] - max_range, mid_points[1] + max_range)
    ax1.set_zlim(mid_points[2] - max_range, mid_points[2] + max_range)

    # --- 우측: 수치적 오차 확인 (누적 오차) ---
    ax2 = fig.add_subplot(122)
    error = np.sqrt((ox - rx)**2 + (oy - ry)**2 + (oz - rz)**2)
    ax2.plot(error, color='purple')
    ax2.set_title("Reconstruction Drift (Numerical Error)")
    ax2.set_xlabel("Sample Index")
    ax2.set_ylabel("Error Distance (meters)")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    IN_FILE = 'vi1.csv'
    OUT_FILE = 'vi1_verified.csv'
    process_and_verify(IN_FILE, OUT_FILE)