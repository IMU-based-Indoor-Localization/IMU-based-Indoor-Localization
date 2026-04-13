import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from model import IMU_ResNet_MTL
from tlio_ekf_manager import TLIO_EKF_Manager

def run_ekf_simulation(csv_path, model_path, max_rows=1500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    df = pd.read_csv(csv_path, nrows=max_rows)
    print(f"Dataset loaded: {len(df)} rows")

    # 2. Check Columns
    feature_cols = [
        'user_acc_x(m/s^2)', 'user_acc_y(m/s^2)', 'user_acc_z(m/s^2)',
        'rotation_rate_x(rad/s)', 'rotation_rate_y(rad/s)', 'rotation_rate_z(rad/s)',
        'gravity_x(m/s^2)', 'gravity_y(m/s^2)', 'gravity_z(m/s^2)',
        'attitude_roll(rad)', 'attitude_pitch(rad)', 'attitude_yaw(rad)'
    ]
    
    # 3. Initialize Model and Manager
    model = IMU_ResNet_MTL(in_channels=12, num_classes=7).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model weights loaded.")
    model.eval()

    manager = TLIO_EKF_Manager(model, device=device)
    
    # Determine start position
    if 'translation.x' in df.columns:
        start_pos = [df['translation.x'].iloc[0], df['translation.y'].iloc[0], df['translation.z'].iloc[0]]
        gt_path = df[['translation.x', 'translation.y', 'translation.z']].values
    else:
        start_pos = [0, 0, 0]
        gt_path = df[['target_delta_x', 'target_delta_y', 'target_delta_z']].cumsum().values

    manager.init_ekf(start_pos=start_pos)

    # 4. Simulation Loop
    ekf_results = []
    ai_results = []
    window_size = 100
    current_ai_pos = np.array(start_pos, dtype=np.float64)
    
    print("Running EKF Simulation...")
    for i in range(len(df)):
        if i % 100 == 0:
            print(f"Step {i}/{len(df)}...")
            
        # Prepare IMU raw
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
        
        if obs:
            current_ai_pos += obs['z'][:3].flatten()
        ai_results.append(current_ai_pos.copy())

    ekf_path = np.array(ekf_results)
    ai_path = np.array(ai_results)

    # 5. Visualization
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(gt_path[:, 0], gt_path[:, 1], gt_path[:, 2], 'g-', label='Ground Truth', alpha=0.5)
    ax.plot(ai_path[:, 0], ai_path[:, 1], ai_path[:, 2], 'r--', label='AI-Only', alpha=0.5)
    ax.plot(ekf_path[:, 0], ekf_path[:, 1], ekf_path[:, 2], 'b-', label='EKF (TLIO Logic)', lw=2)
    
    ax.set_title("15-State EKF Fusion Result")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    csv_file = r'c:\Users\hs091\Documents\GitHub\IMU-based-Indoor-Localization\Dataset\handbag_13.csv'
    model_file = 'imu_resnet_mtl_best.pth'
    run_ekf_simulation(csv_file, model_file)
