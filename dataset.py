import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset

class IMUDataset(Dataset):
    def __init__(self, data_dir, window_size=100, step_size=50):
        self.window_size = window_size
        self.step_size = step_size
        
        if os.path.isfile(data_dir):
            csv_files = [data_dir]
        else:
            csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
            
        if not csv_files:
            raise ValueError(f"'{data_dir}'에서 CSV 파일을 찾을 수 없습니다.")

        print(f"✅ 총 {len(csv_files)}개의 CSV 파일을 로드합니다.")
        df_list = [pd.read_csv(f) for f in csv_files]
        self.df = pd.concat(df_list, ignore_index=True)
        
        self.feature_cols = [
            'user_acc_x(m/s^2)', 'user_acc_y(m/s^2)', 'user_acc_z(m/s^2)',
            'rotation_rate_x(rad/s)', 'rotation_rate_y(rad/s)', 'rotation_rate_z(rad/s)',
            'gravity_x(m/s^2)', 'gravity_y(m/s^2)', 'gravity_z(m/s^2)',
            'attitude_roll(rad)', 'attitude_pitch(rad)', 'attitude_yaw(rad)'
        ]
        self.target_cols = [
            'target_delta_x', 'target_delta_y', 'target_delta_z',
            'target_delta_roll', 'target_delta_pitch', 'target_delta_yaw'
        ]
        
        self.windows = list(range(0, len(self.df) - self.window_size, self.step_size))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start = self.windows[idx]
        window = self.df.iloc[start : start + self.window_size]
        
        x_tensor = torch.tensor(window[self.feature_cols].values, dtype=torch.float32).t()
        y_reg = torch.tensor(window[self.target_cols].iloc[-1].values, dtype=torch.float32)
        
        label = int(window['placement_label'].iloc[-1])
        y_cls = torch.tensor(label, dtype=torch.long)
        
        return x_tensor, y_cls, y_reg