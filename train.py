import torch
import numpy as np
import matplotlib.pyplot as plt  # 시각화를 위한 라이브러리 추가
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from torch.utils.data import DataLoader

# 모듈화된 파일들 불러오기
from dataset import IMUDataset
from model import IMU_ResNet_MTL
from loss import MTL_CustomLoss

def train_model():
    data_dir = r"c:\Users\hs091\Desktop\train2\dataset"
    
    dataset = IMUDataset(data_dir, window_size=100, step_size=50)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 사용하는 디바이스: {device}")
    
    model = IMU_ResNet_MTL(in_channels=12, num_classes=7).to(device)
    criterion = MTL_CustomLoss(lambda_pos=10.0, lambda_unc=1.0, lambda_cls=1.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 15
    
    # ==========================================================
    # 🌟 시각화를 위한 히스토리 기록 리스트 생성
    # ==========================================================
    history_loss = []
    history_rmse = []
    history_acc = []
    
    print("\n🚀 학습을 시작합니다...")
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_rmse = 0.0
        epoch_acc = 0.0
        valid_cls_batches = 0
        
        for batch_idx, (x, y_cls, y_reg) in enumerate(dataloader):
            x, y_cls, y_reg = x.to(device), y_cls.to(device), y_reg.to(device)
            
            optimizer.zero_grad()
            pred_cls, pred_mu, pred_log_var = model(x)
            
            total_loss, l_mse, l_nll, l_cls = criterion(pred_mu, pred_log_var, y_reg, pred_cls, y_cls)
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
            rmse = torch.sqrt(l_mse).item()
            epoch_rmse += rmse
            
            valid_mask = (y_cls != -1)
            if valid_mask.sum() > 0:
                _, predicted = torch.max(pred_cls[valid_mask], 1)
                correct = (predicted == y_cls[valid_mask]).sum().item()
                acc = (correct / valid_mask.sum().item()) * 100.0
                epoch_acc += acc
                valid_cls_batches += 1
            
            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}] "
                      f"Loss: {total_loss.item():.2f} | "
                      f"RMSE: {rmse:.4f} | Acc: {acc if valid_mask.sum() > 0 else 0.0:.1f}%")
                
        # 에포크 단위 평균 계산
        avg_loss = epoch_loss / len(dataloader)
        avg_rmse = epoch_rmse / len(dataloader)
        avg_acc = epoch_acc / valid_cls_batches if valid_cls_batches > 0 else 0.0
        
        # 🌟 계산된 평균값을 리스트에 저장
        history_loss.append(avg_loss)
        history_rmse.append(avg_rmse)
        history_acc.append(avg_acc)
        
        print(f"==> 🏁 Epoch [{epoch+1}] 완료! "
              f"Avg Loss: {avg_loss:.4f} | Avg RMSE: {avg_rmse:.4f} | Avg Acc: {avg_acc:.2f}%\n")

    print("💾 학습된 모델의 가중치를 저장합니다...")
    torch.save(model.state_dict(), 'imu_resnet_mtl_best.pth')
    print("✅ 'imu_resnet_mtl_best.pth' 저장 완료!")
    
    print("🎉 학습이 완료되었습니다. 결과를 그래프로 출력합니다.")
    
    # ==========================================================
    # 🌟 Matplotlib을 활용한 학습 결과 시각화
    # ==========================================================
    epochs_range = range(1, num_epochs + 1)
    
    plt.figure(figsize=(15, 5)) # 가로로 긴 형태의 도화지 생성
    
    # 1. Total Loss 그래프
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history_loss, marker='o', color='red', label='Total Loss')
    plt.title('Training Total Loss (NLL included)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # 2. RMSE (이동 오차) 그래프
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, history_rmse, marker='s', color='blue', label='RMSE (Trajectory Error)')
    plt.title('Training RMSE (Lower is better)')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.legend()
    
    # 3. Accuracy (분류 정확도) 그래프
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, history_acc, marker='^', color='green', label='Classification Accuracy')
    plt.title('Training Accuracy (Higher is better)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_result.png') # 이미지 파일로 먼저 저장
    plt.show() # 화면에 띄우기

if __name__ == '__main__':
    train_model()
    