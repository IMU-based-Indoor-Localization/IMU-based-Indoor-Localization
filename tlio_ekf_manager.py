import torch
import torch.nn.functional as F
import numpy as np

class TLIO_EKF_Manager:
    """
    학습된 IMU_ResNet_MTL 모델의 출력값을 받아 
    Extended Kalman Filter(EKF)의 관측치와 노이즈 행렬을 생성하는 클래스입니다.
    """
    def __init__(self, model, device='cpu', ekf=None):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.ekf = ekf # TLIO_EKF instance
        
        # dataset.py / test.py 기준 레이블 이름
        self.label_names = ['Trolley', 'Handbag', 'Handheld', 'Pocket', 'Running', 'Slow Walking', 'Standing']
        
        # 각 이동 상태별 허용 분산 임계값 (Threshold)
        self.class_thresholds = {
            0: 0.05, # Trolley
            1: 0.08, # Handbag
            2: 0.10, # Handheld
            3: 0.02, # Pocket
            4: 0.40, # Running
            5: 0.05  # Slow Walking
        }
        
        self.confidence_threshold = 0.7

    def init_ekf(self, start_pos=None, acc_init=None, **ekf_kwargs):
        from ekf_processor import TLIO_EKF
        self.ekf = TLIO_EKF(**ekf_kwargs)
        if start_pos is not None:
            self.ekf.p = np.array(start_pos).reshape(3, 1)
        if acc_init is not None:
            self.ekf.initialize_orientation(acc_init)
            
        # Temporal Alignment: 과거 위치 버퍼
        self.pos_buffer = [] 
        print(f"EKF Initialized with params: {ekf_kwargs}")

    @torch.no_grad()
    def get_observation(self, imu_window):
        """
        Args:
            imu_window: (12, 100) 형태 (user_acc, rotation_rate, gravity, attitude)
        """
        # 1. 텐서 변환 및 디바이스 이동
        if isinstance(imu_window, np.ndarray):
            x = torch.from_numpy(imu_window).float().to(self.device)
        else:
            x = imu_window.float().to(self.device)
            
        # [차원 교정 로직 생략 - 이미 구현됨]
        if x.dim() == 2:
            if x.shape[0] == 100 and x.shape[1] == 12: x = x.t()
            x = x.unsqueeze(0)
        elif x.dim() == 3:
            if x.shape[1] == 100 and x.shape[2] == 12: x = x.transpose(1, 2)
        
        # 3. 모델 추론
        pred_cls, pred_mu, pred_log_var = self.model(x)
        
        # 4. 결과 해석
        probs = F.softmax(pred_cls, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
        conf = conf.item()
        pred_idx = pred_idx.item()
        
        z_net = pred_mu.squeeze(0).cpu().numpy()
        var_net = torch.exp(pred_log_var).squeeze(0).cpu().numpy()
        
        # 5. 임계값 기반 노이즈 공분산(R) 조정
        final_R_diag = np.copy(var_net)
        
        if conf < self.confidence_threshold:
            status = "Uncertain"
            final_R_diag = var_net * 2.0 
        else:
            threshold = self.class_thresholds.get(pred_idx, 0.1)
            for i in range(len(var_net)):
                if var_net[i] > threshold:
                    final_R_diag[i] = var_net[i] * 5.0 # 더 보수적으로 필터링
            status = f"Steady ({self.label_names[pred_idx]})"

        return {
            'z': z_net,
            'R': np.diag(final_R_diag),
            'status': status,
            'confidence': conf
        }

    def step(self, imu_raw, dt, ai_window=None):
        if self.ekf is None:
            self.init_ekf()

        # 1. Prediction (IMU Propagation)
        acc = imu_raw[0:3]
        gyr = imu_raw[3:6]
        self.ekf.predict(dt, gyr, acc)
        
        # 기하학적 정렬을 위해 현재 위치 버퍼링
        self.pos_buffer.append(self.ekf.p.copy())
        if len(self.pos_buffer) > 200:
            self.pos_buffer.pop(0)

        # 2. Update (AI Observation)
        if ai_window is not None:
            obs = self.get_observation(ai_window)
            
            # Stochastic Cloning Lite: 
            # 모델이 쓴 윈도우(100샘플)의 시작점 위치를 찾아 기준점으로 삼습니다.
            window_size = 100
            if len(self.pos_buffer) >= window_size:
                p_old = self.pos_buffer[-window_size]
                z_pos = p_old + obs['z'][:3].reshape(3, 1)
                self.ekf.update(z_pos, obs['R'][:3, :3])
            
            return self.ekf.get_state(), obs
        
        return self.ekf.get_state(), None

if __name__ == "__main__":
    import os
    try:
        from model import IMU_ResNet_MTL
        print("✅ model.py 로드 성공")
    except ImportError:
        print("❌ 'model.py' 파일이 같은 폴더에 있어야 합니다.")
        exit()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"💻 현재 디바이스: {device}")

    model = IMU_ResNet_MTL(in_channels=12, num_classes=6).to(device)
    weight_path = 'imu_resnet_mtl_best.pth'
    
    if os.path.exists(weight_path):
        try:
            model.load_state_dict(torch.load(weight_path, map_location=device))
            print(f"✅ 가중치 로드 완료: {weight_path}")
        except:
            print(f"⚠️ 가중치 로드 실패. 무작위 가중치로 진행합니다.")
    else:
        print(f"⚠️ {weight_path} 가중치 파일이 없어 무작위 가중치로 테스트합니다.")

    manager = TLIO_EKF_Manager(model, device=device)

    # [에러 재현용 데이터] (1, 100, 12) 형태로 생성
    mock_imu_data = np.random.randn(1, 100, 12).astype(np.float32)
    
    # get_observation 내부에서 자동으로 (1, 12, 100)으로 변환됨
    obs = manager.get_observation(mock_imu_data)

    print("\n" + "="*50)
    print(f"📊 분석 결과")
    print("-" * 50)
    print(f"📍 상태: {obs['status']} (확신도: {obs['confidence']:.2f})")
    print(f"📍 예측 변화량 z: {obs['z']}")
    print(f"📍 노이즈 R (대각): {np.diag(obs['R'])}")
    print("="*50)