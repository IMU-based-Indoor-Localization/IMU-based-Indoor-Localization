import numpy as np

class TLIO_EKF:
    """
    상태 변수 x: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z] (6차원)
    IMU 데이터를 통해 예측(Dead Reckoning)하고, 네트워크(AI) 출력값으로 보정합니다.
    
    참고: 시각화 시 EKF를 거치지 않고 네트워크 출력값(z)만 누적하면 
    'AI-Only' 궤적을 그릴 수 있습니다.
    """
    def __init__(self, state_dim=6, obs_dim=3):
        # 초기 상태 (위치 0, 속도 0)
        self.x = np.zeros((state_dim, 1))
        
        # 오차 공분산 행렬 P (초기 불확실성)
        self.P = np.eye(state_dim) * 0.1
        
        # 관측 행렬 H: AI 모델이 '변위(속도 성분)'를 예측한다고 가정 (3x6)
        # 상태 변수의 [3, 4, 5]번째 인덱스인 속도 항을 관측함
        self.H = np.zeros((obs_dim, state_dim))
        self.H[:, 3:6] = np.eye(obs_dim)

    def predict(self, dt, raw_acc):
        """
        IMU 가속도 데이터를 이용한 예측 단계 (Dead Reckoning)
        """
        # 1. 상태 전이 행렬 F 정의 (단순 등가속도/등속 모델)
        F = np.eye(6)
        F[0:3, 3:6] = np.eye(3) * dt
        
        # 2. 제어 입력 모델 B (가속도를 입력으로 사용)
        B = np.zeros((6, 3))
        B[0:3, :] = 0.5 * (dt**2) * np.eye(3)
        B[3:6, :] = dt * np.eye(3)
        
        # 3. 상태 예측: x = Fx + Bu
        u = raw_acc.reshape(3, 1)
        self.x = F @ self.x + B @ u
        
        # 4. 공분산 예측: P = FPF' + Q (Q는 시스템 노이즈)
        Q = np.eye(6) * 0.001
        self.P = F @ self.P @ F.T + Q

    def update(self, z, R):
        """
        네트워크(AI) 관측치를 이용한 보정 단계
        z: 네트워크가 예측한 변위(또는 속도 성분)
        R: 네트워크가 예측한 분산(Manager에서 상태별로 조정됨)
        """
        z = z.reshape(3, 1)
        
        # 1. 칼만 이득(Kalman Gain) 계산: K = PH'(HPH' + R)^-1
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 2. 상태 보정: x = x + K(z - Hx)
        innovation = z - (self.H @ self.x)
        self.x = self.x + K @ innovation
        
        # 3. 공분산 보정: P = (I - KH)P
        I = np.eye(self.x.shape[0])
        self.P = (I - K @ self.H) @ self.P
        
        return self.x, K