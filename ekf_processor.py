import numpy as np

def hat(v):
    """Skew-symmetric matrix from a 3D vector."""
    v = v.flatten()
    return np.array([[0, -v[2], v[1]], 
                     [v[2], 0, -v[0]], 
                     [-v[1], v[0], 0]])

def mat_exp(omega):
    """Exponential map for SO(3)."""
    omega = omega.flatten()
    angle = np.linalg.norm(omega)
    if angle < 1e-10:
        return np.eye(3) + hat(omega)
    axis = omega / angle
    s = np.sin(angle)
    c = np.cos(angle)
    return c * np.eye(3) + (1 - c) * np.outer(axis, axis) + s * hat(axis)

class TLIO_EKF:
    """
    15-state Error-State EKF (ESEKF) mirror of TLIO architecture.
    States: Position (3), Velocity (3), Rotation (3x3), Gyro Bias (3), Accel Bias (3).
    Error States (dX): d_theta (3), d_v (3), d_p (3), d_bg (3), d_ba (3).
    """
    def __init__(self, g_norm=9.81, sigma_na=0.1, sigma_ng=0.01, ita_ba=1e-4, ita_bg=1e-6):
        # Nominal state
        self.p = np.zeros((3, 1))
        self.v = np.zeros((3, 1))
        self.R = np.eye(3)
        self.bg = np.zeros((3, 1))
        self.ba = np.zeros((3, 1))
        
        # Error-state covariance (15x15)
        self.P = np.eye(15) * 0.1
        
        # Constants
        self.g = np.array([[0], [0], [-g_norm]])
        
        # Noise parameters
        self.sigma_na = sigma_na  # accel noise (standard deviation)
        self.sigma_ng = sigma_ng  # gyro noise (standard deviation)
        self.ita_ba = ita_ba      # accel bias random walk
        self.ita_bg = ita_bg      # gyro bias random walk

    def initialize_orientation(self, acc_init):
        """
        Initialize the rotation matrix R based on the initial gravity vector.
        Aligns the World Z-axis with the measured gravity direction.
        """
        acc_init = np.array(acc_init).flatten()
        # gravity in sensor frame: g_s = -R.T @ World_G
        # Here we do a simple alignment:
        # z_axis (body) = acc_init / norm(acc_init)
        z_b = acc_init / np.linalg.norm(acc_init)
        
        # Pick an arbitrary x_b that is not parallel to z_b
        if abs(z_b[0]) < 0.9:
            x_b = np.array([1, 0, 0])
        else:
            x_b = np.array([0, 1, 0])
            
        y_b = np.cross(z_b, x_b)
        y_b /= np.linalg.norm(y_b)
        x_b = np.cross(y_b, z_b)
        x_b /= np.linalg.norm(x_b)
        
        # R maps Body to World. 
        # Since acc_init is pointing 'up' in sensory space (assuming gravity subtraction hasn't happened yet),
        # we align it with positive World Z? 
        # Wait, if phone is on table, acc = [0, 0, 9.8], which is +Z in body.
        # We want this to be +Z in World too.
        self.R = np.column_stack([x_b, y_b, z_b])
        print("EKF Orientation initialized from gravity.")

    def predict(self, dt, gyr, acc):
        """
        IMU Propagation step.
        gyr: list or array [gx, gy, gz]
        acc: list or array [ax, ay, az]
        """
        gyr = np.array(gyr).reshape(3, 1)
        acc = np.array(acc).reshape(3, 1)
        
        # 1. Nominal state propagation
        unbiased_gyr = gyr - self.bg
        unbiased_acc = acc - self.ba
        
        dR = mat_exp(unbiased_gyr * dt)
        
        # Save old values for covariance propagation
        R_old = self.R.copy()
        v_old = self.v.copy()
        
        # Propagate nominal state
        self.R = self.R @ dR
        acc_w = R_old @ unbiased_acc + self.g
        self.v = self.v + acc_w * dt
        self.p = self.p + v_old * dt + 0.5 * acc_w * (dt**2)
        
        # 2. Covariance propagation (Error-state transition matrix F)
        # Error state order: d_theta (0:3), d_v (3:6), d_p (6:9), d_bg (9:12), d_ba (12:15)
        F = np.eye(15)
        F[3:6, 0:3] = -R_old @ hat(unbiased_acc) * dt
        F[6:9, 0:3] = -0.5 * R_old @ hat(unbiased_acc) * (dt**2)
        F[6:9, 3:6] = np.eye(3) * dt
        # F[0:3, 9:12] = -self.R * dt # Jacobian w.r.t bg
        F[0:3, 9:12] = -R_old * dt # Simpler Jacobian
        F[3:6, 12:15] = -R_old * dt
        F[6:9, 12:15] = -0.5 * R_old * (dt**2)
        
        # System noise Q
        Q = np.zeros((15, 15))
        var_g = (self.sigma_ng**2) * dt
        var_a = (self.sigma_na**2) * dt
        var_bg = (self.ita_bg**2) * dt
        var_ba = (self.ita_ba**2) * dt
        
        Q[0:3, 0:3] = np.eye(3) * var_g
        Q[3:6, 3:6] = np.eye(3) * var_a
        Q[9:12, 9:12] = np.eye(3) * var_bg
        Q[12:15, 12:15] = np.eye(3) * var_ba
        
        self.P = F @ self.P @ F.T + Q

    def update(self, z, R_meas):
        """
        Measurement update using AI predicted displacement.
        In this project, 'z' is often the accumulated displacement from AI.
        """
        z = z.reshape(3, 1)
        
        # Measurement matrix H for position (observed by AI)
        H = np.zeros((3, 15))
        H[:, 6:9] = np.eye(3) # Position error state
        
        S = H @ self.P @ H.T + R_meas
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Innovation
        innovation = z - self.p
        dX = K @ innovation
        
        # Apply correction to nominal state
        self.p = self.p + dX[6:9]
        self.v = self.v + dX[3:6]
        self.R = mat_exp(dX[0:3]) @ self.R
        self.bg = self.bg + dX[9:12]
        self.ba = self.ba + dX[12:15]
        
        # Update covariance
        I = np.eye(15)
        self.P = (I - K @ H) @ self.P
        
        return self.get_state(), K

    def get_state(self):
        """Returns the 6D state for visualizer compatibility (Pos, Vel)"""
        state = np.zeros((6, 1))
        state[0:3] = self.p
        state[3:6] = self.v
        return state