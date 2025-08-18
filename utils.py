
import os
from os.path import join as pjoin

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R


import torch 
import torch.nn as nn
from scipy.signal import butter, filtfilt
import gzip, pickle
from glob import glob



class strokeDetector(torch.nn.Module):

    def __init__(self, number_of_labels):
        super(strokeDetector, self).__init__()
        self.n_channel = 6
        self.n_filters = 128
        self.n_hidden = 128
        self.n_rnn_layers = 1
        self.acc_cnn = torch.nn.Conv1d(self.n_channel, self.n_filters, 25, stride=1, padding='same')
        # Making the LSTM bidirectional
        self.acc_rnn = nn.GRU(input_size=self.n_filters, hidden_size=self.n_hidden,
            num_layers=self.n_rnn_layers, batch_first=True, bidirectional=True)
        # Adjusting the input size of the Linear layer to 64 * 2 because LSTM is bidirectional
        self.acc_linear = nn.Linear(self.n_hidden * 2  , number_of_labels, bias=True)

    def forward(self, acc, h_prev=None):
        B, T, F = acc.shape
        # CNN processing
        cnn_output = self.acc_cnn(acc.permute(0, 2, 1)).permute(0, 2, 1)
        # RNN processing
        rnn_output, h = self.acc_rnn(cnn_output, h_prev)
        # Linear layer processing
        label_estimates = self.acc_linear(rnn_output)

        return label_estimates, h

def run_model(acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, mag_x, mag_y, mag_z, arm_model, h_prev=None):
    
    # Format the input (current model only uses the acc)
    acc = np.c_[acc_x.ravel().copy(), acc_y.ravel().copy(), acc_z.ravel().copy()]
    gyr = np.c_[gyr_x.ravel().copy(), gyr_y.ravel().copy(), gyr_z.ravel().copy()]
    mag = np.c_[mag_x.ravel().copy(), mag_y.ravel().copy(), mag_z.ravel().copy()]

    # Check for NaN values in accelerometer data
    if np.isnan(acc).any():
        print("NaN values found in accelerometer data, replacing with 0.")
        acc = np.nan_to_num(acc)

    if np.isnan(gyr).any():
        print("NaN values found in accelerometer data, replacing with 0.")
        gyr = np.nan_to_num(gyr)

    # Check for NaN values in magnetometer data
    if np.isnan(mag).any():
        print("NaN values found in magnetometer data, replacing with 0.")
        mag = np.nan_to_num(mag)

    # Normalize and clip the data
    acc = np.clip(acc / 8000., -4, 4)
    gyr = np.clip(gyr / 8000., -2, 2)
    # mag = np.clip(mag / 500., -2, 2)
    mag_norm = np.linalg.norm(mag, axis=1, keepdims=True)
    mag_norm[mag_norm == 0] = 1e-6  # Avoid division by zero
    mag = mag / mag_norm

    

    # Concatenate accelerometer and magnetometer data
    # acc_mag = mag #np.concatenate((acc, mag), axis=-1)
    acc_mag = np.concatenate((acc, mag), axis=-1)

    # Convert to tensor
    acc = torch.tensor(acc_mag.astype('float32')).unsqueeze(0)

    # Run the model
    label_estimates, h_prev = arm_model(acc, h_prev=h_prev)
    label_estimates = label_estimates.squeeze(0)
    label_estimates = label_estimates.detach().numpy()

    return label_estimates, h_prev

def get_pos_from_course_speed(course, speed, dt):
    """
    Compute the position from the course and speed.
    """
    x = np.zeros(len(course))
    y = np.zeros(len(course))
    for i in range(1, len(course)):
        x[i] = x[i-1] + speed[i] * dt * np.sin(np.radians(course[i]))
        y[i] = y[i-1] + speed[i] * dt * np.cos(np.radians(course[i]))
    pos = np.column_stack((x, y))
    return pos


def get_data(patient, add_gyro=False, scale=True):
    column_names = ['x', 'y', 'z', 't', 'p']
    acc = pd.read_csv(pjoin(patient, 'acc.csv'), skiprows=1, names=column_names)
    mag = pd.read_csv(pjoin(patient, 'mag.csv'), skiprows=1, names=column_names)
    gyro_path = pjoin(patient, 'gyro.csv')
    if not os.path.exists(gyro_path):
        gyro = None
    else:
        gyro = pd.read_csv(gyro_path, skiprows=1, names=column_names)
    try:
        label = pd.read_csv(pjoin(patient, 'new_swim_label.csv'), skiprows=1, names=['target'])
        if label.shape[0] != acc.shape[0]:
            # print('Warning: label length does not match acc length')
            min_length = min(label.shape[0], acc.shape[0], gyro.shape[0] if gyro is not None else acc.shape[0], mag.shape[0])
            label = label.iloc[-min_length:]
            acc = acc.iloc[-min_length:]
            gyro = gyro.iloc[-min_length:] if gyro is not None else None
            mag = mag.iloc[-min_length:]
            target = label['target'].values
            target[target==-1] = 2
    except FileNotFoundError:
        target = None
    length = acc.shape[0]
    acc = acc[['x', 'y', 'z']].values
    mag = mag[['x', 'y', 'z']].values

    if gyro is not None:
        gyro = gyro[['x', 'y', 'z']].values
    else:
        gyro = np.zeros((length, 3))


    # if any nan, replace with 0
    acc = np.nan_to_num(acc)
    gyro = np.nan_to_num(gyro)
    mag = np.nan_to_num(mag)

    if scale:
        grav        = 9.80665
        acc_scale   = 4096.0       # counts → g
        gyro_scale  = 16.4         # counts → deg / s
        mag_scale   = 10.0         # counts → μT  (10 mG = 1 μT)


        acc  = (acc  / acc_scale)  * grav                 # m/s²
        gyro = (gyro / gyro_scale) * np.pi / 180.0        # rad/s
        mag  =  mag / mag_scale                  # μT

    if add_gyro:
        X = np.concatenate([acc, gyro, mag], axis=1)
    else:
        X = np.concatenate([acc, mag], axis=1)  # (T, 6)

    X = X.astype(np.float32)

    return X, target


def get_orientation(acc, gyro, mag):

    args = {
        'sigma_g_sq' :  1e-4,
        'sigma_q_sq' : 1e-4,
        'ca'         : 0.01,
        'sigma_mag'  :1e-10,
    }
    Rot  = KF_orientation(acc, gyro, mag, args)

    rot_obj   = R.from_matrix(Rot)
    roll, pitch, yaw = rot_obj.as_euler('xyz', degrees=True).T
    yaw = smooth_yaw_deg(yaw, N=25*4)
    roll = smooth_yaw_deg(roll, N=25*4)
    pitch = smooth_yaw_deg(pitch, N=25*4)
    yaw = (yaw + 360.0) % 360.0 

    acc_rotated  = rot_obj.apply(acc)
    mag_rotated  = rot_obj.apply(mag)
    gyro_rotated = rot_obj.apply(gyro)

    acc_external_rotated = acc_rotated - np.array([0, 0, 9.81])  # subtract gravity
    # back to watch frame:
    acc_external = rot_obj.inv().apply(acc_external_rotated)

    return acc_external, acc_rotated, gyro_rotated, mag_rotated, yaw, pitch, roll



def plot_signal(acc, gyro, mag, target=None, prediction=None, yaw=None, pitch=None, roll=None, title='Signal', fs=25, t=None, savepath=None):
    label_colors = { 0: 'gray', 1: 'cyan', 2: 'yellow'}

    # Use provided t if available, else generate from fs
    if t is None:
        t = np.arange(acc.shape[0]) / fs
    else:
        t = np.asarray(t)
        if t.shape[0] != acc.shape[0]:
            raise ValueError("Length of t does not match acc data length")

    num_figs = 3
    if gyro is not None:
        num_figs += 1
    if yaw is not None:
        num_figs += 1
    if pitch is not None:
        num_figs += 1
    if roll is not None:
        num_figs += 1

    plt.figure(figsize=(20, num_figs*2.5))
    plt.suptitle(title)
    i = 0
    for data, title_str in [
        ([acc[:, 0], acc[:, 1], acc[:, 2]], 'Accelerometer'),
        ([gyro[:, 0], gyro[:, 1], gyro[:, 2]], 'Gyroscope') if gyro is not None else ([], 'Gyroscope (not available)'),
        ([mag[:, 0], mag[:, 1], mag[:, 2]], 'Magnetometer'),
        ([np.cos(yaw * np.pi / 180), np.sin(yaw * np.pi / 180)], 'yaw') if yaw is not None else ([], 'yaw'),
        ([np.cos(roll * np.pi / 180), np.sin(roll * np.pi / 180)], 'roll') if roll is not None else ([], 'roll'),
        ([np.cos(pitch * np.pi / 180), np.sin(pitch * np.pi / 180)], 'pitch') if pitch is not None else ([], 'pitch'),
    ]:
        if title_str == 'Gyroscope (not available)':
            continue
        if title_str == 'yaw' and yaw is None:
            continue
        if title_str == 'pitch' and pitch is None:
            continue
        if title_str == 'roll' and roll is None:
            continue

        ax = plt.subplot(num_figs, 1, i+1)
        i += 1
        for d, label in zip(data, ['cos (yaw)', 'sin (yaw)'] if title_str == 'yaw' else ['cos (roll)', 'sin (roll)'] if title_str == 'roll' else ['cos (pitch)', 'sin (pitch)'] if title_str == 'pitch' else ['x', 'y', 'z']):
            ax.plot(t, d, label=label)
        # Add label shadows
        if target is not None:
            prev = target[0]
            start = t[0]
            for idx, val in enumerate(target):
                if val != prev or idx == len(target)-1:
                    end = t[idx] if val != prev else t[-1]
                    if prev in label_colors:
                        ax.axvspan(start, end, color=label_colors[prev], alpha=0.15 )
                    start = t[idx]
                    prev = val
        plt.legend()
        ax.set_ylabel(title_str)
    ax = plt.subplot(num_figs, 1, num_figs)
    if target is not None:
        ax.plot(t, target, '.', markersize=5, label='Target', color='blue')
    if prediction is not None:
        ax.plot(t, prediction + 0.1, '.', markersize=5, label='Prediction', color='orange', alpha=0.5)
        ax.legend()
    ax.set_xlabel('t seconds \n Yellow: turn, Gray: rest, Cyan: swim')
    ax.set_ylabel('Target')
    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()


def KF_orientation(acc_org, gyro_org, mag_org, args):
    
    fs = args.get('fs', 25)
    grav        =args.get('grav', 9.80665)

    sigma_g_sq = args.get('sigma_g_sq', 1e-4)    # gyro noise
    sigma_q_sq = args.get('sigma_q_sq', 1e-4)    # acc noise
    ca         = args.get('ca', 0.01)           # acc bias %0.01;%ca=1:fully rely on gyroscope, ca=0:fully rely on accelerometer
    sigma_mag  = args.get('sigma_mag', 1e-10)    # mag noise


    acc = acc_org.copy()
    gyro = gyro_org.copy()
    mag = mag_org.copy()
    

    dt = 1.0 / fs
    n  = acc.shape[0]         # number of samples

    # ------------------------------------------------------------------
    # 0.  OPTIONAL  –  low‑pass or median filtering
    # ------------------------------------------------------------------
    # Example: 4th‑order Butterworth  fc = 10/3 Hz  (same as MATLAB block)

    # ------------------------------------------------------------------
    # 3.  Kalman filter initialisation
    # ------------------------------------------------------------------
    z_prev        = np.array([0, 0, 1], dtype=np.float32)   # gravity dir
    P_prev_z      = np.eye(3, dtype=np.float32)
    ex_acc        = np.zeros(3, dtype=np.float32)

    x_prev       = np.array([1, 0, 0], dtype=np.float32)   # body‑x row
    P_prev_x     = np.eye(3, dtype=np.float32)

    # Output arrays
    R_glob  = np.zeros((n, 3, 3), dtype=np.float32)   # lobal ENU frame
    acc_lin = np.zeros((n, 3), dtype=np.float32)        # external acc

    # Helper lambdas
    skew = lambda v: np.array([[   0, -v[2],  v[1]],
                                [ v[2],    0, -v[0]],
                                [-v[1],  v[0],    0]], dtype=np.float32)

    # ------------------------------------------------------------------
    # 4.  Main loop
    # ------------------------------------------------------------------
    for k in range(n):
        # -------- Roll & Pitch predict
        omega = gyro[k]
        deltaT = dt
        Phi_w2 = np.eye(3) - deltaT*skew(omega)
        z_pri  = Phi_w2 @ z_prev
        z_pri /= np.linalg.norm(z_pri)

        Q_z    = -(deltaT**2 * skew(z_prev)) @ (sigma_g_sq * skew(z_prev))

        P_pri_z  = Phi_w2 @ P_prev_z @ Phi_w2.T + Q_z

        # -------- Roll & Pitch update (acc)
        ex_acc =acc[k] - grav*z_prev
        acc_lin[k] = ex_acc
        z_meas = acc[k] - ca*ex_acc
        H_z    = grav * np.eye(3)
        sigma_acc = (1/3) * ca**2 * np.linalg.norm(ex_acc)**2
        R_z    = (sigma_acc + sigma_q_sq) * np.eye(3)

        K_z    = P_pri_z @ H_z.T @ np.linalg.inv(H_z @ P_pri_z @ H_z.T + R_z)
        z_prev = z_pri + K_z @ (z_meas - H_z @ z_pri)
        z_prev /= np.linalg.norm(z_prev)
        # print('z_prev', z_prev)
        P_prev_z = (np.eye(3) - K_z @ H_z) @ P_pri_z

        # External acceleration
        roll  =  np.arctan2(z_prev[1], z_prev[2])
        # print('roll meas', roll)
        pitch =  np.arctan2(-z_prev[0], np.sqrt(z_prev[1]**2 + z_prev[2]**2))
        # print('pitch meas', pitch)

        R_pitch = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                            [0,              1,             0],
                            [-np.sin(pitch), 0, np.cos(pitch)]])
        R_roll  = np.array([[1, 0,           0],
                            [0, np.cos(roll), -np.sin(roll)],
                            [0, np.sin(roll),  np.cos(roll)]])
        Rot_rp = R_pitch @ R_roll
                    
        

        # -------- Yaw predict (same gyro transition)
        x_pri  = Phi_w2 @ x_prev
        x_pri /= np.linalg.norm(x_pri)
        Q_x    = -(deltaT**2 * skew(x_prev)) @ (sigma_g_sq * skew(x_prev))
        P_pri_x  = Phi_w2 @ P_prev_x @ Phi_w2.T + Q_x

        # -------- Yaw update (mag)
        mag_rp   = Rot_rp @ mag[k]


        yaw_measured_rad = np.arctan2(mag_rp[1], mag_rp[0])  # NED (+Y=E, +X=N)
        # print('yaw_measured_rad', yaw_measured_rad)


        cy, sy  = np.cos(yaw_measured_rad), np.sin(yaw_measured_rad)
        cr, sr  = np.cos(roll), np.sin(roll)
        cp, sp  = np.cos(pitch), np.sin(pitch)

        x_meas = np.array([ cy*cp,
                            -sy*cp,
                        -sp ], dtype=np.float32)

        
        x_meas /= np.linalg.norm(x_meas)
        # print('x_meas', x_meas)
        R_x = sigma_mag * np.eye(3)
        K_x = P_pri_x @ np.linalg.inv(P_pri_x + R_x)
        x_prev = x_pri + K_x @ (x_meas - x_pri)
        x_prev /= np.linalg.norm(x_prev)
        # print('x_prev', x_prev)
        P_prev_x = (np.eye(3) - K_x) @ P_pri_x
        # Yaw update
        yaw = np.arctan2(x_prev[1], x_prev[0])
        
        R_yaw = np.array([[ np.cos(yaw), -np.sin(yaw), 0],
                  [ np.sin(yaw),  np.cos(yaw), 0],
                  [          0,            0, 1]], dtype=np.float32)
        
        R_glob[k] = R_yaw @ Rot_rp

    return R_glob


def smooth_yaw_deg(yaw_deg, N=5):
    # Convert degrees to radians
    yaw_rad = np.deg2rad(yaw_deg)

    # ⚠️ Unwrap BEFORE converting to (cos, sin)
    yaw_rad_unwrapped = np.unwrap(yaw_rad)

    # Convert to unit vectors
    vec = np.column_stack((np.cos(yaw_rad_unwrapped), np.sin(yaw_rad_unwrapped)))

    # Apply moving average filter (you can replace with any 1D filter)
    kernel = np.ones(N) / N
    vec_smooth = np.apply_along_axis(lambda c: np.convolve(c, kernel, mode='same'), axis=0, arr=vec)

    # Back to angle
    yaw_smooth_rad = np.arctan2(vec_smooth[:,1], vec_smooth[:,0])

    # Convert to degrees
    yaw_smooth_deg = np.rad2deg(yaw_smooth_rad)

    # Wrap result to [0, 360)
    yaw_smooth_deg = np.mod(yaw_smooth_deg, 360)

    return yaw_smooth_deg