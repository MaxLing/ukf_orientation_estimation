import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from ukf_functions import *
from panorama import *
from transforms3d import taitbryan

def main():
    ''' modify this part accordingly '''
    # flags
    UKF = True # use UKF or just gyro data for estimate
    Panorama = True # generate panorama or not
    Estimate = False # the panorama is based on estimate or ground truth
    # dataset idx
    idx = 8
    # mat file location+prefix
    imu_prefix = "imu/imuRaw"
    vicon_prefix = "vicon/viconRot"
    cam_prefix = "cam/cam"

    # load data
    imu_ts, imu_vals, vicon_ts, vicon_euler = load_data(idx, imu_prefix, vicon_prefix)

    # Unscented Kalman Filter
    # init
    qk = np.array([1,0,0,0]) # last mean in quaternion
    Pk = np.identity(3) * 0.1 # last cov in vector
    Q = np.identity(3) * 2 # process noise cov
    R = np.identity(3) * 2 # measurement noise cov

    time = imu_ts.shape[0]
    ukf_euler = np.zeros((time, 3))  # represent orientation in euler angles

    for t in range(time):
        # extract sensor data
        acc = imu_vals[t,:3]
        gyro = imu_vals[t,3:]

        # Prediction
        X = compute_sigma_pts(qk, Pk, Q)

        if t == time-1: # last iter
            dt = np.mean(imu_ts[-10:] - imu_ts[-11:-1])
        else:
            dt = imu_ts[t+1] - imu_ts[t]
        Y = process_model(X, gyro, dt)

        q_pred, P_pred, W = prediction(Y, qk)

        if UKF:
            # Measurement
            vk, Pvv, Pxz = measurement_model(Y, acc, W, R)

            # Update
            K = np.dot(Pxz,np.linalg.inv(Pvv)) # Kalman gain
            qk, Pk = update(q_pred, P_pred, vk, Pvv, K)

        else:
            # estimate just based on control input gyro
            qk, Pk = q_pred, P_pred

        # save for visualization
        ukf_euler[t, :] = taitbryan.quat2euler(qk)

    np.save('result/'+'ukf'+str(idx),ukf_euler)

    # orientation plot UKF + only gyro
    orientation_plot(idx, imu_ts, ukf_euler, vicon_ts, vicon_euler)

    # panoramic by image stitching
    if Panorama:
        if Estimate:
            panorama(idx, cam_prefix, imu_ts, ukf_euler)
        else:
            panorama(idx, cam_prefix, vicon_ts, vicon_euler)
    return 0

def load_data(idx, imu_prefix, vicon_prefix):
    # load data from imu/vicon, calibrate and save

    # load imu
    imu = io.loadmat(imu_prefix+str(idx)+".mat")
    imu_vals = np.array(imu['vals'])
    imu_ts = np.array(imu['ts']).T

    # scale and bias based on IMU reference
    acc_x = -imu_vals[0,:]
    acc_y = -imu_vals[1,:]
    acc_z = imu_vals[2,:]
    acc = np.array([acc_x, acc_y, acc_z]).T

    Vref = 3300
    acc_sensitivity = 330
    acc_scale_factor = Vref/1023/acc_sensitivity
    acc_bias = np.mean(acc[:10], axis = 0) - (np.array([0,0,1])/acc_scale_factor)
    acc = (acc-acc_bias)*acc_scale_factor

    gyro_x = imu_vals[4,:]
    gyro_y = imu_vals[5,:]
    gyro_z = imu_vals[3,:]
    gyro = np.array([gyro_x, gyro_y, gyro_z]).T

    gyro_sensitivity = 3.33
    gyro_scale_factor = Vref/1023/gyro_sensitivity
    gyro_bias = np.mean(gyro[:10], axis = 0)
    gyro = (gyro-gyro_bias)*gyro_scale_factor*(np.pi/180)

    imu_vals = np.hstack((acc,gyro))

    # load vicon
    vicon = io.loadmat(vicon_prefix+str(idx)+".mat")
    vicon_vals = np.array(vicon['rots'])
    vicon_ts = np.array(vicon['ts']).T

    n = np.shape(vicon_vals)[2]
    vicon_euler = np.zeros((n,3))
    for i in range(n):
        R = vicon_vals[:,:,i]
        vicon_euler[i] = taitbryan.mat2euler(R)

    return imu_ts, imu_vals, vicon_ts, vicon_euler

def orientation_plot(idx, imu_ts, ukf_euler, vicon_ts, vicon_euler):
    # plot YPR time series of estimate and ground truth
    plt.figure(1)
    plt.subplot(3, 1, 1)
    true, = plt.plot(vicon_ts,  vicon_euler[:, 0], 'g', label='Ground Truth')
    ukf, = plt.plot(imu_ts, ukf_euler[:, 0], 'r', label='UKF Estimate')
    plt.title('Z-Yaw')
    plt.ylabel('Angle [rad]')
    plt.legend(handles=[true, ukf])

    plt.subplot(3, 1, 2)
    true, = plt.plot(vicon_ts, vicon_euler[:, 1], 'g', label='Ground Truth')
    ukf, = plt.plot(imu_ts, ukf_euler[:, 1], 'r', label='UKF Estimate')
    plt.title('Y-Pitch')
    plt.ylabel('Angle [rad]')
    plt.legend(handles=[true, ukf])

    plt.subplot(3, 1, 3)
    true, = plt.plot(vicon_ts, vicon_euler[:, 2], 'g', label='Ground Truth')
    ukf, = plt.plot(imu_ts, ukf_euler[:, 2], 'r', label='UKF Estimate')
    plt.title('X-Roll')
    plt.ylabel('Angle [rad]')
    plt.legend(handles=[true, ukf])

    plt.savefig('result/orientation'+str(idx)+'.png')

if __name__ == '__main__':
    main()
