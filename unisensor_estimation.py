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
    Panorama = False # generate panorama or not
    Estimate = False # the panorama is based on estimate or ground truth
    # dataset idx
    idx = 8
    # mat file location+prefix
    imu_prefix = "imu/imuRaw"
    vicon_prefix = "vicon/viconRot"
    cam_prefix = "cam/cam"

    # load data
    imu_ts, imu_vals, vicon_ts, vicon_euler = load_data(idx, imu_prefix, vicon_prefix)

    # init
    qk = np.array([1,0,0,0]) # last mean in quaternion

    time = imu_ts.shape[0]
    gyro_euler = np.zeros((time, 3))  # represent orientation in euler angles
    acc_estimate = np.zeros((time, 3))  # represent orientation in euler angles
    acc_truth = imu_vals[:,:3]
    g_q = np.array([0,0,0,1])

    for t in range(time):
        # extract sensor data
        gyro = imu_vals[t,3:]

        # Prediction
        if t == time-1: # last iter
            dt = np.mean(imu_ts[-10:] - imu_ts[-11:-1])
        else:
            dt = imu_ts[t+1] - imu_ts[t]
        qk = quat_multiply(qk,vec2quat(gyro*dt))

        # save for visualization
        gyro_euler[t, :] = taitbryan.quat2euler(qk)
        acc_estimate[t, :] = quat_multiply(quat_multiply(quat_inverse(qk), g_q), qk)[1:]


    # orientation plot gyro
    orientation_plot(idx, imu_ts, gyro_euler, vicon_ts, vicon_euler)

    # measurement plot acc
    measurement_plot(idx, imu_ts, acc_estimate, acc_truth)

    # panoramic by image stitching
    if Panorama:
        if Estimate:
            panorama(idx, cam_prefix, imu_ts, gyro_euler)
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

def orientation_plot(idx, imu_ts, gyro_euler, vicon_ts, vicon_euler):
    plt.figure(1)
    plt.subplot(3, 1, 1)
    true, = plt.plot(vicon_ts,  vicon_euler[:, 0], 'g', label='Ground Truth')
    gyro, = plt.plot(imu_ts, gyro_euler[:, 0], 'b', label='Gyro Estimate')
    plt.title('Z-Yaw')
    plt.ylabel('Angle [rad]')
    plt.legend(handles=[true, gyro])

    plt.subplot(3, 1, 2)
    true, = plt.plot(vicon_ts, vicon_euler[:, 1], 'g', label='Ground Truth')
    gyro, = plt.plot(imu_ts, gyro_euler[:, 1], 'b', label='Gyro Estimate')
    plt.title('Y-Pitch')
    plt.ylabel('Angle [rad]')
    plt.legend(handles=[true, gyro])

    plt.subplot(3, 1, 3)
    true, = plt.plot(vicon_ts, vicon_euler[:, 2], 'g', label='Ground Truth')
    gyro, = plt.plot(imu_ts, gyro_euler[:, 2], 'b', label='Gyro Estimate')
    plt.title('X-Roll')
    plt.ylabel('Angle [rad]')
    plt.legend(handles=[true, gyro])

    plt.savefig('result/uni_orientation'+str(idx)+'.png')

def measurement_plot(idx, imu_ts, acc_estimate, acc_truth):
    plt.figure(2)
    plt.subplot(3, 1, 1)
    true, = plt.plot(imu_ts,  acc_truth[:, 0], 'g', label='acc truth')
    est, = plt.plot(imu_ts, acc_estimate[:, 0], 'b', label='acc estimate')
    plt.title('X')
    plt.ylabel('g/s')
    plt.legend(handles=[true,est])

    plt.subplot(3, 1, 2)
    true, = plt.plot(imu_ts, acc_truth[:, 1], 'g', label='acc truth')
    est, = plt.plot(imu_ts, acc_estimate[:, 1], 'b', label='acc estimate')
    plt.title('Y')
    plt.ylabel('g/s')
    plt.legend(handles=[true,est])

    plt.subplot(3, 1, 3)
    true, = plt.plot(imu_ts, acc_truth[:, 2], 'g', label='acc truth')
    est, = plt.plot(imu_ts, acc_estimate[:, 2], 'b', label='acc estimate')
    plt.title('Z')
    plt.ylabel('g/s')
    plt.legend(handles=[true,est])

    plt.savefig('result/uni_measurement'+str(idx)+'.png')

if __name__ == '__main__':
    main()
