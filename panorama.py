import cv2
import numpy as np
from scipy import io
from transforms3d import taitbryan

def panorama(idx, cam_prefix, imu_ts, ukf_euler):
    # camera projection params
    Wfov = np.pi/3
    Hfov = np.pi/4
    frame = None

    # load cam data
    cam = io.loadmat(cam_prefix+str(idx)+".mat")
    cam_vals = np.array(cam['cam']) # m*n*3*k
    cam_ts = np.array(cam['ts']).T # k*1
    imu_idx = 0

    # init video writer
    outfile = 'result/panorama' + str(idx) + '.avi'
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    video = cv2.VideoWriter(filename=outfile, fourcc=fourcc, fps=20.0, frameSize=(1920, 960))

    for i in range(cam_ts.shape[0]):
        RGB = cam_vals[:,:,:,i]
        height, width, channel = RGB.shape
        cam_t = cam_ts[i]

        # generate spherical coordinate
        radius = 1
        azimuth = -(np.arange(width)/(width-1)-0.5)*Wfov
        altitude = -(np.arange(height)/(height-1)-0.5)*Hfov
        # negate because pixel origin is at top left
        azimuth, altitude = np.meshgrid(azimuth, altitude)

        # transform to cartesian coordinates on unit sphere
        X = radius * np.cos(altitude) * np.cos(azimuth)
        Y = radius * np.cos(altitude) * np.sin(azimuth)
        Z = radius * np.sin(altitude)
        C = np.dstack((X, Y, Z)) # (m,n,3)

        # rotate by the nearest timestamp
        if imu_idx > len(imu_ts)-1: # check boundary
            break
        while imu_ts[imu_idx]<cam_t and imu_idx < len(imu_ts)-1:
            imu_idx += 1
        if imu_ts[imu_idx]+imu_ts[imu_idx-1] > 2*cam_t: # choose closest
            imu_idx -= 1
        euler = ukf_euler[imu_idx]
        imu_idx += 1 # prevent replication

        R = taitbryan.euler2mat(euler[0],euler[1],euler[2])
        C = np.einsum('pr,mnr->mnp', R, C)

        # transform cartesian back to spherical
        X = C[:, :, 0]
        Y = C[:, :, 1]
        Z = C[:, :, 2]
        azimuth = -np.arctan2(Y, X)
        altitude = -np.arctan2(Z, np.sqrt(X ** 2 + Y ** 2)) # altitude range(-pi/2,pi/2)
        # negate back
        # radius doesn't change

        # project sphere to a plane, convert plane into pixel by scaling
        Px = ((azimuth+np.pi)/Wfov * width).astype(np.uint)
        Py = ((altitude+np.pi/2)/Hfov * height).astype(np.uint)
        # Py = np.flipud(Py.reshape((height, width))).reshape((1,-1))

        # paint your pixels on to image
        if frame is None: # init panorama after knowing image size and camera params
            frame = np.zeros((int(np.pi / Hfov * height), int(2 * np.pi / Wfov * width), 3), dtype=np.uint8)
        frame[Py,Px,:] = RGB

        # display animation
        cv2.imshow('Stitching Panorama', frame)
        cv2.waitKey(10)

        # write frame to video file
        video.write(frame)

    video.release()
    cv2.destroyAllWindows()

