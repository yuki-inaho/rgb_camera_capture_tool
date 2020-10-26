import cv2
import numpy as np
from zense_camera_param import CameraParam
import open3d as o3d

DOWNSAMPLE_VOXEL_SIZE = 0.005


def cvt_numpy2open3d(pcl, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl.astype(np.float64))
    if color is not None:
        pcd.paint_uniform_color(color)
    return pcd


def cvt_depth2pcl(depth_img, camera_param, pcl_cutoff_boundary=(0.001, 2.0)):
    cx, cy = camera_param.center
    fx, fy = camera_param.focal
    height, width = depth_img.shape

    arr_y = np.arange(height, dtype=np.float32)
    arr_x = np.arange(width, dtype=np.float32)

    val_x, val_y = np.meshgrid(arr_x, arr_y)

    tmp_x = depth_img * (val_x - cx) * (1. / fx)
    tmp_y = depth_img * (val_y - cy) * (1. / fy)
    tmp_z = depth_img

    filled = (depth_img > pcl_cutoff_boundary[0]) * (depth_img < pcl_cutoff_boundary[1])

    filled_x = tmp_x[filled]
    filled_y = tmp_y[filled]
    filled_z = tmp_z[filled]

    _pcd = np.stack([filled_x, filled_y, filled_z], axis=-1)
    pcd = _pcd.reshape(-1,3)

    # Get XY-index image space corresponded with each valid 3D point
    filled_index = np.arange(height*width)[filled.reshape(-1)]
    y_index = filled_index // width
    x_index = filled_index % width
    image2pcd_index = np.c_[x_index, y_index]

    return pcd, image2pcd_index

def colorize_depth(image_depth, max_var):
    '''
    Convert 16bit single-channel depth image to color scaled image
    '''
    w = image_depth.shape[1]
    h = image_depth.shape[0]

    img_depth_colorized = np.zeros([h, w, 3]).astype(np.uint8)
    img_depth_colorized[:, :, 1] = 255
    img_depth_colorized[:, :, 2] = 255

    # > 2m data is removed
    _img_depth_zense_hue = image_depth.copy().astype(np.float32)
    _img_depth_zense_hue[np.where(_img_depth_zense_hue > max_var)] = 0
    zero_idx = np.where((_img_depth_zense_hue > max_var)
                        | (_img_depth_zense_hue == 0))
    _img_depth_zense_hue *= 255.0 / max_var

    img_depth_colorized[:, :, 0] = _img_depth_zense_hue.astype(np.uint8)
    img_depth_colorized = cv2.cvtColor(img_depth_colorized, cv2.COLOR_HSV2RGB)
    img_depth_colorized[zero_idx[0], zero_idx[1], :] = 0
    return img_depth_colorized