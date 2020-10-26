from zense_camera_param import CameraParam
import numpy as np
import sys
import pdb


class PixelProjectorRGBDepth():
    def __init__(
        self, depth_camera_param, rgb_camera_param, transform_matrix_d2c
    ):
        self.tfm_mat_d2c = transform_matrix_d2c
        self.tfm_mat_c2d = np.linalg.inv(self.tfm_mat_d2c)
        self.K = rgb_camera_param.intrinsic_matrix

        self.depth_camera_param = depth_camera_param
        self.rgb_camera_param = rgb_camera_param

        self.depth_image_width = depth_camera_param.width
        self.depth_image_height = depth_camera_param.height
        self.rgb_image_width = rgb_camera_param.width
        self.rgb_image_height = rgb_camera_param.height

    def transform_points_between_rgb_d(self, points):
        points_extd = np.c_[points, np.repeat(1.0, points.shape[0])]
        points_tfm = points_extd.dot(self.tfm_mat_d2c.T)[:, :3]
        return points_tfm

    def _depth2point_between_rgb_d(self, depth, depth_intrinsic, depth_max_threshold=2000, depth_min_threshold=1e-3):
        """Generate point clouds from depth information for color<->depth mapping
        Args :
            depth(ndarray, CV_16UC1): depth image
            depth_intrinsic(ndarray 1x12) : flattened 3x4 intrinsic matrix
                                            np.array([fx ,0 ,cx  0], [0, fx, cx, 0], [0, 0, 1, 0])
        """
        arr_y = np.arange(depth.shape[0], dtype=np.float32)
        arr_x = np.arange(depth.shape[1], dtype=np.float32)

        val_x, val_y = np.meshgrid(arr_x, arr_y)

        tmp_z = depth
        # depth_intrinsic[0]: fx, depth_intrinsic[2]: cx
        tmp_x = tmp_z * \
            (val_x - depth_intrinsic[2]) * (1.0 / depth_intrinsic[0])
        # depth_intrinsic[5]: fy, depth_intrinsic[6]: cy
        tmp_y = tmp_z * \
            (val_y - depth_intrinsic[6]) * (1.0 / depth_intrinsic[5])

        filled = (depth > depth_min_threshold) * (depth < depth_max_threshold)

        filled_x = tmp_x[filled]
        filled_y = tmp_y[filled]
        filled_z = tmp_z[filled]
        point_list = np.stack([filled_x, filled_y, filled_z], axis=-1)
        return point_list

    def get_projection_to_image_idxs(self, points_3d_xyz):
        # points are obey [x':horisontal, y':vertical, z':depth] coordinate
        points_3d_tfm = self.transform_points_between_rgb_d(points_3d_xyz)
        _points_proj = points_3d_tfm.copy()
        _points_proj = np.array(
            [_points_proj[:, i] / _points_proj[:, 2] for i in range(3)]
        ).reshape((3, -1)).T
        projected_2d_idxs = _points_proj.dot(self.K.T)[:, :2].astype(np.int32)
        projected_2d_idxs_raw = projected_2d_idxs.copy()
        self.clipping(projected_2d_idxs)
        return projected_2d_idxs, projected_2d_idxs_raw

    def clipping(self, projected_2d_idxs):
        under_width_elem = projected_2d_idxs[:, 0] < 0
        over_width_elem = projected_2d_idxs[:, 0] >= self.rgb_image_width
        under_height_elem = projected_2d_idxs[:, 1] < 0
        over_height_elem = projected_2d_idxs[:, 1] >= self.rgb_image_height
        projected_2d_idxs[under_width_elem, 0] = 0
        projected_2d_idxs[over_width_elem, 0] = self.rgb_image_width - 1
        projected_2d_idxs[under_height_elem, 1] = 0
        projected_2d_idxs[over_height_elem, 1] = self.rgb_image_height - 1

    def transform_single_point(self, point_x, point_y, point_z, extrinsic_matrix):
        point = [[point_x, point_y, point_z]]
        point_extd = np.c_[point, 1]
        point_tfm = point_extd.dot(extrinsic_matrix.T)[:, :3]
        return point_tfm

    def get_projected_points_depth_to_color(self, pcd_ary_xyz, scale=1000):
        proj_idx, proj_idx_without_clip = self.get_projection_to_image_idxs(pcd_ary_xyz)
        dimg_projected = np.zeros(
            (self.rgb_camera_param.height, self.rgb_camera_param.width),
            dtype=np.uint16
        )
        dimg_projected[proj_idx[:, 1], proj_idx[:, 0]] = (pcd_ary_xyz[:, 2]*scale)
        dimg_projected = dimg_projected.astype(np.uint16)
        return dimg_projected, proj_idx, proj_idx_without_clip
