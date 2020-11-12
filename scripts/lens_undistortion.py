import cv2
import toml
import numpy as np
from scripts.camera_parameter import Intrinsic
from pathlib import Path

class LensUndistorter:
    def __init__(self, toml_path):
        toml_dict = toml.load(open(toml_path))

        # Set Camera Parameters
        intrinsic_elems = ["fx", "fy", "cx", "cy"]
        self.intrinsic_params = Intrinsic()
        self.intrinsic_params.set_intrinsic_parameter(
            *[toml_dict["Rgb"][elem] for elem in intrinsic_elems]
        )
        self.intrinsic_params.set_image_size(
            *[toml_dict["Rgb"][elem] for elem in ["width", "height"]])
        K_rgb = np.array(
            [[self.intrinsic_params.fx, 0, self.intrinsic_params.cx],
             [0, self.intrinsic_params.fy, self.intrinsic_params.cy],
             [0, 0, 1]]
        )
        self.distortion_params = np.array(
            [toml_dict["Rgb"]["k{}".format(i+1)] for i in range(4)]
        )

        image_width = toml_dict["Rgb"]["width"]
        image_height = toml_dict["Rgb"]["height"]
        self.DIM = (image_width, image_height)

        self.K_rgb_raw = K_rgb
        self.K_rgb = cv2.getOptimalNewCameraMatrix(
            self.K_rgb_raw, self.distortion_params,
            self.DIM, 0
        )[0]

        _map1, _map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K_rgb_raw, self.distortion_params, np.eye(3),
            self.K_rgb, self.DIM, cv2.CV_16SC2
        )
        self.map1 = _map1
        self.map2 = _map2
        self.P_rgb = (self.K_rgb[0][0], 0., self.K_rgb[0][2], 0.,
                      0., self.K_rgb[1][1], self.K_rgb[1][2], 0.,
                      0., 0., 1., 0.)

    def correction(self, image):
        return cv2.remap(image, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    @property
    def K(self):
        return self.K_rgb

    @property
    def P(self):
        return self.P_rgb
