import cv2
import toml


class RgbCameraManager:
    def __init__(self, _toml_path):
        self.toml_path = _toml_path
        self._setting(self.toml_path)
        self.stopped = False
        self.is_grabbed = False
        self.frame = None

    def _setting(self, toml_path):
        dict_toml = toml.load(open(toml_path))['Rgb']
        self.device_id = dict_toml["device_id"]
        self.width = dict_toml["width"]
        self.height = dict_toml["height"]
        self.fps = dict_toml["fps"]
        self.cap = cv2.VideoCapture(self.device_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, int(self.fps))

    def update(self):
        # For latency related with buffer
        [self.cap.read() for i in range(2)]
        (self.is_grabbed, self.frame) = self.cap.read()
        return self.is_grabbed

    def read(self):
        return self.frame

    @property
    def grabbed(self):
        return self.is_grabbed
