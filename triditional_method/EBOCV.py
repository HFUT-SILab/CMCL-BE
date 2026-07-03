import cv2
import numpy as np

from .tools import gabor2d


class EBOCV:
    def __init__(self, filter_size=(35, 35), filter_num=6, target_size=(128, 128)):
        self.filter_size = filter_size
        self.filter_num = filter_num
        self.target_size = target_size
        self.filters = np.zeros([*filter_size, filter_num], dtype=np.float64)
        self._create_gabor_filters()

    def _create_gabor_filters(self):
        for idx in range(self.filter_num):
            self.filters[:, :, idx] = gabor2d(*self.filter_size, 0.5, idx / self.filter_num * 180)

    def get_feature(self, image):
        if image.ndim > 2:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        image = np.float64(cv2.resize(image, self.target_size, interpolation=cv2.INTER_CUBIC))

        feature_maps = np.zeros([*self.target_size, self.filter_num], dtype=np.float64)
        for idx in range(self.filter_num):
            feature_maps[:, :, idx] = cv2.filter2D(
                image,
                -1,
                self.filters[:, :, idx],
                borderType=cv2.BORDER_REPLICATE,
            )

        bit_maps = np.zeros([*self.target_size, self.filter_num], dtype=np.float64)
        bit_maps[feature_maps > 0] = 1

        downsampled_binary = bit_maps[::4, ::4, :]
        downsampled_response = feature_maps[::4, ::4, :]

        ratio = 0.08
        total_elements = downsampled_response.size
        num_to_mark = int(total_elements * ratio)
        flat_indices = np.argsort(np.abs(downsampled_response), axis=None)

        enhanced_binary = np.zeros(
            [self.target_size[0] // 4, self.target_size[1] // 4, self.filter_num],
            dtype=np.float64,
        )
        enhanced_binary.flat[flat_indices[:num_to_mark]] = 1

        feature1 = downsampled_binary.transpose(2, 0, 1)
        feature2 = enhanced_binary.transpose(2, 0, 1)
        return np.concatenate((feature1, feature2), axis=0)
