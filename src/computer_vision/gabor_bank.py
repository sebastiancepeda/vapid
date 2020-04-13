"""
@author: Sebastian Cepeda
@email: sebastian.cepeda.fuentealba@gmail.com
"""

import cv2
import numpy as np


class GaborBank:

    def __init__(self,
                 ksize: int = 101,
                 angle_steps: int = 18,
                 sigma: float = 0.9,
                 lambd: float = 100,
                 gamma: float = 0.1,
                 ):
        """
        Constructor of the Gabor filter bank
        :param ksize:
        :param angle_steps:
        :param sigma:
        :param lambd:
        :param gamma:
        """
        self.ksize = ksize
        self.angle_steps = angle_steps
        self.sigma = sigma
        self.lambd = lambd
        self.gamma = gamma

    def execute(self,
                image_gray: np.ndarray):
        """
        Passes the image through a gabor filter bank, taking the maximum over
        all the applied angles (theta).

        :param image_gray: Image to process
        :return: Processed image
        """
        params = {
            'ksize': (self.ksize, self.ksize),
            'sigma': self.sigma,
            'lambd': self.lambd,
            'gamma': self.gamma,
            'psi': 0,
            'ktype': cv2.CV_32F
        }
        gabor_bank_result = np.zeros_like(image_gray)
        for theta in np.arange(0, np.pi, np.pi / self.angle_steps):
            params['theta'] = theta
            kernel = cv2.getGaborKernel(**params)
            norm = kernel.sum()
            kernel /= norm
            filtered_image = cv2.filter2D(image_gray, cv2.CV_8U, kernel)
            np.maximum(gabor_bank_result, filtered_image, gabor_bank_result)
        return gabor_bank_result
