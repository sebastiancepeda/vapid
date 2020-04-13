"""
@author: Sebastian Cepeda
@email: sebastian.cepeda.fuentealba@gmail.com
"""

import cv2
import numpy as np

from src.computer_vision.colors import RED_COLOR, GREEN_COLOR, BLUE_COLOR, \
    WHITE_COLOR
from src.computer_vision.gabor_bank import GaborBank
from src.computer_vision.line import line_from_2_points, \
    segment_from_line_equation
from src.computer_vision.segment_filter import SegmentFilter
from src.computer_vision.vanishing_point import vps_from_lines
from src.computer_vision.vanishing_point_accumulator import \
    VanishingPointAccumulatorFilter
from src.utils.print_logger import PrintLogger as logger


def _vp_lines_translation(dx, dy, vp_lines):
    vps_list = []
    lines_list = []
    for vp, lines in vp_lines:
        x, y = vp
        x += dx
        y += dy
        vp2 = (x, y)
        vps_list.append(vp2)
        lines2 = []
        for line in lines:
            a, b, c = line
            c2 = c - (a * dx + b * dy)
            line2 = a, b, c2
            lines2.append(line2)
        lines_list.append(lines2)
    vp_lines = list(zip(vps_list, lines_list))
    return vp_lines


def _are_vps_outside(vp_lines, x_max, y_max):
    vps_outside_image = False
    for vp, lines in vp_lines:
        x, y = vp
        if not (0 < x < x_max):
            vps_outside_image = True
        if not (0 < y < y_max):
            vps_outside_image = True
    return vps_outside_image


def draw_vps(image, vp_lines, draw_lines):
    """
    Draws detected vanishing points and lines in an image

    :param image: Image where to draw the vanishing points and lines
    :param vp_lines: Vanishing points and lines
    :return: Image with vanshing points and lines
    """
    image_shape = image.shape
    line_size = max(1, int(image_shape[0] * 0.001))
    x_max = image_shape[1]
    y_max = image_shape[0]
    vps_outside_image = _are_vps_outside(vp_lines, x_max, y_max)
    if vps_outside_image:
        dx = int(x_max / 2)
        dy = int(y_max / 2)
        image = cv2.copyMakeBorder(image, dy, dy, dx, dx,
                                   cv2.BORDER_CONSTANT, None,
                                   WHITE_COLOR)
        vp_lines = _vp_lines_translation(dx, dy, vp_lines)
    image_shape = image.shape[0:2]
    for vp, lines in vp_lines:
        if draw_lines:
            draw_vp_lines(image, image_shape, line_size, lines)
        draw_vp(image, line_size, vp)
        break
    return image


def draw_vp(image, line_size, vp):
    cv2.circle(image, vp, line_size * 10, BLUE_COLOR,
               line_size * 10)
    cv2.circle(image, vp, line_size * 5, GREEN_COLOR,
               line_size * 5)


def draw_vp_lines(image, image_shape, line_size, lines):
    for line in lines:
        x1, y1, x2, y2 = segment_from_line_equation(line, image_shape)
        p1 = (x1, y1)
        p2 = (x2, y2)
        cv2.line(image, p1, p2, GREEN_COLOR, line_size * 2)
        cv2.line(image, p1, p2, RED_COLOR, line_size)


class VanishingPointsDetector:

    def __init__(self,
                 out_path: str,
                 debug_level: int = 0,
                 _logger=logger):
        """
        Constructor of VanishingPointsDetector

        :param out_path: Output path
        :param debug_level: Level of debug info
        :param _logger: Logger
        """
        self.out_path = out_path
        self.debug_level = debug_level
        self.logger = _logger

    def _preprocessing(self, image, image_shape):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.debug_level >= 1:
            cv2.imwrite(f"{self.out_path}image_gray.png", image_gray)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_gray = clahe.apply(image_gray)
        if self.debug_level >= 1:
            cv2.imwrite(f"{self.out_path}image_clahe.png", image_gray)
        gabor_size = int(image_shape[0] * 0.01)
        gb = GaborBank(ksize=gabor_size, angle_steps=18, sigma=0.9, lambd=100,
                       gamma=0.1)
        image_gray = gb.execute(image_gray)
        if self.debug_level >= 1:
            cv2.imwrite(f"{self.out_path}gabor_bank.png", image_gray)
        gaussian_size = int(image_shape[0] * 0.0005)
        gaussian_size = gaussian_size + (1 - (gaussian_size % 2))
        gaussian_shape = (gaussian_size, gaussian_size)
        image_gray = cv2.GaussianBlur(image_gray, gaussian_shape, 0)
        return image_gray

    def _edge_detection(self, image_gray):
        edges = cv2.Canny(image_gray, 50, 200, apertureSize=3)
        if self.debug_level >= 1:
            cv2.imwrite(f"{self.out_path}image_edges.png", edges)
        return edges

    def _lines_detection(self, edges, image, image_shape):
        max_line_gap = max(1, int(image_shape[0] * 0.10))
        px_resolution = 1
        segments = cv2.HoughLinesP(edges, px_resolution, np.pi / 180, 100,
                                   maxLineGap=max_line_gap)
        if segments is None:
            segments = np.array([])
        # self.logger.info(f"Detected lines: {segments.shape[0]}")
        image_copy = image.copy()
        sf = SegmentFilter(distance_threshold_ratio=0.1)
        filtered_segments = sf.execute(segments, image_shape[0:2])
        for segment in filtered_segments:
            x1, y1, x2, y2 = segment[0]
            cv2.line(image_copy, (x1, y1), (x2, y2), GREEN_COLOR, 5)
            cv2.line(image_copy, (x1, y1), (x2, y2), RED_COLOR, 2)
        if self.debug_level >= 1:
            cv2.imwrite(f"{self.out_path}lines_detected.png", image_copy)
        lines = []
        for segment in filtered_segments:
            x1, y1, x2, y2 = segment[0]
            line = line_from_2_points(x1, y1, x2, y2)
            lines.append(line)
        return lines

    def _vps_detection(self, image_gray, image_shape, lines):
        vps, vp_lines_map = vps_from_lines(lines)
        # self.logger.info(f"Detected intersection points: {len(vps)}")
        detected_image = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
        line_size = max(1, int(image_shape[0] * 0.001))
        for vp in vps:
            cv2.circle(detected_image, vp, line_size * 15, BLUE_COLOR,
                       line_size * 15)
            cv2.circle(detected_image, vp, line_size * 10, GREEN_COLOR,
                       line_size * 10)
        if self.debug_level >= 1:
            cv2.imwrite(f"{self.out_path}detected_vps.png", detected_image)
        vp_filter = VanishingPointAccumulatorFilter(accumulator_size=30,
                                                    count_threshold=3,
                                                    percentage_threshold=0.6)
        vp_lines = vp_filter.execute(image_shape[0:2], vps, vp_lines_map)
        vp_lines = list(vp_lines)
        # self.logger.info(f"Detected vanishing points: {len(vp_lines)}")
        return vp_lines

    def execute(self, image):
        """
        Detects the vanishing points in an image and the lines that give
        origin to them

        :param image: Input image
        :return: Vanishing points and lines
        """
        image_shape = image.shape
        image_gray = self._preprocessing(image, image_shape)
        edges = self._edge_detection(image_gray)
        lines = self._lines_detection(edges, image, image_shape)
        vp_lines = self._vps_detection(image_gray, image_shape, lines)
        return vp_lines
