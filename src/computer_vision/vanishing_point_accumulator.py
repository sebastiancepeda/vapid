"""
@author: Sebastian Cepeda
@email: sebastian.cepeda.fuentealba@gmail.com
"""

import numpy as np

from src.computer_vision.point import distance


def _get_vps_accumulator(dx, dy, vps):
    accumulator_count = dict()
    accumulator_points = dict()
    max_accumulator = 0
    for vp in vps:
        (x, y) = vp
        x2 = int(np.floor(x / dx))
        y2 = int(np.floor(y / dy))
        cell = (x2, y2)
        if cell in accumulator_count:
            accumulator_count[cell] = accumulator_count[cell] + 1
            accumulator_points[cell].append(vp)
        else:
            accumulator_count[cell] = 1
            accumulator_points[cell] = [vp]
        if accumulator_count[cell] > max_accumulator:
            max_accumulator = accumulator_count[cell]
    return accumulator_count, accumulator_points, max_accumulator


def _average_vps(vps):
    average_vps = []
    for points in vps:
        x, y = 0, 0
        points_len = len(points)
        for point in points:
            x += point[0] / points_len
            y += point[1] / points_len
        average_point = (int(x), int(y))
        average_vps.append(average_point)
    return average_vps


class VanishingPointAccumulatorFilter:

    def __init__(self,
                 accumulator_size: int = 30,
                 count_threshold: int = 4,
                 percentage_threshold: float = 0.8,
                 ):
        """
        Constructor of the VanishingPointAccumulatorFilter

        :param accumulator_size: Size of the matrix used as accumulator for
        vanishing points
        :param count_threshold: Threshold for the amount of vanishing points
        in a cell be considered as a valid vanishing point
        :param percentage_threshold: Threshold for the percentage of vanishing
        points with respect to the maximum in all the accumulator to be
        considered as a valid vanishing point
        """
        self.accumulator_size = accumulator_size
        self.count_threshold = count_threshold
        self.percentage_threshold = percentage_threshold

    def _accumulator_filter(self, accumulator, count_threshold, max_acc,
                            percentage_threshold, points_map):
        vps = list()
        for cell in accumulator.keys():
            condition_list = list()
            condition = accumulator[cell] >= count_threshold
            condition_list.append(condition)
            ratio = accumulator[cell] / max_acc
            condition = ratio >= percentage_threshold
            condition_list.append(condition)
            if all(condition_list):
                vp_points = points_map[cell]
                vps.append(vp_points)
        return vps

    def execute(self,
                image_shape: tuple,
                vps: list,
                vp_lines_map: list):
        """
        Creates an accumulator of vanishing points, filters the points using
        this accumulator and the averages the vanishing point of the winning
        cells of the accumulator

        :param image_shape: Image shape
        :param vps: List of detected vanishing points
        :param vp_lines_map: Map of lines that given origin to each vp
        :return: List of vanishing points and the lines that originate them
        """
        dy = image_shape[0] / self.accumulator_size
        dx = image_shape[1] / self.accumulator_size
        accumulator, points_map, max_acc = _get_vps_accumulator(dx, dy, vps)
        vps = self._accumulator_filter(accumulator, self.count_threshold,
                                       max_acc,
                                       self.percentage_threshold, points_map)
        lines_per_vp = []
        for vp_points in vps:
            lines = set()
            for point in vp_points:
                mapped_lines = vp_lines_map[point]
                lines = lines.union(mapped_lines)
            lines_list = list(lines)
            lines_per_vp.append(lines_list)
        vps = _average_vps(vps)
        vp_lines = zip(vps, lines_per_vp)
        # Filtering vps to close to each other
        distance_threshold = (image_shape[0] / self.accumulator_size) * 2
        vp_lines2 = []
        for vp_in, lines_in in vp_lines:
            insert_point = True
            for vp, _ in vp_lines2:
                if distance(vp_in, vp) < distance_threshold:
                    insert_point = False
                    break
            if insert_point:
                vp_lines2.append((vp_in, lines_in))
        return vp_lines2
