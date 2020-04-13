"""
@author: Sebastian Cepeda
@email: sebastian.cepeda.fuentealba@gmail.com
"""

from src.computer_vision.point import distance


class SegmentFilter:

    def __init__(self,
                 distance_threshold_ratio: float = 0.10):
        """
        Constructor of SegmentFilter

        :param distance_threshold_ratio:
        """
        self.distance_threshold_ratio = distance_threshold_ratio

    def execute(self,
                detected_segments: list,
                image_shape: tuple,
                ):
        """
        Filters the detected segments by distance

        :param detected_segments: Detected segments
        :param image_shape: Image shape
        :return: Filtered segments
        """
        filtered_segments = []
        mean_image_lenght = (image_shape[0] + image_shape[1]) / 2
        distance_threshold = self.distance_threshold_ratio * mean_image_lenght
        for segment in detected_segments:
            x1, y1, x2, y2 = segment[0]
            p1 = x1, y1
            p2 = x2, y2
            segment_length = distance(p1, p2)
            if segment_length > distance_threshold:
                filtered_segments.append(segment)
        return filtered_segments
