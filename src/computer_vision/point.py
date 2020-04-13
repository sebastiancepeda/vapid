"""
@author: Sebastian Cepeda
@email: sebastian.cepeda.fuentealba@gmail.com
"""

import numpy as np


def distance(p1, p2):
    """
    Distance between two points

    :param p1: First point
    :param p2: Second point
    :return: Distance
    """
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    segment_length = np.sqrt(dx * dx + dy * dy)
    return segment_length
