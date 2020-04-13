"""
@author: Sebastian Cepeda
@email: sebastian.cepeda.fuentealba@gmail.com
"""
import numpy as np


def line_from_2_points(x1, y1, x2, y2):
    """
    Gets the coefficients of a line, given 2 points

    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :return:
    """
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    return a, b, c


def segment_from_line_equation(line, image_shape):
    """
    Creates a segment, covering the size of a whole image, from a line
    coefficients

    :param line: Lines coefficients
    :param image_shape: Shape of the image
    :return: Segment of line
    """
    y_max, x_max = image_shape[0:2]
    a, b, c = line
    points = []
    if not np.isclose(b, 0):
        x = 0
        y = int(-c / b)
        if -y_max < y < 2 * y_max:
            points.append((x, y))
        x = x_max
        y = int(-(c + a * x_max) / b)
        if -y_max < y < 2 * y_max:
            points.append((x, y))
    if not np.isclose(a, 0):
        y = 0
        x = int(-c / a)
        if -x_max < x < 2 * x_max:
            points.append((x, y))
        y = y_max
        x = int(-(c + b * y_max) / a)
        if -x_max < x < 2 * x_max:
            points.append((x, y))
    assert len(points) >= 2
    x1, y1 = points[0]
    x2, y2 = points[1]
    return x1, y1, x2, y2
