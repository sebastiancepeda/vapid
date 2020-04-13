"""
@author: Sebastian Cepeda
@email: sebastian.cepeda.fuentealba@gmail.com
"""
import numpy as np

from src.computer_vision.line import line_from_2_points


def test_line_from_2_points():
    line = np.array((1, -1, 1))
    a, b, c = line
    x1, x2 = 0, 1
    y1 = -(c + a * x1) / b
    y2 = -(c + a * x2) / b

    det_line = line_from_2_points(x1, y1, x2, y2)
    assert all(np.isclose(line, det_line)) or all(np.isclose(-line, det_line))

    line = np.array((1, -1, 0.5))
    a, b, c = line
    y1 = -(c + a * x1) / b
    y2 = -(c + a * x2) / b

    det_line = line_from_2_points(x1, y1, x2, y2)
    assert all(np.isclose(line, det_line)) or all(np.isclose(-line, det_line))
