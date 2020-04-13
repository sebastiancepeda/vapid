"""
@author: Sebastian Cepeda
@email: sebastian.cepeda.fuentealba@gmail.com
"""
import numpy as np

from src.computer_vision.vanishing_point import vp_from_two_lines


def test_vp_from_two_lines():
    line_1 = np.array((-1, 0, 5))  # x=5
    line_2 = np.array((0, -1, 5))  # y=5

    vp = vp_from_two_lines(line_1, line_2)
    expected_vp = np.array((5, 5, 1))
    assert all(np.isclose(vp, expected_vp))

    line_1 = np.array((-1, 0, 5)) * 2  # x=5
    line_2 = np.array((0, -1, 5)) * 2  # y=5

    vp = vp_from_two_lines(line_1, line_2)
    expected_vp = np.array((5, 5, 1))
    assert all(np.isclose(vp, expected_vp))

    line_1 = np.array((0, -1, 5))  # y=5
    line_2 = np.array((0, -1, 10))  # y=10

    vp = vp_from_two_lines(line_1, line_2)
    assert np.isclose(vp[2], 0)
