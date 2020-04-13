"""
@author: Sebastian Cepeda
@email: sebastian.cepeda.fuentealba@gmail.com
"""
import numpy as np


def vp_from_two_lines(l1, l2):
    """
    Gets the vanishing point from two lines

    :param l1:
    :param l2:
    :return:
    """
    vp = np.cross(l1, l2)
    x, y, z = vp
    if not np.isclose(z, 0.0, rtol=10.0 ** -10.0, atol=10.0 ** -10.0):
        x = int(x / z)
        y = int(y / z)
        vp = (x, y, 1.0)
    return vp


def vps_from_lines(lines):
    """
    Gets the vanishing points from a list of the detected lines

    :param lines: Detected lines
    :return: Detected vanishing points
    """
    vps = []
    vp_lines_map = {}
    for l1 in lines:
        for l2 in lines:
            vp = vp_from_two_lines(l1, l2)
            x, y, z = vp
            if not np.isclose(z, 0.0, rtol=10.0 ** -10.0, atol=10.0 ** -10.0):
                vp = (x, y)
                vps.append(vp)
                if vp not in vp_lines_map:
                    vp_lines_map[vp] = set()
                vp_lines_map[vp].add(l1)
                vp_lines_map[vp].add(l2)
    return vps, vp_lines_map
