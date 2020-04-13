"""
@author: Sebastian Cepeda
@email: sebastian.cepeda.fuentealba@gmail.com
"""

import os
import traceback
from os import listdir
from os.path import isfile, join

import cv2
from loguru import logger

from src.algorithms.vanishing_point_detector import VanishingPointsDetector, \
    draw_vps

# from src.utils.print_logger import PrintLogger as logger


def _create_folder(_out_path):
    try:
        os.mkdir(_out_path)
    except OSError as ex:
        pass


def _list_files(_folder):
    files = [f for f in listdir(_folder) if isfile(join(_folder, f))]
    return files


if __name__ == '__main__':
    input_path = "input/"
    output_path = "output/"

    image_files = _list_files(input_path)
    _create_folder(output_path)

    for image_file in image_files:
        try:
            image_path = f"{input_path}{image_file}"
            logger.info(f"Image path: {image_path}")
            out_path = f"{output_path}{image_file}"
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise Exception(f"Couldn't load image {image_path}")
            vp_detector = VanishingPointsDetector(out_path, debug_level=0,
                                                  _logger=logger)
            vp_lines = vp_detector.execute(image)
            detected_image = draw_vps(image, vp_lines)
            cv2.imwrite(f"{out_path}filtered_vps.png", detected_image)
        except Exception as e:
            t = traceback.format_exc()
            logger.info(f"Error: {e} \n {t}")
            raise e
