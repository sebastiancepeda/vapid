"""
@author: Sebastian Cepeda
@email: sebastian.cepeda.fuentealba@gmail.com
"""

import os
import traceback
from os import listdir
from os.path import isfile, join

import cv2

from src.algorithms.vanishing_point_detector import VanishingPointsDetector, \
    draw_vps, draw_vp


def _create_folder(_out_path):
    try:
        os.mkdir(_out_path)
    except OSError as ex:
        pass


def _list_files(_folder):
    files = [f for f in listdir(_folder) if isfile(join(_folder, f))]
    files2 = []
    for f in files:
        if '.jpg' in f:
            files2.append((f, 'image'))
        if '.mp4' in f:
            files2.append((f, 'video'))
    return files2


def main_method(in_path, out_path, logger):
    _create_folder(out_path)
    files = _list_files(in_path)
    for file in files:
        try:
            _process_file(file, in_path, out_path, logger)
        except Exception as e:
            t = traceback.format_exc()
            logger.info(f"Error: {e} \n {t}")
            raise e


def _process_file(file, in_path, out_path, logger):
    f, type = file
    if type == 'image':
        process_image(f, in_path, out_path, logger)
    if type == 'video':
        process_video(f, in_path, out_path, logger)


def process_image(file, in_path, out_path, logger):
    in_path = f"{in_path}{file}"
    logger.info(f"Processing image: {in_path}")
    out_path = f"{out_path}{file}"
    image = cv2.imread(in_path, cv2.IMREAD_COLOR)
    if image is None:
        raise Exception(f"Error loading image {in_path}")
    vp_detector = VanishingPointsDetector(out_path, debug_level=0,
                                          _logger=logger)
    vp_lines = vp_detector.execute(image)
    detected_image = draw_vps(image, vp_lines, draw_lines=False)
    cv2.imwrite(f"{out_path}filtered_vps.png", detected_image)


def process_video(file, in_path, out_path, logger):
    in_path = f"{in_path}{file}"
    logger.info(f"Processing video: {in_path}")
    out_path = f"{out_path}{file}"
    video = cv2.VideoCapture(in_path)
    if not video.isOpened():
        raise Exception(f"Error loading video {in_path}")
    frame_idx = 0
    w = int(video.get(3))
    h = int(video.get(4))

    scale = 0.50
    w = int(w * scale)
    h = int(h * scale)
    dim = (w, h)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    framerate = 30
    out_video = cv2.VideoWriter(
        f'{out_path}.avi',
        fourcc, framerate,
        (w, h)
    )
    vp_ma = None
    dv = 5.0/framerate
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            if frame is None:
                msg = f"Couldn't read frame of video {in_path}-{frame_idx}"
                raise Exception(msg)
            logger.info(f"Processing video frame: {in_path}-{frame_idx}")
            vp_detector = VanishingPointsDetector(out_path, debug_level=0,
                                                  _logger=logger)
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            vp_lines = vp_detector.execute(frame)
            vp, lines = vp_lines[0]
            # Moving average of vanishing point
            if vp_ma is None:
                vp_ma = vp
            else:
                x, y = vp
                x_ma, y_ma = vp_ma
                x_ma, y_ma = (1-dv)*x_ma + dv*x, (1-dv)*y_ma + dv*y
                vp_ma = int(x_ma), int(y_ma)
            line_size = max(1, int(h * 0.001))
            draw_vp(frame, line_size, vp_ma)
            out_video.write(frame)
            frame_idx = frame_idx + 1
        else:
            break
    video.release()
    out_video.release()


if __name__ == '__main__':
    from loguru import logger as logger_instance

    input_path = "input/"
    output_path = "output/"
    logger_instance.add(f"./logging.log")
    main_method(input_path, output_path, logger_instance)
