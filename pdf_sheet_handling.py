import pathlib
from typing import List

import numpy as np
import cv2
import itertools

from PIL import Image
from pdf2image import convert_from_path


KEY_PATH = "templates/treble_clef.png"


def pdf_read(pdf_path: pathlib.Path) -> List[np.ndarray]:
    def pil_image_to_cv(image: Image) -> np.ndarray:
        return cv2.cvtColor(np.array(image).copy(), cv2.COLOR_BGR2GRAY)
    raw_pages = convert_from_path(pdf_path)
    return list(map(pil_image_to_cv, raw_pages))


def filter_relevant(image: np.ndarray, template: np.ndarray, threshold=0.9) -> List[np.ndarray]:
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    matches = list(zip(*loc[::-1]))
    filtered = [image[loc[1]:loc[1] + h] for loc in matches]
    return filtered


def to_fitted_pages(filtered_parts: List[np.ndarray], height: int) -> List[np.ndarray]:
    res = []
    curr_height = 0
    curr_parts = []
    for part in filtered_parts:
        curr_height += part.shape[0]

        if curr_height > height:
            if len(curr_parts) == 0:
                curr_parts = [part[:height, :]]
                part = part[height:, :]

            res.append(np.vstack(curr_parts))
            curr_parts = []
            curr_height = part.shape[0]
        curr_parts.append(part)

    if len(curr_parts) > 0:
        res.append(np.vstack(curr_parts))
    return res


def get_pdf_fitted_pages(pdf_path: pathlib.Path, height: int, key_path: str = KEY_PATH) -> List[np.ndarray]:
    key = cv2.imread(str(key_path), 0)
    all_parts = list(itertools.chain.from_iterable([filter_relevant(img, key) for img in pdf_read(pdf_path)]))
    return to_fitted_pages(all_parts, height)