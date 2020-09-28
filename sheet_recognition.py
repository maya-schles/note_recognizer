import itertools
from typing import Tuple, List, Dict

import numpy as np
import cv2
import templates
DEFAULT_SEARCH_THRESHOLD = 0.7


def get_line_start(line: np.ndarray) -> int:
    return np.argwhere(line < 10)[0][0]


def dedup_lines(line_heights: np.ndarray):
    diffs = np.diff(line_heights)
    is_first = np.hstack([[True], diffs > 1])
    return line_heights[is_first]


def get_line_heights(image: np.ndarray) -> np.ndarray:
    line_avg = image.mean(axis=1)
    line_heights = np.argwhere(line_avg < 100)[:, 0]
    return dedup_lines(line_heights)


def get_staffs(image: np.ndarray) -> (np.ndarray, int):
    line_heights = get_line_heights(image)
    assert len(line_heights) % 5 == 0, "number of lines not divisible by 5"
    grouped_staffs = line_heights.reshape((len(line_heights)//5, 5))
    staff_locations = grouped_staffs[:, 2]  # Middle line
    line_width = int(np.median(np.diff(grouped_staffs[0])))
    return staff_locations, line_width


def dedup_search(template_shape: Tuple[int, int], matches: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    def is_close(loc1, loc2):
        return (abs(loc1[0] - loc2[0]) < template_shape[0]) & (abs(loc1[1] - loc2[1]) < template_shape[1])

    def get_cluster(clusters, loc):
        for cluster in clusters:
            if is_close(cluster, loc):
                return cluster
        return None

    matches_clusters = {}
    for match in matches:
        cluster = get_cluster(matches_clusters.keys(), match)
        if cluster is None:
            cluster = match
            matches_clusters[cluster] = []
        matches_clusters[cluster].append(match)
    return [tuple(np.round(np.mean(cluster_elements, axis=0)).astype(int)) for cluster_elements in matches_clusters.values()]


def search(image: np.ndarray, template: np.ndarray, threshold: float = None) -> List[Tuple[int, int]]:
    threshold = threshold or DEFAULT_SEARCH_THRESHOLD
    print("Searching for image with threshold", threshold)
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    matches = list(zip(*loc))
    return dedup_search(template.shape, matches)


def center_location(location: Tuple[int, int], object_shape: Tuple[int, int]) -> Tuple[int, int]:
    return location[0] + object_shape[0]//2, location[1] + object_shape[1]//2


def center_locations(locations: List[Tuple[int, int]], object_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    return [center_location(location, object_shape) for location in locations]


def get_all_notes(image: np.ndarray, note_templates: List[np.ndarray] = templates.NOTES, search_threshold: float = None) \
        -> List[Tuple[int, int]]:
    return list(itertools.chain(*(
        center_locations(search(image, template, threshold=search_threshold), template.shape)
        for template in note_templates)))


DEFAULT_TOP_PAD = 10
DEFAULT_BOTTOM_PAD = 10
DEFAULT_LEFT_PAD = 30
COLOR_WHITE = 255
COLOR_BLACK = 0


def classify_notes_by_staffs(note_locations: List[Tuple[int, int]], staffs: np.ndarray) -> List[int]:
    res = [None] * len(note_locations)
    for i, note_location in enumerate(note_locations):
        res[i] = np.argmin(np.abs(staffs - note_location[0]))
    return res


def cluster_notes_by_staffs(note_locations: List[Tuple[int, int]], staffs: np.ndarray) -> Dict[int, List[Tuple[int, int]]]:
    note_classification = classify_notes_by_staffs(note_locations, staffs)
    res = {k: [] for k in range(len(staffs))}
    for i, classification in enumerate(note_classification):
        res[classification].append(note_locations[i])
    return res


def clear_notes(
        image: np.ndarray,
        pad_left: int = DEFAULT_LEFT_PAD,
        pad_top: int = DEFAULT_TOP_PAD,
        pad_bottom: int = DEFAULT_BOTTOM_PAD,
        fill_color: int = COLOR_WHITE,
        line_color: int = COLOR_BLACK) -> np.ndarray:
    new_image = image.copy()
    staffs, line_width = get_staffs(image)
    clustered_notes = cluster_notes_by_staffs(get_all_notes(image), staffs)
    for i, staff in enumerate(staffs):
        notes_x = [note_location[0] for note_location in clustered_notes[i]]

        line_start = min([note_location[1] for note_location in clustered_notes[i]]) - pad_left
        x_start = staff - 4*line_width
        default_x_end = staff + 3*line_width # TODO: find better name
        x_end = max(max(notes_x) + pad_bottom, default_x_end)
        new_image[x_start:x_end, line_start:] = fill_color

    line_heights = get_line_heights(image)
    for line_height in line_heights:
        line_start = get_line_start(image[line_height])
        new_image[line_height, line_start:] = line_color

    return new_image
