from typing import List

import numpy as np
from enum import Enum


class Note(Enum):
    B = 0
    A = 1
    G = 2
    F = 3
    E = 4
    D = 5
    C = 6


class Line:
    def __init__(self, staff_height, notes, note_locations):
        self.staff_height = staff_height
        self.notes = notes
        self.note_locations = note_locations





def get_closest_staff(staff_locations: np.ndarray, note_height) -> int:
    return staff_locations[np.argmin(np.abs(staff_locations - note_height))]


def classify_note(staff_locations: np.ndarray, line_width: int, note_height: int) -> Note:
    closest_staff = get_closest_staff(staff_locations, note_height)
    note_estimation = np.round((2*(note_height-closest_staff) / line_width))
    return Note(note_estimation % len(Note))


def classify_notes(staff_locations: np.ndarray, line_width: int, note_heights: List[int]) -> List[Note]:
    return [classify_note(staff_locations, line_width, note_height) for note_height in note_heights]