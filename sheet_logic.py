from typing import List, Iterable

import numpy as np
from enum import Enum


class NoteClass(Enum):
    C = 0
    D = 1
    E = 2
    F = 3
    G = 4
    A = 5
    B = 6


class NoteModifier(Enum):
    NONE = 0
    SHARP = 1
    FLAT = 2


class Note:
    """
    This class represents a musical note, by class and objects (does not include accidentals. )
    It supports indexing, by counting the notes from middle C (C4).
    """
    def __init__(self, octave: int, note_class: NoteClass):
        assert 1 <= octave <= 7, "Octave for note is outside acceptable range. "
        self.octave = octave
        self.note_class = note_class

    @staticmethod
    def from_index(i: int) -> "Note":
        octave = i//7 + 4  # index 0 is 6 notes above middle C
        note_class = NoteClass(i % len(NoteClass))
        return Note(octave, note_class)

    def to_index(self) -> int:
        return (self.octave - 4) * 7 + self.note_class.value

    def __lt__(self, other):
        return self.to_index() < other.to_index()

    def __le__(self, other):
        return self.to_index() <= other.to_index()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.to_index() == other.to_index()
        return False

    def __ne__(self, other):
        return self.to_index() != other.to_index()

    def __gt__(self, other):
        return self.to_index() > other.to_index()

    def __ge__(self, other):
        return self.to_index() >= other.to_index()

    def __hash__(self) -> int:
        return hash(self.to_index())

    def __repr__(self) -> str:
        return f"Note({self.note_class.name}{str(self.octave)})"


def get_closest_staff(staff_locations: np.ndarray, note_height) -> int:
    return staff_locations[np.argmin(np.abs(staff_locations - note_height))]


def classify_note(staff_locations: np.ndarray, line_width: int, note_height: int) -> Note:
    closest_staff = get_closest_staff(staff_locations, note_height)
    note_estimation = round((2*(note_height-closest_staff) / line_width)) + 6 # difference between center of treble clef (B4) to middle C (C4) TODO: const
    return Note.from_index(note_estimation)


def classify_notes(staff_locations: np.ndarray, line_width: int, note_heights: List[int]) -> List[Note]:
    return [classify_note(staff_locations, line_width, note_height) for note_height in note_heights]


def note_range(lowest_note: Note, highest_note: Note) -> Iterable[Note]:
    for i in range(lowest_note.to_index(), highest_note.to_index() + 1):
        yield Note.from_index(i)
