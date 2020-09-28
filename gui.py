import tkinter as tk
from typing import Tuple, List, Optional

import numpy as np
from PIL import ImageTk, Image
import sheet_logic as sl


def get_tk_image(cv_image: np.ndarray):
    return ImageTk.PhotoImage(image=Image.fromarray(cv_image))


class OptionalNote:
    NOTE_WIDTH = 7
    NOTE_COLOR = 'red'
    SELECTED_LOCKED_COLOR = 'green'

    def __init__(self, canvas: tk.Canvas, center_location: Tuple[int, int]):
        self.canvas = canvas
        self.center_location = center_location
        self.id = self.canvas.create_oval(
            *center_location, *center_location, activewidth=self.NOTE_WIDTH, outline=self.NOTE_COLOR)

    def is_selected(self) -> bool:
        return "current" in self.canvas.gettags(self.id)

    def lock(self) -> None:
        if self.is_selected():
            self.canvas.itemconfig(self.id, width=self.NOTE_WIDTH, outline=self.SELECTED_LOCKED_COLOR)
        else:
            self.canvas.itemconfig(self.id, activewidth=0)


class NoteSelector:
    def __init__(self, canvas: tk.Canvas, center_location: Tuple[int, int], lowest_note: sl.Note, highest_note: sl.Note, linewidth: int):
        self.optional_notes = {
            note: OptionalNote(canvas, np.add(center_location, self.get_note_relative_center(note, linewidth)))
            for note in sl.note_range(lowest_note, highest_note)}
        self.is_locked = False

    @staticmethod
    def get_note_relative_center(note: sl.Note, line_width: int) -> Tuple[int, int]:
        i = note.to_index() - 6
        height_delta = -np.round(i*line_width/2)  # Higher note - lower index...
        return 0, height_delta

    def is_selected(self) -> bool:
        return np.any([view.is_selected() for view in self.optional_notes.values()])

    def get_selection(self) -> Optional[sl.Note]:
        for note, view in self.optional_notes.items():
            if view.is_selected():
                return note
        return None

    def lock_selection(self) -> None:
        for view in self.optional_notes.values():
            view.lock()
        self.is_locked = True

    def unlock(self) -> None:
        self.is_locked = False


class MainWindow:
    DEFAULT_LOWEST_NOTE = sl.Note(4, sl.NoteClass.C)
    DEFAULT_HIGHEST_NOTE = sl.Note(5, sl.NoteClass.A)

    def __init__(
            self,
            root: tk.Tk,
            image: np.ndarray,
            note_locations: List[Tuple[int, int]],
            note_classifications: List[sl.Note],
            staffs: np.ndarray,
            line_width: int):
        self.root = root
        self.canvas = tk.Canvas(self.root, width=image.shape[1], height=image.shape[0])
        self.display_image = get_tk_image(image)
        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.display_image)
        self.line_width = line_width
        self.note_classifications = note_classifications

        lowest_note = min(min(note_classifications), self.DEFAULT_LOWEST_NOTE)
        highest_note = max(max(note_classifications), self.DEFAULT_HIGHEST_NOTE)

        self.note_selectors = []
        for note_location in note_locations:
            staff = sl.get_closest_staff(staffs, note_location[0])
            canvas_location = (note_location[1], staff)
            note_selector = NoteSelector(self.canvas, canvas_location, lowest_note, highest_note, line_width)
            self.note_selectors.append(note_selector)
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.pack()

    def mainloop(self):
        return self.root.mainloop()

    def get_active_note(self) -> (int, sl.Note):
        for i, note_selector in enumerate(self.note_selectors):
            if note_selector.is_selected():
                return i, note_selector.get_selection()
        return None, None

    def click(self, event):
        note_ind, note = self.get_active_note()
        if note_ind is not None:
            if note == self.note_classifications[note_ind]:
                self.note_selectors[note_ind].lock_selection()
            print(str(note), note == self.note_classifications[note_ind])
