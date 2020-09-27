import tkinter as tk
from typing import Tuple, List, Optional

import numpy as np
from PIL import ImageTk, Image
import sheet_logic as sl


def get_tk_image(cv_image: np.ndarray):
    return ImageTk.PhotoImage(image=Image.fromarray(cv_image))


NOTE_WIDTH = 7


class MainWindow:
    def __init__(
            self,
            root: tk.Tk,
            image: np.ndarray,
            note_locations: List[Tuple[int, int]],
            note_classifications: List[sl.Note],
            staffs: List[int],
            line_width: int):
        self.root = root
        self.canvas = tk.Canvas(self.root, width=image.shape[1], height=image.shape[0])
        self.display_image = get_tk_image(image)
        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.display_image)
        self.line_width = line_width
        self.note_classifications = note_classifications

        self.note_indicator_lists = []
        for note_location in note_locations:
            staff = sl.get_closest_staff(staffs, note_location[0])
            self.note_indicator_lists.append(self._create_note_indicator(note_location[1], staff))
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.pack()

    def _create_note_indicator(self, note_x, staff_y, above_num=5, below_num=9):
        note_indicators = []
        for i in range(-above_num, below_num+1):
            note_indicator = self.canvas.create_oval(
                note_x, staff_y+round(i*self.line_width/2), note_x+2, staff_y+round(i*self.line_width/2)+2,
                activewidth=NOTE_WIDTH, outline='red')
            note_indicators.append((note_indicator, sl.Note(i % len(sl.Note))))
        return note_indicators

    def mainloop(self):
        return self.root.mainloop()

    def get_active_note(self) -> (int, sl.Note):
        for i, note_indicators in enumerate(self.note_indicator_lists):
            for indicator in note_indicators:
                if "current" in self.canvas.gettags(indicator[0]):
                    return i, indicator[1]
        return None, None

    def click(self, event):
        note_ind, note = self.get_active_note()
        if note_ind is not None:
            print(str(note), note == self.note_classifications[note_ind])
