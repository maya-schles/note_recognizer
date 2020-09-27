import itertools
import tkinter as tk
from typing import List, Tuple

import cv2 as cv
from PIL import Image, ImageTk
import numpy as np
import pathlib
from pdf2image import convert_from_path
import sheet_recognition as sr


def pil_image_to_cv(image: Image) -> np.ndarray:
    return cv.cvtColor(np.array(image).copy(), cv.COLOR_BGR2GRAY)


def image_fitted_pages(image: np.ndarray, height: int):
    first, last = np.split(image, [-(len(image) % height)])
    return np.split(first, len(image)//height)+[last]


def filter_relevant(image: np.ndarray, template: np.ndarray, threshold=0.99) -> List[np.ndarray]:
    w, h = template.shape[::-1]
    res = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)
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
        print(curr_height)
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


def get_pdf_fitted_pages(pdf_path: pathlib.Path, height: int) -> List[np.ndarray]:
    raw_pages = convert_from_path(pdf_path)
    cv_pages = map(pil_image_to_cv, raw_pages)
    return list(np.concatenate([image_fitted_pages(img, height) for img in cv_pages]))


def pdf_read(pdf_path: pathlib.Path) -> List[np.ndarray]:
    raw_pages = convert_from_path(pdf_path)
    return list(map(pil_image_to_cv, raw_pages))


def get_lines(image: np.ndarray):
    line_avg = image.mean(axis=1)
    line_locations = [i[0] for i in np.argwhere(line_avg < 100)]
    line_start = 0
    if len(line_locations) > 0:
        line_start = np.argwhere(image[line_locations[0]] < 10)[0][0]
    return [(line_start, y) for y in line_locations]


class PageViewer:
    def __init__(self, root: tk.Tk, pages: List[np.ndarray], page_height: int, search_function, search_shape):
        self.root = root
        self.pages = pages
        self.page_index = 0
        self.canvas = tk.Canvas(self.root, width=self.pages[self.page_index].shape[1], height=page_height)
        self.display_image = self.get_tk_image(self.pages[self.page_index])
        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.display_image)
        self.canvas.pack()
        self.root.bind("<Right>", self.next_page)
        self.root.bind("<Left>", self.prev_page)
        self.search_indicators = []
        self.search_function = search_function
        self.search_shape = search_shape

    def mark_locations(self, locations: List[Tuple[int, int]], height: int, width: int):
        print("marking", len(locations))
        for indicator in self.search_indicators:
            self.canvas.delete(indicator)

        for y, x in locations:
            self.search_indicators.append(
                self.canvas.create_rectangle(x, y, x + width, y + height, outline="#ff5555", width=2))

    @staticmethod
    def get_tk_image(cv_image: np.ndarray):
        return ImageTk.PhotoImage(image=Image.fromarray(cv_image))

    def mainloop(self):
        self.root.mainloop()

    def update_page(self):
        self.display_image = self.get_tk_image(self.pages[self.page_index])
        # line_locations = get_lines(self.pages[self.page_index])
        # self.mark_locations(line_locations, self.pages[0].shape[1]/2, 2)
        locations = self.search_function(self.pages[self.page_index])
        self.mark_locations(locations, *self.search_shape)
        self.canvas.itemconfig(self.canvas_image, image=self.display_image)

    def next_page(self, event):
        self.page_index = min(self.page_index+1, len(self.pages)-1)
        self.update_page()

    def prev_page(self, event):
        self.page_index = max(self.page_index-1, 0)
        self.update_page()


class MainWindow:
    def __init__(self, root, img):
        self.root = root
        self.img = img
        self.display_image =ImageTk.PhotoImage(image=Image.fromarray(self.img))
        height = min(img.shape[0], root.winfo_screenheight() - 100)
        self.canvas = tk.Canvas(self.root, width=img.shape[1], height=min(img.shape[0], height))
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.display_image)
        self.start_x = 0
        self.start_y = 0
        self.end_x = 0
        self.end_y = 0
        self.select_rec = self.canvas.create_rectangle(self.start_x, self.start_y, self.end_x, self.end_y, outline='#5555ff', width=2)
        self.canvas.bind("<Button-1>", self.press)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.root.bind("<Return>", self.save)
        self.root.bind("<space>", self.search)
        self.threshold = 0.7
        self.search_indicators = []

    def search(self, event):
        print("Searching for image with threshold", self.threshold)
        template = cv.imread('temp.png', 0)
        w, h = template.shape[::-1]
        res = cv.matchTemplate(self.img, template, cv.TM_CCOEFF_NORMED)
        loc = np.where(res >= self.threshold)
        matches = list(zip(*loc[::-1]))
        for indicator in self.search_indicators:
            self.canvas.delete(indicator)
        print("found matches:", matches)
        for pt in matches:
            self.search_indicators.append(
                self.canvas.create_rectangle(*pt, pt[0] + w, pt[1] + h, outline="#ff5555", width=2))

    def save(self, event):
        print("Saving image at:", self.start_x, self.end_x, self.start_y, self.end_y)
        new_image = self.img[self.start_y:self.end_y, self.start_x:self.end_x]
        cv.imwrite('temp.png', new_image)

    def press(self, event):
        self.start_x = event.x
        self.start_y = event.y

    def drag(self, event):
        self.end_x = event.x
        self.end_y = event.y
        self.canvas.coords(self.select_rec, self.start_x, self.start_y, self.end_x, self.end_y)

    def mainloop(self):
        tk.mainloop()

# PDF_PATH = pathlib.Path("C:\\Users\\mayas\\Downloads\\Wheres-The-Bathroom-Revised-Final-10.31.16.pdf")
# root = tk.Tk()
# height = root.winfo_screenheight() - 100
# # pages = get_pdf_fitted_pages(PDF_PATH, height)
#
# key = cv.imread("templates/treble_clef.png", 0)
# all_parts = list(itertools.chain.from_iterable([filter_relevant(img, key, 0.9) for img in pdf_read(PDF_PATH)]))
# pages = to_fitted_pages(all_parts, height)
# # window = MainWindow(root, pages[0])
# template = cv.imread("templates/full_circle.png", 0)
# search_function = lambda x: sr.dedup_search(template.shape, sr.search(x, template))
# window = PageViewer(root, pages, height, search_function, template.shape)
# window.mainloop()
img = cv.imread("C:/Users/mayas/Desktop/notebooks/cadenCV/temp.png", 0)

root = tk.Tk()
main_window = MainWindow(root, img)
main_window.mainloop()