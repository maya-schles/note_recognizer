import pathlib
import tkinter as tk
import gui
from pdf_sheet_handling import get_pdf_fitted_pages
import sheet_recognition as sr
import sheet_logic as sl


PDF_PATH = pathlib.Path("C:\\Users\\mayas\\Downloads\\Remember-That-We-Suffered-v2-Rev-07_20_19.pdf")
root = tk.Tk()
height = root.winfo_screenheight() - 100

pages = get_pdf_fitted_pages(PDF_PATH, height)
page = pages[0]

staff_locations, line_width = sr.get_staffs(page)
note_locations = sr.get_all_notes(page)
note_heights = [-note_location[0] for note_location in note_locations]  # invert index for higher note to have a higher index
staff_heights = -staff_locations
note_classifications = sl.classify_notes(staff_heights, line_width, note_heights)

main_window = gui.MainWindow(root, sr.clear_notes(page), note_locations, note_classifications, staff_locations, line_width)
main_window.mainloop()
# canvas = tk.Canvas(root, width=page.shape[1], height=height)
#
# display_image = get_tk_image(page)
# canvas_image = canvas.create_image(0, 0, anchor=tk.NW, image=display_image)

# for note, location in zip(note_classifications, note_locations):
#     canvas.create_text(*location[::-1], fill="red", font="Times 20  bold", text=note.name)
# canvas.pack()
#
# root.mainloop()