#%%
from tkinter import *
from tkinter import filedialog
from openpyxl.workbook import Workbook
from openpyxl import load_workbook
import main as bg

wb = Workbook() 
wb = load_workbook('OUTPUT.xlsx')
ws = wb.active

column_a = ws ['A']
column_b = ws ['B']
column_c = ws ['C']
column_d = ws ['D']


def get_a():
    list = ''
    for cell in column_a:
        list = f"{list + str(cell.value)}\n"
    Label(window, text = list ).pack(padx = 20,pady=10, side=LEFT)

def get_b():
    list = ''
    for cell in column_b:
        list = f"{list + str(cell.value)}\n"
    Label(window, text = list ).pack(padx = 20,pady=10, side=LEFT)

def get_c():
    list = ''
    for cell in column_c:
        list = f"{list + str(cell.value)}\n"
    Label(window, text = list ).pack(padx = 20,pady=10, side=LEFT)

def get_d():
    list = ''
    for cell in column_d:
        list = f"{list + str(cell.value)}\n"
    Label(window, text = list ).pack(padx = 20,pady=10, side=LEFT)


def get_file_path():
    global file_path
    # Open and return file path
    file_path= filedialog.askopenfilename(title = "Select A File", filetypes = (("type1", "*.png"), ("type2", "*.jpg")))
    l1 = Label(window, text = "File path: " + file_path).pack()

def submit():
    bg.background(file_path)


window = Tk()
window.title("Image to Text")

b1 = Button(window, text = "Open File", command = get_file_path).pack(side=LEFT)
b2 = Button(window, text= "Submit", command= submit).pack(padx = 20,pady=10, side=LEFT)
bA = Button(window, text = "Words", command = get_a).pack(padx = 20,pady=10, side=LEFT)
bB = Button(window, text = "Hindi", command = get_b).pack(padx = 20,pady=10, side=LEFT)
bC = Button(window, text = "Punjabi",command = get_c).pack(padx =20,pady=10, side=LEFT)
bD = Button(window, text = "Tamil", command = get_d).pack(padx = 20,pady=10, side=LEFT)

window.mainloop()

# %%
