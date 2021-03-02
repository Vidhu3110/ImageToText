#%%
from tkinter import *
from tkinter import filedialog
from openpyxl.workbook import Workbook
from openpyxl import load_workbook
import main as bg
import mail as m
import os
import pdf
wb = Workbook() 


def get_file_path():
    global file_path
    # Open and return file path
    file_path= filedialog.askopenfilename(title = "Select A File", filetypes = (("type1", "*.png"), ("type2", "*.jpg"),("type3","*jpeg"),("type4",".pdf")))
    l1 = Label(window, text = "File path: " + file_path).pack()

def submit():
    bg.background(file_path)

def submit_pdf():
    pdf.get_pdf(file_path)

def get_e():
    m.bg()

window = Tk()
window.title("Image to Text")

b1 = Button(window, text = "Open File", command = get_file_path).pack(side=LEFT)
b2 = Button(window, text= "Submit", command= submit).pack(padx = 20,pady=10, side=LEFT)
se = Button(window, text = "send mail", command = get_e).pack(padx = 20,pady=10, side=LEFT)
pd = Button(window, text = "submit pdf", command = submit_pdf).pack(padx = 20,pady=10, side=LEFT)
window.mainloop()

# %%
