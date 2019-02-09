from tkinter import *

from tkinter import messagebox
import os

window = Tk()

window.title("Welcome to LikeGeeks app")

window.geometry('350x200')


def clicked():
    exec(open("obj.py").read())


btn = Button(window, text='Click here', command=clicked)

btn.grid(column=0, row=0)

window.mainloop()

