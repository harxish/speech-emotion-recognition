import time
from tkinter import *
from tkinter.ttk import * 
from test import test
  
# create root window 
root = Tk() 
  
# root window title and dimension 
root.title("Emotion Detector") 
root.geometry('700x375') 
  
style = Style() 
style.configure('TButton', font = 
			   ('calibri', 15, 'bold'), 
					borderwidth = '4') 
  
# Changes will be reflected 
# by the movement of mouse. 
style.map('TButton', foreground = [('active', 'green')], background = [('active', 'black')]) 

# adding a label to the root window 
lbl = Label(root, text = "Find the Emotion of your speech by clicking the record button",
	font=('calibri', 14, 'bold'), foreground='red') 
lbl.place(relx=.03, rely=.1) 

dura = Label(root, text='Duration', font=('calibri', 12, 'bold'), foreground='blue')
dura.place(relx=.37, rely=.25)

duration = Entry()
duration.place(relx=.5, rely=.25)

# function to display text when 
# button is clicked 

def clicked():
	
	lbl.configure(text = f"Please Talk...", foreground='green')	
	root.update()
	try:
		emotion = test(int(duration.get()))
		print(int(duration.get()))
	except:
		emotion = test()
	lbl.configure(text = f"Emotion : {emotion}", foreground='purple')
	root.update()
	

def reset(): 
	lbl.configure(text = "Find the Emotion of your speech by clicking the record button", foreground='red')
	duration.delete(0)
  
# button widget with red color text 
# inside 
btn1 = Button(root, text = "Record" ,command=clicked) 
btn2 = Button(root, text = "Reset" ,command=reset) 
  
btn1.place(relx = 0.25, rely = 0.5, anchor = CENTER) 
btn2.place(relx = 0.75, rely = 0.5, anchor = CENTER) 
  
root.mainloop()
