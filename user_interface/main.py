import tkinter as tk
import model_utils
import select_image

window = tk.Tk()

label = tk.Label(text="hey kata")
label.pack()

#load unet model for (no button)
model = model_utils.load_model()

panelA = None

#open button to select orthophoto of golf course
button = tk.Button(
    text="Select an image",
    width=25,
    height=5,
    command= lambda: select_image.select_image(panelA, model)
)

button.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
#use model to predict the orthophoto
#x = x.to(model_utils.DEVICE)
#y = y.to(model_utils.device)
#preds = model(x)
#overlay prediction image onto orthophoto

#exit button


window.mainloop()