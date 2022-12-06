from tkinter import *
import tkinter as tk
from  tkinter import ttk
import model_utils
from tkinter import filedialog
import cv2
from PIL import Image
from PIL import ImageTk
import model_utils
import numpy as np
from get_classes import get_class_coords as gcc
import select_image

from course_rating import size
from course_rating import return_everything


class App():
    def __init__(self):
        self.window = tk.Tk()

        self.label = tk.Label(text="hey kata")
        self.label.pack()

        self.image_label = None

        
        self.current_image = None # The current rgb orthophoto image of a golf hole displayed on the GUI (including all of the stuff drawn on it e.g tee circles)
        self.raw_image = None # The rgb orthophoto image of a golf hole without extra stuff drawn on it (no circles or lines)
        self.mask_image = None # The mask image (the different classes) of the prediction
        self.scale = 1250 # The scale of the image

        self.playertype = 'scratch_female'

        # Values for selecting tees
        self.selecting_tees = False
        self.tee_counter = 0
        self.female_tee = []
        self.male_tee = []

        # Load unet model
        self.model = model_utils.load_model()

        # Open button to select orthophoto of golf course
        self.button_image = tk.Button(
            text="Select an image",
            width=25,
            height=5,
            command= lambda: self.show_prediction_image(self.model)
        )
        self.button_image.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

        # Button to manually select the tees
        self.button_select_tees = tk.Button(
            text="Select tees",
            width=25,
            height=5,
            command=lambda: self.set_selecting_tees()
        )
        self.button_select_tees.pack(side="right", fill="both", expand="yes", padx="10", pady="10")

        self.button_calcs = tk.Button(
            text="Show calcs",
            width = 25,
            height=5,
            command=lambda: self.show_calcs()
        )
        self.button_calcs.pack(side="right", fill="both", expand="yes", padx="10", pady="10")

        self.window.mainloop()

    # Sets the selecting tee value to either False or True (the opposite of the current value)
    # This allows the user to select where the tees are using the "click" function
    def set_selecting_tees(self):
        #Set the displayed image to be raw_image (this removes any previously placed circles)
        self.update_label_image(self.raw_image)
        self.current_image = self.raw_image.copy()
        
        #Flip the seleting tees value (from False -> True or True -> False)
        self.selecting_tees = not self.selecting_tees
        print("Selecting tees: ", self.selecting_tees)
        self.tee_counter = 0
        if self.selecting_tees == True:
            self.button_select_tees.config(text="Select the male tee")
        elif self.selecting_tees == False:
            self.button_select_tees.config(text="Select tees")

    # A click event for pressing on a pixel in the image (Used for selecting the tee areas)
    def click(self, event):
        # Only allow clicking tees if selecting_tees is True
        if self.selecting_tees == True:
            if self.tee_counter == 0: # Male tee
                self.male_tee = [event.x, event.y] # Get the pixel coordinates
                self.tee_counter = 1
                self.button_select_tees.config(text="Select the female tee")
                # Draw tee circle
                self.draw_tee_circle(self.male_tee, (255, 255, 0))
            elif self.tee_counter == 1: # Female tee
                self.female_tee = [event.x, event.y] # Get the pixel coordinates
                self.button_select_tees.config(text="Select tees")
                self.selecting_tees = False
                # Draw tee circle
                self.draw_tee_circle(self.female_tee, (250, 113, 103))
            print("PX: ", event.x)
            print("PY: ", event.y)
            
    def draw_tee_circle(self, pos, col):
        img = cv2.circle(self.current_image, pos, 3, col, -1)
        self.update_label_image(img)

    # Takes a cv image as input and displays it on the image_label
    def update_label_image(self, image):
        img = Image.fromarray(image)
        img = ImageTk.PhotoImage(img)
        self.image_label.configure(image=img)
        self.image_label.image = img      

    # Selects an image and predicts the segmentation mask using a model
    def show_prediction_image(self, model):
        PIL_image, self.current_image, self.mask_image = select_image.select_image(model) #Calculates the prediction image using the loaded model
        self.raw_image = self.current_image.copy() # Save the same image to both variables. (Current image will update with all of the circles and so on)
    
        # Create a Tkinter label to show the image
        if self.image_label is None:
            print("Creating new Panel")
            self.image_label = Label(image=PIL_image)
            self.image_label.image = PIL_image
            self.image_label.pack(side="left", padx=10, pady=10)
            self.image_label.bind("<Button-1>", self.click)
        # Use same Tkinter label (if choosing a new image, after selecting one)
        else:
            print("Using same Panel")
            self.image_label.configure(image=PIL_image)
            self.image_label.image = PIL_image
            self.image_label.pack(side="left", padx=10, pady=10)
    
    def draw_line(self, start_point, end_point, color):
        img = cv2.line(self.current_image, start_point, end_point, color, 3)
        self.update_label_image(img)

    '''
    def generate_table(self, golf_info):
        columns = ['golf_feature', 'value']
        tree = ttk.TreeView(self.window, columns=columns)

        tree.heading('golf_feature', text='Golf Feature')
        tree.heading('value', text='Value')

        for info in golf_info:
            tree.insert('', tk.END, value=info)
        
        tree.grid(row=0, column=0, sticky='nsew')
    '''

    def show_calcs(self):
        length, width, mp = size.get_green_size(self.mask_image, self.scale)

        all_fairway_widths, all_bunker_dists, all_water_dists, landing_points, all_strokes, all_length_of_holes, all_bunker_dists_from_tees, all_water_dists_from_tees, all_distances_to_green = return_everything.return_everything(self.mask_image, self.scale, self.male_tee, self.female_tee)
        print('Length of all holes: ', all_length_of_holes)
        print('Type of above: ', type(all_length_of_holes))
        
        print('bunker', all_bunker_dists)
        print('b type', type(all_bunker_dists))
        print("lord please save us")
        # print('palyer ', all_bunker_dists[0]) #player 1
        # print('stroke ', all_bunker_dists[0][0]) #stroke 1
        # print('obs ',all_bunker_dists[0][0][0]) #obstacle 1
        # print("obst point", all_bunker_dists[0][0][0][0]) #obstacle 1 POINT
        # print("obs dist", all_bunker_dists[0][0][0][1]) #obstacle 1 DISTANCE

        print("Distances to green: ", all_distances_to_green)
        #stracht male, stracht female, bogey male, bogey female
        
        player_type={
            'scratch male' :  0,
            'scratch female' : 1,
            'bogey male' : 2,
            'bogey female' : 3
        }
        
        # TODO: choose player type - doing by user
        #stroke_num = all_strokes[player_type]
        stroke_num = all_strokes[player_type['scratch female']]

        #drawing length line of green
        self.draw_line(length[0], length[1], (242, 121, 94))
        #drawing width line of green
        self.draw_line(width[0], width[1], (245, 71, 27))

        for i in range(stroke_num):
            self.label_stroke = tk.Label(text=f"Stroke {i + 1}")
            self.label_stroke.pack()
            
            set = ttk.Treeview(self.window)
            set.pack()
            
            set['columns']= ('Golf Feature', 'Value')
            set.column("#0", width=0,  stretch=NO)
            set.column("Golf Feature", anchor=CENTER, width=160)
            set.column("Value", anchor=CENTER, width=160)

            set.heading("#0", text="", anchor=CENTER)
            set.heading("Golf Feature", text="Golf Feature", anchor=CENTER)
            set.heading("Value", text="Value [m]", anchor=CENTER)

            if i <= (len(all_fairway_widths[stroke_num]) - 1):
                fairway_width = round(all_fairway_widths[stroke_num][i],2)
                
                print('FW : ', fairway_width)
            else :
                fairway_width = 'No data'
            
            set.insert(parent='',index='end',iid=0,text='', values=('Green size', f'{length[2]:.2f} x {width[2]:.2f}')) #outside
            set.insert(parent='',index='end',iid=1,text='', values=('Fairway width', f'{fairway_width}'))
            set.insert(parent='',index='end',iid=2,text='', values=('Length of hole', f'{all_length_of_holes[stroke_num]:.2f}')) #outside
            set.insert(parent='',index='end',iid=3,text='', values=('Obstacle - Landing Point','jack'))
            set.insert(parent='',index='end',iid=4,text='', values=('Obstacle - Tee','joy'))
            set.insert(parent='',index='end',iid=5,text='', values=('Final Landing Point - Front of Green','joy'))
            set.insert(parent='',index='end',iid=6,text='', values=('Final Landing Point - Back of Green','joy'))
        
                

###
# 4 Player types
    # Green size
    # Fairway width
    # Length of the hole
    # Length to obstacles from landing point
    # Length to obstacles from tee
    # Distance from final landing point to front and back of green
    # Show landing zones


#TODO
# Center the circle for tee areas
# Dont allow the user to select tee areas if no image is loaded. Same with show calcs
# Edit the prediction image to fix any errors that might occur

if __name__ == "__main__":
    print("Starting the Golf Course Rating System")
    App()