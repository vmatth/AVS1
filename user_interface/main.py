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
import re

from course_rating import size
from course_rating import return_everything
from course_rating import draw_elipse

class App():
    def __init__(self):
        # Values for image information
        self.current_image = None # The current rgb orthophoto image of a golf hole displayed on the GUI (including all of the stuff drawn on it e.g tee circles)
        self.raw_image = None # The rgb orthophoto image of a golf hole without extra stuff drawn on it (no circles or lines)
        self.focus_image = None #The current image with highlighted features on top
        self.mask_image = None # The mask image (the different classes) of the prediction
        self.scale = None # The scale of the image
        self.items = []
        self.general_table = None

        # Values for measurements
        self.playertype = 'scratch female'
        self.measurements_ready = False

        # Values for selecting tees
        self.selecting_tees = False
        self.tee_counter = 0
        self.female_tee = None
        self.male_tee = None

        # Load unet model
        self.model = model_utils.load_model()

        # Graphical User Interface
        self.window = tk.Tk()

        # Top frame for the menu buttons
        top = Frame(self.window)
        top.pack(side=TOP, anchor=NW)

        # Bottom frame
        bot = Frame(self.window)
        bot.pack(side=BOTTOM, anchor=S)

        # Open button to select orthophoto of golf course
        self.button_image = tk.Button(text="Open", width=8, height=2, command=lambda:self.show_prediction_image(self.model))
        self.button_image.pack(in_=top, side=LEFT)

        # Select tees button
        self.button_select_tees = tk.Button(text="Tees", width=8, height=2, command=lambda:self.set_selecting_tees())
        self.button_select_tees.pack(in_=top, side=LEFT)

        # Input scale field
        self.entry_scale = tk.Entry(self.window) 
        self.entry_scale.pack(in_=top, side=LEFT)

        # self.button_calcs = tk.Button(text="Calculate", width = 8, height=2, command=lambda: self.calculate_measurements())
        # self.button_calcs.pack(in_=top, side=LEFT)


        self.label_info = tk.Label(text="Open an image containing a single golf hole.")
        self.label_info.pack()

        self.button_scratch_male = tk.Button(text="Scratch Male", width = 18, height=2, command=lambda: self.change_player_type("scratch male"))
        self.button_scratch_male.pack(in_=bot, side=LEFT)

        self.button_scratch_female = tk.Button(text="Scratch Female", width = 18, height=2, command=lambda: self.change_player_type("scratch female"))
        self.button_scratch_female.pack(in_=bot, side=LEFT)

        self.button_bogey_male = tk.Button(text="Bogey Male", width = 18, height=2, command=lambda: self.change_player_type("bogey male"))
        self.button_bogey_male.pack(in_=bot, side=LEFT)

        self.button_bogey_female = tk.Button(text="Bogey Female", width = 18, height=2, command=lambda: self.change_player_type("bogey female"))
        self.button_bogey_female.pack(in_=bot, side=LEFT)       

        # Empty label for the golf hole
        self.image_label = None

        self.window.mainloop()


    # Updates the info label with the given text
    def update_information_label(self, txt):
        self.label_info.config(text=txt)

    # Sets the selecting tee value to either False or True (the opposite of the current value)
    # This allows the user to select where the tees are using the "click" function
    def set_selecting_tees(self):
        if(self.raw_image is None): #Make sure an image is selected
            self.update_information_label("Please open an image first.")
            return
        #Set the displayed image to be raw_image (this removes any previously placed circles)
        self.update_label_image(self.raw_image)
        self.current_image = self.raw_image.copy()
        self.measurements_ready = False
        
        #Flip the selecting tees value (from False -> True or True -> False)
        self.selecting_tees = not self.selecting_tees
        print("Selecting tees: ", self.selecting_tees)
        self.tee_counter = 0
        if self.selecting_tees == True:
            self.update_information_label("Select the male tee")
        elif self.selecting_tees == False:
            self.update_information_label("Canceling selecting tees")
            self.male_tee = None
            self.female_tee = None

    # A click event for pressing on a pixel in the image (Used for selecting the tee areas)
    def click(self, event):
        # Only allow clicking tees if selecting_tees is True
        if self.selecting_tees == True:
            if self.tee_counter == 0: # Male tee
                self.male_tee = [event.x, event.y] # Get the pixel coordinates
                self.tee_counter = 1
                self.update_information_label("Select the female tee")
                # Draw tee circle
                self.draw_circle(self.male_tee, (255, 255, 0), 3)
            elif self.tee_counter == 1: # Female tee
                self.female_tee = [event.x, event.y] # Get the pixel coordinates
                self.update_information_label("Finished selecting tees")
                self.selecting_tees = False
                # Draw tee circle
                self.draw_circle(self.female_tee, (250, 113, 103), 3)
            
    # Draws the tee circles using cv2
    def draw_circle(self, pos, color, size):
        img = cv2.circle(self.current_image, pos, size, color, -1)
        self.update_label_image(img)

    def select_player_type(self, player_type):
        self.playertype = player_type
        #self.update_information_label("")
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
        self.update_information_label("Opening image.")
        self.measurements_ready = False # Reset the all_fairway_widths value when opening a new picture (this ensures that the measurements have to be be recalculated)
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
    
    def draw_line(self, img, start_point, end_point, color, thickness):
        img = cv2.line(img, start_point, end_point, color, thickness)
        self.update_label_image(img)


    def draw_focused_line(self, start_point, end_point, color, thickness):
        self.focus_image = self.current_image.copy()
        img = cv2.line(self.focus_image, start_point, end_point, color, thickness)
        self.update_label_image(img)        

    def draw_dotted_line(self, img, pt1, pt2, color, thickness=1, style='dotted', gap=20):
        dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
        pts= []
        for i in  np.arange(0,dist,gap):
            r=i/dist
            x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
            y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
            p = (x,y)
            pts.append(p)

        if style=='dotted':
            for p in pts:
                cv2.circle(img ,p,thickness,color,-1)
        else:
            s=pts[0]
            e=pts[0]
            i=0
            for p in pts:
                s=e
                e=p
                if i%2==1:
                    cv2.line(img,s,e,color,thickness)
                i+=1

        self.update_label_image(img)

    def reset_information(self):
        for item in self.items:
            item.destroy()

    def generate_table(self, golf_info):
        columns = ['golf_feature', 'value']
        tree = ttk.Treeview(self.window, columns=columns, selectmode=tk.BROWSE)
        tree.pack()
        tree.column("#0", width=0,  stretch=NO)
        tree.heading('#0', text="", anchor=CENTER)
        tree.heading('golf_feature', text='Golf Feature', anchor=CENTER)
        tree.heading('value', text='Value [m]', anchor=CENTER)

        if golf_info is not None:
            for info in golf_info:
                tree.insert('', tk.END, value=info)

        tree.bind("<ButtonRelease-1>", lambda event: self.table_click(event, tree))
        self.items.append(tree)
        
        #tree.grid(row=0, column=0, sticky='nsew')

    def table_click(self, event, tree):
        print("event: ", event)
        print("tree:", tree)

        selected_item = tree.focus()
        print("selected item", selected_item)
        print("row item: ", tree.item(selected_item)['values'])
        
        # table_features = {
        #     'Green size' : [self.green_length, self.green_width],
        #     'Length of hole' : self.all_length_of_holes,
        #     'Tee - Bunker' : self.all_bunker_dists_from_tees,
        #     'Tee - Water' : self.all_water_dists_from_tees,
        #     'Total Stroke Distance' :  self.all_stroke_distances_total,
        #     'Total Fairway Width' : self.all_fairway_widths_total,
        #     'Landing Point' : self.landing_points,
        # }

        self.focus_image = self.current_image.copy()

        item = tree.item(selected_item)['values']
        
        if "Green size" in item:  
            print("you clicked on green size B)")
            cv2.line(self.focus_image, self.green_length[0], self.green_length[1], (170, 250, 0), 4)
            cv2.line(self.focus_image, self.green_width[0], self.green_width[1], (170, 250, 0), 4)
        elif "Length of hole" in item:
            print("l o h")
            self.draw_stroke_line(self.focus_image, self.PLAYER, (255, 255, 255), 4)
        elif re.match("Tee - Bunker", item[0]):
            print('Working regex') 
            bunker_num = int(item[0][-1]) - 1
            if self.PLAYER ==  0 or self.PLAYER == 2:
                point1 = self.male_tee
                tee = 0
            else:
                point1 = self.female_tee
                tee = 1
            point2 = self.all_bunker_dists_from_tees[tee][bunker_num][0]
            print("point1", point1)
            print("2", point2)
            self.draw_dotted_line(self.focus_image, point1, point2, (55,50,55), gap=5)
        elif re.match("Tee - Water", item[0]):
            pass
        elif "Total Stroke Distance" in item:
            
            pass
        elif "Total Faiway Width":
            pass
        
            
        

        
        self.update_label_image(self.focus_image)    


    def draw_tee_obs_dist(self, obs_lst, p_type, color):
        if p_type ==  0 or p_type == 2:
            point1 = self.male_tee
            lst = obs_lst[0]
        else:
            point1 = self.female_tee
            lst = obs_lst[1]
        
        if len(lst) > 0:
            for i in range(len(lst)):
                point2 = lst[i][0]
                self.draw_dotted_line(self.current_image, point1, point2, color, gap=5)
        pass
    
    # Getting the distances from tees to obstacles
    def get_tee_obs_dist(self, info_lst, feature_str, dist_lst):
        if len(dist_lst) > 0:
            for i in range(len(dist_lst)):
                dist = dist_lst[i][1]
                dist_str = (feature_str + f' #{i + 1}', f'{dist:.2f}')
                info_lst.append(dist_str)
            
        return info_lst
    
    # Getting fairway distances // total, carry, average
    def get_fairway_dist(self, dist_lst, info_lst, feature_str, player_type_num, stroke_num):
        if stroke_num < len(dist_lst[player_type_num]):
                fairway_width = dist_lst[player_type_num][stroke_num][2]
                fairway_string = (feature_str, f'{fairway_width:.2f}')
                info_lst.append(fairway_string)
        return info_lst

    # Getting distance from tees to obstacles e.g. water, bunker
    def get_lp_obs_dist(self, dist_lst, info_lst, feature_str, player_type_num, stroke_num):
        if stroke_num < len(dist_lst[player_type_num]): 
                for j in range(len(dist_lst[player_type_num][stroke_num])):
                    dist = dist_lst[player_type_num][stroke_num][j][1]
                    dist_str = (feature_str + f' #{j + 1}', f'{dist:.2f}')
                    info_lst.append(dist_str)
        return info_lst

    def calculate_measurements(self):
        if(self.raw_image is None):
            self.update_information_label("Please open an image first")
            return False
        if (self.male_tee is None or self.female_tee is None):
            self.update_information_label("Please select tee locations first")
            return False

        self.scale = self.is_scale_valid()
        if(self.scale == None):
            return False
            
        # Calculate the green size
        self.green_length, self.green_width, self.green_center = size.get_green_size(self.mask_image.copy(), scale=self.scale)
        # Calculate everything else
        self.all_fairway_widths_total, self.all_fairway_widths_carry, self.all_fairway_widths_average, self.all_bunker_dists, self.all_water_dists, self.landing_points, self.all_strokes, self.all_length_of_holes, self.all_bunker_dists_from_tees, self.all_water_dists_from_tees, self.all_distances_to_green, self.all_stroke_distances_total, self.all_stroke_distances_carry = return_everything.return_everything(self.mask_image.copy(), self.scale, self.male_tee, self.female_tee)
        self.measurements_ready = True
        return True

    def change_player_type(self, p_type):
        self.playertype = p_type
        self.show_calcs()

    def draw_stroke_line(self, img, p_type, color=(210, 147, 235), thickness=2):
        lp_for_player = self.landing_points[p_type] 

        if(p_type == 0 or p_type == 2):
            prev_point = self.male_tee
        else:
            prev_point = self.female_tee

        if len(lp_for_player) > 0:
            for i in range(len(lp_for_player)):
                next_point = lp_for_player[i]
                self.draw_line(img, prev_point, next_point, color, thickness)
                prev_point = next_point
                    
        next_point = self.green_center
        self.draw_line(img, prev_point, next_point, color, thickness)
        
    def is_scale_valid(self):
        scale = self.entry_scale.get()
        if(scale is None):
            self.update_information_label("Please add a scale for the image (1000 - 2000)")
            return None
        if(scale.isdigit() == False):
            self.update_information_label("Please enter a valid scale (1000 - 2000)")
            return None

        try:
            scale = int(scale)
        except:
            self.update_information_label("Please enter a valid integer scale (1000 - 2000)")
            return None

        if scale > 2000 or scale < 1000:
            self.update_information_label("Please enter a valid scale (1000 - 2000)")
            return None      
        
        return scale

    def show_calcs(self):
        # Calculate the measurements if it has not been done
        if(self.measurements_ready == False):
            if self.calculate_measurements() == False:
                return

        # If the scale was updated AFTER calculating the measurements, recalculate them
        new_scale = self.is_scale_valid()
        if new_scale == None:
            return
        if self.scale != new_scale:
            if self.calculate_measurements() == False:
                return

        self.update_information_label("Showing measurements for " + self.playertype)

        # Update the image to remove currently placed circles and lines for other player types
        #Set the displayed image to be raw_image (this removes any previously placed circles)
        self.update_label_image(self.raw_image)
        self.current_image = self.raw_image.copy()
        
        #Draw tee circles
        self.draw_circle(self.male_tee, (255, 255, 0), 3)
        self.draw_circle(self.female_tee, (250, 113, 103), 3)

        player_type={
            'scratch male' :  0,
            'scratch female' : 1,
            'bogey male' : 2,
            'bogey female' : 3
        }
        
        #Reset all information
        self.reset_information()
        
    
        PLAYER = player_type[self.playertype]
        STROKE = self.all_strokes[PLAYER]
        self.PLAYER = PLAYER
        self.STROKE = STROKE

        # Drawing length line of green
        self.draw_line(self.current_image, self.green_length[0], self.green_length[1], (170, 250, 95), 2)
        # Drawing width line of green
        self.draw_line(self.current_image, self.green_width[0], self.green_width[1], (170, 250, 95), 2)

        print('LP ', self.landing_points)
        print('length of holes: ', self.all_length_of_holes)
        print(15 * '-')
        
        self.draw_stroke_line(self.current_image, PLAYER)

        if PLAYER == 0 or PLAYER == 2:
            tee = self.male_tee
        elif PLAYER == 1 or PLAYER == 3:
            tee = self.female_tee
        
        # Draw all of the landingpoints and landingzones for this PLAYER
        if self.landing_points[PLAYER] is not None:
            for stroke in range(len(self.landing_points[PLAYER])):
                self.draw_circle(self.landing_points[PLAYER][stroke], (255, 255, 255), 3)
                self.current_image = draw_elipse.draw_elipse(self.current_image, self.landing_points[PLAYER][stroke], tee, self.playertype, self.all_stroke_distances_total[PLAYER][stroke], self.scale, 2, (255,255,255))

        # Draw all of the fairway widths for this PLAYER
        if self.all_fairway_widths_total[PLAYER] is not None:
            for stroke in range(len(self.all_fairway_widths_total[PLAYER])):
                self.draw_dotted_line(self.current_image, self.all_fairway_widths_total[PLAYER][stroke][0], self.all_fairway_widths_total[PLAYER][stroke][1], (0, 255, 0), 2, style="", gap=8)

        # Draw all of the distances to nearby bunkers for this PLAYER from each landingpoint
        if self.all_bunker_dists[PLAYER] is not None:
            for stroke in range(len(self.all_bunker_dists[PLAYER])):
                for obs in range(len(self.all_bunker_dists[PLAYER][stroke])):
                    self.draw_dotted_line(self.current_image, self.all_bunker_dists[PLAYER][stroke][obs][0], self.landing_points[PLAYER][stroke], (231, 200, 46), 2, style="", gap=8)

        # Draw all of the distances to nearby water hazards for this PLAYER from each landingpoint
        if self.all_water_dists[PLAYER] is not None:
            for stroke in range(len(self.all_water_dists[PLAYER])):
                for obs in range(len(self.all_water_dists[PLAYER][stroke])):
                    self.draw_dotted_line(self.current_image, self.all_water_dists[PLAYER][stroke][obs][0], self.landing_points[PLAYER][stroke], (158, 246, 246), 2, style="", gap=8)

        print("dist from tee to obs: ", self.all_bunker_dists_from_tees)

        # Drawing distance between tee and obstacles i.e. bunker and water
        self.draw_tee_obs_dist(self.all_bunker_dists_from_tees, PLAYER, (246, 239, 108))
        self.draw_tee_obs_dist(self.all_water_dists_from_tees, PLAYER, (155, 193, 246))


        # Table stuff
        general_data = []
        
        green_string = ('Green size', f'{self.green_length[2]:.2f} x {self.green_width[2]:.2f}')
        general_data.append(green_string)

        hole_string = ('Length of hole', f'{self.all_length_of_holes[PLAYER]:.2f}')
        general_data.append(hole_string)

        if PLAYER == 0 or PLAYER == 2:
            general_data = self.get_tee_obs_dist(general_data, 'Tee - Bunker', self.all_bunker_dists_from_tees[0])
            general_data = self.get_tee_obs_dist(general_data, 'Tee - Water', self.all_water_dists_from_tees[0]) 

        elif PLAYER == 1 or PLAYER == 3: 
            general_data = self.get_tee_obs_dist(general_data, 'Tee - Bunker', self.all_bunker_dists_from_tees[1])
            general_data = self.get_tee_obs_dist(general_data, 'Tee - Water', self.all_water_dists_from_tees[1]) 

        self.label_general = tk.Label(text=f"General data")
        self.label_general.pack()
        self.items.append(self.label_general)
        
        #self.general_table = 
        self.generate_table(general_data)
        
        print('fairway width: ', self.all_fairway_widths_total)
        
        for i in range(STROKE):
            self.label_stroke = tk.Label(text=f"Stroke {i + 1}")
            self.label_stroke.pack()
            self.items.append(self.label_stroke)
            
            stroke_data = []
            stroke_data.append(('Total Stroke Distance ', f'{self.all_stroke_distances_total[PLAYER][i]:.2f}'))
            stroke_data.append(('Carry Stroke Distance ', f'{self.all_stroke_distances_carry[PLAYER][i]:.2f}'))

            self.get_fairway_dist(self.all_fairway_widths_total, stroke_data, 'Total Fairway Width', PLAYER, i)
            self.get_fairway_dist(self.all_fairway_widths_carry, stroke_data, 'Carry Fairway Width', PLAYER, i)
            self.get_fairway_dist(self.all_fairway_widths_average, stroke_data, 'Average Fairway Width', PLAYER, i)

            self.get_lp_obs_dist(self.all_bunker_dists, stroke_data, 'Landing Point - Bunker', PLAYER, i)
            self.get_lp_obs_dist(self.all_water_dists, stroke_data, 'Landing Point - Water', PLAYER, i)
            
            front_of_green = self.all_distances_to_green[PLAYER][i][0]
            middle_of_green = self.all_distances_to_green[PLAYER][i][1]
            back_of_green = self.all_distances_to_green[PLAYER][i][2]

            front_str = ('Starting Point - Front of Green', f'{front_of_green:.2f}')
            middle_str = ('Starting Point - Middle of Green', f'{middle_of_green:.2f}')
            back_str = ('Starting Point - Back of Green', f'{back_of_green:.2f}')
            
            stroke_data.append(front_str)
            stroke_data.append(middle_str)
            stroke_data.append(back_str)

            self.generate_table(stroke_data)

        print(15 * '-')
        print('wtf is this: ', self.items)
        print(15 * '-')
        print('wtf is this: ', self.items[0])
        print('wtf is this: ', type(self.items[0]))
        print(15 * '-')
        print('wtf is this: ', self.items[1])
        print('wtf is this: ', type(self.items[1]))
        print(15 * '-')

###
# 4 Player types
    # Green size
    # Fairway width
    # Length of the hole
    # Length to obstacles from landing point
    # Length to obstacles from tee
    # Distance from final landing point to front and back of green
    # Show landing zones (elipse)


#TODO
# Center the circle for tee areas
# Dont allow the user to select tee areas if no image is loaded. Same with show calcs
# Edit the prediction image to fix any errors that might occur

if __name__ == "__main__":
    print("Starting the Golf Course Rating System")
    App()