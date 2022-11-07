import cv2

def get_class_coords(image):
    #fairway_list, green_list, tee_list, bunker_list, water_list = ([] for _ in range(5))
    
    #r g b -> b g r
    #colors of each class
    colors_rgb= {
        'tee': np.array([0, 36, 250]), #red
        'water': np.array([231, 200, 46]), #blue
        'bunker': np.array([158, 246, 246]), #yellow
        'fairway': np.array([77, 156, 77]), #dark green
        'green': np.array([122, 243, 142]) #light green
    }
    
    #mask for each class
    mask = {}
    for item in colors_rgb:
        mask.update({item : cv2.inRange(image, colors_rgb[item], colors_rgb[item])})
    
    '''
    for y , value_y in enumerate(image):
        for x, value_x in enumerate(value_y):
            if value_x[0]==0 and value_x[1]==140 and value_x[2]==0:
                fairway_list.append((x,y))
            if value_x[0]==0 and value_x[1]==255 and value_x[2]==0:
                green_list.append((x,y))
            if value_x[0]==255 and value_x[1]==0 and value_x[2]==0:
                tee_list.append((x,y))
            if value_x[0]==217 and value_x[1]==230 and value_x[2]==122:
                bunker_list.append((x,y))
            if value_x[0]==7 and value_x[1]==15 and value_x[2]==247:
                water_list.append((x,y))
    '''
    # print("fair: ", len(fairway_list))
    # print("green: ", len(green_list))
    # print("tee: ", len(tee_list))
    # print("bunker: ", len(bunker_list))
    # print("water: ", len(water_list))
    return mask['tee'], mask['water'], mask['fairway'], mask['green'], mask['bunker']
