# Getting the scale of the image from the image path
def get_scale(img_path):
    img_name = img_path.split('\\')[-1]

    img_name = img_name.split('_')

    for val in img_name:
        
        if val <= '2000' and val >= '1250':
            img_scale = val
            
    if img_scale == None:
        print(f'Houston, we have a problem.')
        return None

    return int(img_scale)
