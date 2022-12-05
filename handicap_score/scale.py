# Getting the scale of the image from the image path
def get_scale(img_path):
    print("Image path: ", img_path)
    img_name = img_path.split('\\')[-1]

    img_name = img_name.split('_')
    print("Image split: ", img_name)
    for val in img_name:
        
        if val <= '2000' and val >= '1000' and len(val)==4:
            img_scale = val
            
    if img_scale == None:
        print(f'Houston, we have a problem.')
        return None
    print("Image SCALE: ", img_scale)
    return int(img_scale), img_path.split('__')
