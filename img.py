import cv2
import os

def rsize(inputpath,outputpath, prefix):
    if not os.path.isfile(inputpath):
        print('File not found!')
    else:
        img = cv2.imread(inputpath)
        img_height = img.shape[0]
        img_width = img.shape[1]
        print(f'Orginal Scale : {img.shape[1]}x{img.shape[0]}')
        scale = max(512.0/img_width,384.0/img_height)
        new_height = int(img_height*scale*1.12)
        new_width = int(img_width*scale*1.12)
        if img_height > img_width:
            img = cv2.transpose(img)
            new_img = cv2.resize(img,(new_height,new_width),interpolation=cv2.INTER_AREA)
            new_img = new_img[30:414,30:542,:]
        else:
            new_img = cv2.resize(img,(new_width,new_height),interpolation=cv2.INTER_AREA)
            new_img = new_img[30:414,30:542,:]
        print(f'New Scale : {new_img.shape[1]}x{new_img.shape[0]}')

        global count
        count = len(os.listdir(outputpath))
        filename = f'{prefix}{count+1}.jpg'
        outputpath = os.path.join(outputpath, filename)
        print(f'Output Path : {outputpath}')
        print(f'===========================')
        cv2.imwrite(outputpath,new_img)


input_dir = 'imgs'
output_dir = 'Garbage classification copy' 
imgfilenames = [(os.path.join(input_dir, subfolder, x), os.path.join(output_dir, subfolder)) 
                for subfolder in os.listdir(input_dir) 
                for x in os.listdir(os.path.join(input_dir, subfolder))]

for path in imgfilenames:
    print(f'Input Path : {path[0]}')
    prefix = os.path.basename(path[1])
    rsize(path[0],path[1], prefix)



