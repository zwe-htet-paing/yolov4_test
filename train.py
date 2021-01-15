import numpy as np
from keras.preprocessing.image import load_img, img_to_array

# load and prepare the image
def load_image_pixels(filename, shape):
    # load image to get its shape
    img = load_img(filename)
    width, height = img.size
    # load image with required size
    img = load_img(filename, interpolation='bilinear', target_size=shape)
    # convert to numpy array
    image = img_to_array(img)
    # Scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0

    # add a dimensoin so that we have one sample
    image = np.expand_dims(image, 0)

def build_train(year, image_set, nb_train, VOC_path):
    train_x  = np.zeros((nb_train, NETWORK_H, NETWORK_W, 3), dtype=np.float32)
    train_y0 = np.zeros((nb_train, grid[0][1], grid[0][0], NB_BOX, (4+1+NB_CLASS)), dtype=np.float32)
    train_y1 = np.zeros((nb_train, grids[1][1], grids[1][0], NB_BOX, (4+1+NB_CLASS)), dtype=np.float32)
    train_y2 = np.zeros((nb_train, grids[2][1], grids[2][0], NB_BOX, (4+1+NB_CLASS)), dtype=np.float32)
    bc = 0
    image_ids = open(VOC_path+ 'VOC%s\ImageSets\Main\%s.txt'%(year, image_set)).read().strip().split()

    for image_id in image_ids:
        img_filename = VOC_path+ 'VOC%s\\JPEGImages\\%s.jpg'%(year, image_id)
        # print(img_filename)
        image, image_w, image_h = load_image_pixels(img_filename, (NETWORK_W, NETWORK_H))
        train_x[bc,:,:,:] = image

        # build true predict train_y0 and box b0
        labels_file = open('dataset\\VOC%s_%s\\labels_0\\%s.txt'%(year, image_set, image_id), 'r')

        rec = np.fromfile(labels_file, dtype=np.float32, sep = " ")
        for i in range(len(rec)//8):
            classid, x, y, w, h, grid_x, grid_y, best_anchor = rec[8*i: 8*(i+1)]
            train_y0[bc, int(grid_y), int(grid_x), int(best_anchor), 5+ int(classid)] = 0.9
            #Class label smoothing, use 0.9 instead of 1.0 in order to mitigate overfitting.
         # build true predict train_y1 and box b1
        labels_file = open('VOCYoloV4\\VOC%s_%s\\labels_1\\%s.txt'%(year, image_set, image_id), 'r')
        
        rec = np.fromfile(labels_file, dtype=np.float32, sep = " ")
        true_box_index = 0
        for i in range(len(rec)//8):
            classid,x,y,w,h,grid_x,grid_y,best_anchor = rec[8*i:8*(i+1)]
            train_y1[bc, int(grid_y),int(grid_x),int(best_anchor), 0:4] = x,y,w,h
            train_y1[bc, int(grid_y),int(grid_x),int(best_anchor), 4] = 1.
            train_y1[bc, int(grid_y),int(grid_x),int(best_anchor), 5+ int(classid)] = 0.9 
            #Class label smoothing, use 0.9 instead of 1.0 in order to mitigate overfitting.

        # build true predict train_y2 and box b2
        labels_file = open('VOCYoloV4\\VOC%s_%s\\labels_2\\%s.txt'%(year, image_set, image_id), 'r')
        
        rec = np.fromfile(labels_file, dtype=np.float32, sep = " ")
        true_box_index = 0
        for i in range(len(rec)//8):
            classid,x,y,w,h,grid_x,grid_y,best_anchor = rec[8*i:8*(i+1)]
            train_y2[bc, int(grid_y),int(grid_x),int(best_anchor), 0:4] = x,y,w,h
            train_y2[bc, int(grid_y),int(grid_x),int(best_anchor), 4] = 1.
            train_y2[bc, int(grid_y),int(grid_x),int(best_anchor), 5+ int(classid)] = 0.9 
            #Class label smoothing, use 0.9 instead of 1.0 in order to mitigate overfitting.
            bc += 1
            if bc == nb_train:
                break
        train_y0 = np.reshape(train_y0, (nb_train, grids[0][1], grids[0][0], NB_BOX*(4+1+NB_CLASS)))
        train_y1 = np.reshape(train_y1, (nb_train, grids[1][1], grids[1][0], NB_BOX*(4+1+NB_CLASS)))
        train_y2 = np.reshape(train_y2, (nb_train, grids[2][1], grids[2][0], NB_BOX*(4+1+NB_CLASS)))

        return(train_x, [train_y0, train_y1, train_y2])