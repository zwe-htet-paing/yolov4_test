from keras.models import Model
import struct
import numpy as np

class WeightReader:
    def __init__(self, weight_file):
        with open(weight_file, 'rb') as w_f:
            major,      = struct.unpack('i', w_f.read(4))
            minor,      = struct.unpack('i', w_f.read(4))
            revison,    = struct.unpack('i', w_f.read(4))

            if (major*10 + minor) >=2 and major < 1000 and minor < 1000:
                print("reading 64 bytes")
                w_f.read(4)
            else: 
                print("reading 32 bytes")
                w_f.read(4)

            transpose = (major > 1000) or (minor > 1000)
            binary = w_f.read()

        self.offset = 0
        self.all_weights = np.frombuffer(binary, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]

    def load_weight(self, model):
        count = 0
        ncount = 0
        for i in range(161):
            try:
                conv_layer = model.get_layer('convn_' + str(i))
                filter = conv_layer.kernel.shape[-1]
                nweights = np.prod(conv_layer.kernel.shape) # kernel*kernel*c*filter

                print("loading weights of convolution #" + str(i) + "-nb parameters: " + str(nweights+filter))

                if i in [138, 149, 160]:
                    print("Special processing for layer "+ str(i))
                    bias = self.read_bytes(filters)
                    weights = self.read_bytes(nweights)

                else:
                    bias  = self.read_bytes(filter) # bias
                    scale = self.read_bytes(filter) # scale
                    mean  = self.read_bytes(filter) # mean
                    var   = self.read_bytes(filter) # variance
                    weights = self.read_bytes(nweights) # weights
                    
                    bias = bias - scale * mean / (np.sqrt(var + 0.00001)) # normalize bias
                    A = scale / (np.sqrt(var + 0.00001))
                    A = np.expand_dims(A, axis=0)
                    weights = weights* A.T
                    weights = np.reshape(weights, (nweights))
                
                weights = weights.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                weights = weights.transpose([2,3,1,0])

                if len(conv_layer.get_weights()) > 1:
                    a = conv_layer.set_weights([weights, bias])
                else:
                    a = conv_layer.set_weights([weights])

                count = count + 1
                ncount = ncount+ nweights + filter 

            except ValueError:
                print("no convolution #" + str(i))

        print(count, "Conv normalized layers loaded", ncount, "parameters")


    def reset(self):
        self.offset = 0


def read_labels(labels_path):
    with open(labels_path) as f:
        labels = f.readlines()
    labels = [c.strip() for c in labels]
    return labels

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1

    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) -x3

def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xamx])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin

    union = w1*h1 + w2*h2 - intersect
    
    return float(intersect) / union

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        return self.label

    def get_socre(self):
        if self.score == -1:
            self.socre = self.classes[self.get_label()()]
        return self.socre

"""-------------------------------------------------------------------------------------"""
import xml.etree.ElementTree as ET
import pickle
import os

def convert(image_wh, box, grid_w, grid_h, Boxanchor, yolo_id):
    dw = image_wh[0]/grid_w
    dh = image_wh[1]/grid_h
    center_x = (box[0] + box[1])/2.0
    center_x = center_x / dw
    center_y = (box[2] + box[3])/2.0
    center_y = center_y / dh

    grid_x = int(np.floor(center_x))
    grid_y = int(np.floor(center_y))

    if grid_x < grid_w and grid_y < grid_h:
        w = (box[1] - box[0]) / dw
        h = (box[3] - box[2]) / dh

        # find the anchor that best predicts this box
        best_anchor = -1
        max_iou     = -1

        shifted_box = BoundBox(0, 0, w, h)

        for i in range(len(anchors[yolo_id])//2):
            iou = bbox_iou(shifted_box, Boxanchor[i])
            if max_iou < iou:
                best_anchor = i
                max_iou = iou

        return(center_x, center_y, w, h, grid_x, grid_y, best_anchor)
    else:
        return(0, 0, 0, 0, 0, 0, -1)


def convert_annotation(year, image_set, image_id, grid_w, grid_h, Boxanchor, yolo_id, VOC_path):
    infile = open(VOC_path + 'VOC%\\Annotations\\%s.xml'%(year, image_id))
    outfile = open('dataset\\VOC%s_%s\\labels_%s\\%s.txt'%(year, image_set,yolo_id,image_id), 'w')

    tree = ET.parse(infile)
    root = tree.getroot()
    size = root.find('size')
    image_w = int(size.find('width').text)
    image_h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in labels or int(difficult) == 1:
            continue
        cls_id = labels.index(cls)

        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), 
             float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((image_w, image_h), b, grid_w, grid_h, Boxanchor, yolo_id)

        if bb[-1] != -1:
            outfile.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def build_label_files(year, image_set, VOC_path):
    yolo_id = 0

    for grid_w, grid_h in grids:
        print("grid :", grid_w, grid_h)

        Boxanchor= [BoundBox(0, 0, anchors[yolo_id][2*i], anchors[yolo_id][2*i+1]) for i in range(int(len(anchors[yolo_id])//2))]

        if not os.path.exists('dataset\\VOC%s_%s\\labels_%s\\' %(year, image_set, yolo_id)):
            os.makedirs('dataset\\VOC%s_%s\\labels_%s\\' %(year, image_set, yolo_id))

            image_ids = open(VOC_path+ 'VOC%s\\ImageSets\\Main\\%s.txt' %(year, image_set)).read().strip().split()

            for image_id in image_ids:
                convert_annotation(year, image_set, image_id, grid_w, grid_h, Boxanchor, yolo_id, VOC_path)
                yolo_id += 1
                