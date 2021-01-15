from utils import make_yolov4_model
from dataset_utils import WeightReader, read_labels
from keras.models import load_model, Model

# create model
model = make_yolov4_model()

# load weights
weight_reader = WeightReader('models/yolo/yolov4.weights')
weight_reader.load_weight(model)

# view model summary
model.summary()

# save the lodel
model.save('models/yolo/yolov4.h5')

# load the model
yolo_model = load_model("models/yolo/yolov4.h5")

# freeze the backbone
# Fine-tune from this layer onwards
fine_tune_at = "convn_136"

# Freeze all the layers before the 'fine_tune_at' layer
train = False
for  l in yolo_model.layers:
    if l.name == fine_tune_at:
        train = True
    l.trainable = train

# Display the trainable indicator
for l in yolo_model.layers:
    print("layer ", l.name, " trainable: ", l.trainable)

# Load the labels
labels = read_labels('models/yolo/coco_classes.txt')


