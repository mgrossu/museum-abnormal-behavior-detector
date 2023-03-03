import cv2
import numpy as np
import sys

img = cv2.imread(cv2.samples.findFile("data/mall.jpg"))

if img is None:
    sys.exit("Could not read the image.")

classes = None
with open("yolov7-coco/coco.names", "r") as f:
 classes = [line.strip() for line in f.readlines()]

#print(classes)

#net
net = cv2.dnn.readNet("yolov7-coco/yolov7-tiny.weights", "yolov7-coco/yolov7-tiny.cfg")
blob = cv2.dnn.blobFromImage(img, 0.00392, (640,640), (0,0,0), True, crop=False)
net.setInput(blob)

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

    return output_layers

outs = net.forward(get_output_layers(net))

print(outs)