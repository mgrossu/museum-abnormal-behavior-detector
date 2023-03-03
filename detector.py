#!/usr/bin/env python3
# coding=utf-8
# Autor: Marius Iustin Grossu xgross10
# Detection of Anomalous Behavior of Visitors in Museum Exhibitions
# This the module to detect visitors of Museum Exhibitions using YOLOv7

import cv2
import numpy as np
import sys

img = cv2.imread(cv2.samples.findFile("data/mall.jpg"))

if img is None:
    sys.exit("Could not read the image.")


def read_coco_classes():

    classes = None
    with open("yolov7-coco/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    return classes

classes = read_coco_classes()

def init_net_yolov7():

    net = cv2.dnn.readNetFromDarknet("yolov7-coco/yolov7.cfg", "yolov7-coco/yolov7.weights")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

    return net
    

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def draw_bounding_box_on_person(img, class_id, x, y, x_plus_w, y_plus_h):
    
    #check if it is a person detection
    if class_id == 0:
        label = str(classes[class_id])
        cv2.rectangle(img, (x,y), (x_plus_w, y_plus_h), (0, 255, 0), 3)
        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
