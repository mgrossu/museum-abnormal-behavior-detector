#!/usr/bin/env python3
# coding=utf-8
# Autor: Marius Iustin Grossu xgross10
# Detection of Anomalous Behavior of Visitors in Museum Exhibitions
# This is the demo module of the application

import numpy as np
import cv2
from  detector import init_net_yolov7, draw_bounding_box_on_person, get_output_layers


net = init_net_yolov7()

def demo_run(file):
    """Demonstrate the capabilties of the application

    Keyword arguments:
    file -- input file caintaining the video to demonstrate the application 
    """
    video = cv2.VideoCapture(file)

    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            print("Can't receive frame (stream end?). Existing ...")
            break
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640,640), swapRB=True, crop=False)
        net.setInput(blob)

        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        Width = frame.shape[1]
        Height = frame.shape[0] 
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.1:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)

        for i in indices:
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]

            draw_bounding_box_on_person(frame, class_ids[i], round(x), round(y), round(x+w), round(y+h))

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
        # When everything done, release the capture
    video.release()
    cv2.destroyAllWindows()

