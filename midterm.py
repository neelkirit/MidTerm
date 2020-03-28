import cv2
import numpy as np
import sys
import os

blob_size = (416, 416)
scale_factor = 1 / 255.0
mean = (0, 0, 0)
nms_threshold = 0.4


def object_detection(image_folder, output_folder, threshold=0.4):
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()

    frame_count = 0
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    for lol in os.listdir(image_folder):
        img = cv2.imread(os.path.join(image_folder, lol))

        height, width, channels = img.shape

        # Create a blob
        blob = cv2.dnn.blobFromImage(img, scale_factor, blob_size, mean, swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Create rectangles for visualisation
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > threshold:
                    # Object found
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Set coordinates of bounding box
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, threshold, nms_threshold)

        frame_count += 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y - 10), font, 1, color, 1)

        cv2.imwrite(output_folder + '/' + str(frame_count) + '.jpg', img)


if __name__ == "__main__":
    object_detection(sys.argv[1], sys.argv[2], float(sys.argv[3]))
