import numpy as np 
from ultralytics import YOLO
import cv2 
import time
import functions
from sort.sort import * 
from functions import get_car, read_license_plate, write_csv, estimate_speed

results = {}
tracker = Sort()

# model 
model = YOLO("yolov8n.pt")
license_detect = YOLO("best.pt")

# video 
cap = cv2.VideoCapture('sample.mp4')
video_fps = cap.get(cv2.CAP_PROP_FPS)
vehicles = [2, 3, 5, 7]
frame_nmr = -1

# read frame
prev_frame = None 
prev_result = None 
ret = True 
while ret :
    frame_nmr +=1 
    ret, frame  = cap.read() # đọc khung hình 
    if ret :
        results[frame_nmr] = {}
        # vehicles detection 
        detections = model(frame)[0] 
        detections_list = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_list.append([x1,y1,x2,y2,score])

    # vehicle_tracking 
    track_ids = tracker.update(np.asarray(detections_list))
    
    # nhan dang bien so 
    license_plates = license_detect(frame)[0]
    license_plates_list = []
    for license_plate in license_plates.boxes.data.tolist():
        x1,y1,x2,y2,score, class_id = license_plate
        license_plates_list.append([x1,y1,x2,y2,score])
    
        # gan bien cho xe tuong ung 
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)    

        if car_id != -1 :
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)] # cat bien so 

            # xu ly bien so 
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

            # doc bien so 
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

            if license_plate_text is not None:
                car_data = {
                    'locations': track_ids,
                    'license_plate': [x1, y1, x2, y2], 
                }
                car_speed = estimate_speed(car_id, car_data) # tinh toc do
                results[frame_nmr][car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'car_speed': car_speed['speed_label'],
                        'license_plate': {'bbox': [x1, y1, x2, y2],
                                          'text': license_plate_text,
                                          'bbox_score': score,
                                          'text_score': license_plate_text_score}} 
write_csv(results, './speed_test.csv')    