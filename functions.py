import string 
import easyocr 
import math 
from collections import deque 
import re 

data_deque = {}
speed_line_queue = {}

reader = easyocr.Reader(['en'], gpu = False)

char_to_int_dict = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

int_to_char_dict = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def get_car(license_plate, vehicle_track_ids):
    # tra về 1 cái tuple gồm toạ độ và id của xe 
    x1, y1, x2, y2 , score, class_id = license_plate
    foundID = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j 
            foundID = True 
            break 
    if foundID :
        return vehicle_track_ids[car_indx]
    return -1, -1, -1, -1, -1

def license_complies_format(text):
    # kiem tra xem cai bien so xe co dung khong 
    if len(text) != 7 :
        return False 
    if (text[0] in string.ascii_uppercase or text[0] in int_to_char_dict.keys()) and \
            (text[1] in string.ascii_uppercase or text[1] in int_to_char_dict.keys()) and \
            (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in char_to_int_dict.keys()) and \
            (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in char_to_int_dict.keys()) and \
            (text[4] in string.ascii_uppercase or text[4] in int_to_char_dict.keys()) and \
            (text[5] in string.ascii_uppercase or text[5] in int_to_char_dict.keys()) and \
            (text[6] in string.ascii_uppercase or text[6] in int_to_char_dict.keys()):
        return True
    else:
        return False


def format_license(text):
    license_plate_ = ''
    mapping = {0: int_to_char_dict, 1: int_to_char_dict, 4: int_to_char_dict, 5: int_to_char_dict, 6: int_to_char_dict,
               2: char_to_int_dict, 3: char_to_int_dict}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_

def read_license_plate(license_plate_crop):
    # trả lại 1 tuple chứa cái biển sô xe và độ chính xác của nó 
    detections = reader.readtext(license_plate_crop)
    for detection in detections :
        bbox, text, score = detection 
        text = text.upper().replace(' ','')
        if license_complies_format(text):
            return format_license(text), score
    
    return None , None 

def estimatespeed(Location1,Location2):
    d_pixel = math.sqrt(math.pow(Location2[0] - Location1[0], 2) + math.pow(Location2[1] - Location1[1], 2))
    ppm = 8
    d_meters = d_pixel / ppm
    time_constant = 15 * 3.6
    speed = d_meters * time_constant

    return int(speed)

def estimate_speed(car_id, car_data):
    global data_deque, speed_line_queue
    track_ids = car_data['locations']
    if car_id not in track_ids[:, -1]:
        if car_id in data_deque:
            data_deque.pop(car_id)

    x1, y1, x2, y2 = car_data['license_plate']
    center = (int((x2 + x1) / 2), int((y2 + y1) / 2))
    if car_id not in data_deque:
        data_deque[car_id] = deque(maxlen=64)
        speed_line_queue[car_id] = []

    data_deque[car_id].appendleft(center)
    if len(data_deque[car_id]) >= 2:
        object_speed = estimatespeed(data_deque[car_id][1], data_deque[car_id][0])
        speed_line_queue[car_id].append(object_speed)
    speed_label = "No speed data available"
    if car_id in speed_line_queue and len(speed_line_queue[car_id]) > 0:
        speed_label = str(sum(speed_line_queue[car_id]) // len(speed_line_queue[car_id])) + "km/h"

    return {'speed_label': speed_label, 'license_plate_info': None}


def extract_numeric_values(string):
    def decode_bytes(string):
        if isinstance(string, bytes):
            return string.decode('utf-8')
        elif isinstance(string, str):
            return string
        elif isinstance(string, list):
            return [decode_bytes(item) for item in string]
        elif isinstance(string, tuple):
            return tuple(decode_bytes(item) for item in string)
        elif isinstance(string, dict):
            return {decode_bytes(key): decode_bytes(value) for key, value in string.items()}
        else:
            return string

    decoded_data = decode_bytes(string)

    pattern = r'\d+'

    numeric_values = re.findall(pattern, decoded_data)

    numeric_values = [float(value) if '.' in value else int(value) for value in numeric_values]

    return numeric_values


def write_csv(results, output_path):
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox', 'car_speed',
                                                   'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                   'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                        'license_plate' in results[frame_nmr][car_id].keys() and \
                        'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                               car_id,
                                                               '[{} {} {} {}]'.format(
                                                                   results[frame_nmr][car_id]['car']['bbox'][0],
                                                                   results[frame_nmr][car_id]['car']['bbox'][1],
                                                                   results[frame_nmr][car_id]['car']['bbox'][2],
                                                                   results[frame_nmr][car_id]['car']['bbox'][3]),
                                                               results[frame_nmr][car_id]['car_speed'],
                                                               '[{} {} {} {}]'.format(
                                                                   results[frame_nmr][car_id]['license_plate']['bbox'][
                                                                       0],
                                                                   results[frame_nmr][car_id]['license_plate']['bbox'][
                                                                       1],
                                                                   results[frame_nmr][car_id]['license_plate']['bbox'][
                                                                       2],
                                                                   results[frame_nmr][car_id]['license_plate']['bbox'][
                                                                       3]),
                                                               results[frame_nmr][car_id]['license_plate'][
                                                                   'bbox_score'],
                                                               results[frame_nmr][car_id]['license_plate']['text'],
                                                               results[frame_nmr][car_id]['license_plate'][
                                                                   'text_score'])
                            )
        f.close()