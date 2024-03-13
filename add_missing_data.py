import csv
import numpy as np
from scipy.interpolate import interp1d
from functions import extract_numeric_values

def interpolate_bounding_boxes(data):
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    car_ids = np.array([int(float(row['car_id'])) for row in data])
    car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split())) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in data])
    speeds_list = [extract_numeric_values(row['car_speed']) for row in data]

    interpolated_data = []
    unique_car_ids = np.unique(car_ids)
    for car_id in unique_car_ids:

        frame_numbers_ = [p['frame_nmr'] for p in data if int(float(p['car_id'])) == int(float(car_id))]
        print(frame_numbers_, car_id)

        car_mask = car_ids == car_id
        car_frame_numbers = frame_numbers[car_mask]
        car_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []
        speeds_interpolated = []

        first_frame_number = car_frame_numbers[0]
        last_frame_number = car_frame_numbers[-1]

        for i in range(len(car_bboxes[car_mask])):
            frame_number = car_frame_numbers[i]
            car_bbox = car_bboxes[car_mask][i]
            license_plate_bbox = license_plate_bboxes[car_mask][i]

            if len(speeds_list[i]) > 0:
                speed = speeds_list[i][0]
            else:
                speed = 0 
            if i > 0:
                prev_frame_number = car_frame_numbers[i - 1]
                prev_car_bbox = car_bboxes_interpolated[-1]
                prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]
                prev_speed = speeds_interpolated[-1]

                if frame_number - prev_frame_number > 1:
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                    interp_func = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
                    interpolated_car_bboxes = interp_func(x_new)
                    interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0,
                                           kind='linear')
                    interpolated_license_plate_bboxes = interp_func(x_new)
                    interp_func_speed = interp1d(x, [prev_speed, speed], kind='linear')
                    interpolated_speed = interp_func_speed(x_new)

                    car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
                    license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])
                    speeds_interpolated.extend(interpolated_speed[1:])

            car_bboxes_interpolated.append(car_bbox)
            license_plate_bboxes_interpolated.append(license_plate_bbox)
            speeds_interpolated.append(speed)


        for i in range(len(car_bboxes_interpolated)):
            frame_number = first_frame_number + i
            row = {}
            row['frame_nmr'] = str(frame_number)
            row['car_id'] = str(car_id)
            row['car_bbox'] = ' '.join(map(str, car_bboxes_interpolated[i]))
            row['license_plate_bbox'] = ' '.join(map(str, license_plate_bboxes_interpolated[i]))
            row['car_speed'] = str(speeds_interpolated[i])

            data_indices = np.where((frame_numbers == frame_number) & (car_ids == car_id))[0]
            if len(data_indices) > 0:
                
                original_row = data[data_indices[0]]
                row['license_plate_bbox_score'] = original_row.get('license_plate_bbox_score', '0')
                row['license_number'] = original_row.get('license_number', '0')
                row['license_number_score'] = original_row.get('license_number_score', '0')

            else:
                
                row['license_plate_bbox_score'] = '0'
                row['license_number'] = '0'
                row['license_number_score'] = '0'

            interpolated_data.append(row)

    return interpolated_data

with open('speed_test.csv', 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)
interpolated_data = interpolate_bounding_boxes(data)

header = ['frame_nmr', 'car_id', 'car_bbox', 'car_speed', 'license_plate_bbox', 'license_plate_bbox_score',
          'license_number', 'license_number_score']
with open('speed_test_interpolated.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(interpolated_data)