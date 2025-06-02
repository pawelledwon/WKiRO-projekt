import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scripts.aruco_localizer import load_aruco_positions, load_fpv_camera_parameters

# Parametry
video_path = "DPJAIT/Real_Data/R18_D1_A/GX010280.MP4"
marker_length = 20 
aruco_excel_path = "DPJAIT/Real_Data/R18_D1_A/ArUco_3D.xlsx"
fpv_camera_data_path = "DPJAIT/Real_Data/R18_D1_A/fpv_camera_data.csv"
output_csv = "my_camera_poses.csv"
distortion = [-0.1577, 0.08576, -0.00068, 0.00008]


# Wczytaj znane pozycje 3D markerów
marker_dict = load_aruco_positions(aruco_excel_path, marker_length=marker_length)

# Kamera FPV – przykładowe parametry wewnętrzne
camera_matrix, dist_coeffs = load_fpv_camera_parameters(fpv_camera_data_path, distortion)

# ArUco
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

cap = cv2.VideoCapture(video_path)

poses = []
frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco_detector.detectMarkers(gray)

    if ids is not None and len(ids) >= 4:
        object_points = []
        image_points = []

        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in marker_dict:
                obj_pts = marker_dict[marker_id]
                img_pts = corners[i][0]

                object_points.extend(obj_pts)
                image_points.extend(img_pts)

        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)

        if len(object_points) >= 4:
            success, rvec, tvec = cv2.solvePnP(
                object_points, image_points,
                camera_matrix, dist_coeffs
            )
            if success:
                poses.append({
                    "frame": frame_id,
                    "rvec_x": rvec[0][0],
                    "rvec_y": rvec[1][0],
                    "rvec_z": rvec[2][0],
                    "tvec_x": tvec[0][0],
                    "tvec_y": tvec[1][0],
                    "tvec_z": tvec[2][0],
                })

    frame_id += 1

cap.release()

# Zapisz do CSV
df = pd.DataFrame(poses)
df.to_csv(output_csv, index=False)
