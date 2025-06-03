import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scripts.aruco_localizer import load_aruco_topright_positions, load_fpv_camera_parameters 

# Parametry
video_path = "DPJAIT/Real_Data/R18_D1_A/GX010280.MP4"
aruco_excel_path = "DPJAIT/Real_Data/R18_D1_A/ArUco_3D.xlsx"
fpv_camera_data_path = "DPJAIT/Real_Data/R18_D1_A/fpv_camera_data.csv"
output_csv = "my_camera_poses.csv"
distortion = [-0.1577, 0.08576, -0.00068, 0.00008]


# Wczytaj znane pozycje 3D markerów
marker_dict = load_aruco_topright_positions(aruco_excel_path)

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
                obj_pt = marker_dict[marker_id]
                img_pt = corners[i][0][1]  # [1] to top-right róg w OpenCV

                object_points.append(obj_pt)
                image_points.append(img_pt)

        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)

        if len(object_points) >= 4:
            success, rvec, tvec = cv2.solvePnP(
                object_points, image_points,
                camera_matrix, dist_coeffs
            )
            if success:
                    tx, ty, tz = tvec.flatten()
                    # Zamiana z układu OpenCV na referencyjny (przykład dopasowania do Vicon)
                    corrected_x = -tx
                    corrected_y = -tz
                    corrected_z = ty

                    poses.append({
                        "frame": frame_id,
                        "rvec_x": rvec[0][0],
                        "rvec_y": rvec[1][0],
                        "rvec_z": rvec[2][0],
                        "tvec_x": corrected_x,
                        "tvec_y": corrected_y,
                        "tvec_z": corrected_z,
                    })

    frame_id += 1

cap.release()

# Zapisz do CSV
df = pd.DataFrame(poses)
df.to_csv(output_csv, index=False)
