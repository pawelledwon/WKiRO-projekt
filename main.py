# Importowanie bibliotek
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scripts.aruco_localizer import load_aruco_topright_positions, load_fpv_camera_parameters 

# Ścieżki do plików z danymi
video_path = "DPJAIT/Real_Data/R18_D1_A/GX010280.MP4"
aruco_excel_path = "DPJAIT/Real_Data/R18_D1_A/ArUco_3D.xlsx"
fpv_camera_data_path = "DPJAIT/Real_Data/R18_D1_A/fpv_camera_data.csv"
output_csv = "my_camera_poses.csv"
distortion = [-0.1577, 0.08576, -0.00068, 0.00008]


# Współczynniki dystorsji obiektywu (zniekształcenia soczewki kamery)
marker_dict = load_aruco_topright_positions(aruco_excel_path)

# Wczytanie wewnętrznych parametrów kamery FPV
camera_matrix, dist_coeffs = load_fpv_camera_parameters(fpv_camera_data_path, distortion)

# Ustawienie słownika znaczników ArUco (4x4, 100 znaczników)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

# Otwarcie pliku wideo
cap = cv2.VideoCapture(video_path)

poses = [] # Lista do przechowywania wyników pozycji
frame_id = 0 # Numer bieżącej klatki

# Przetwarzanie każdej klatki wideo
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret: # Jeśli nie udało się odczytać klatki – koniec filmu
        break
    
    # Konwersja obrazu z RGB (BGR) do odcieni szarości – wymagane przez detektor ArUco
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Wykrycie znaczników ArUco
    corners, ids, _ = aruco_detector.detectMarkers(gray)

    # Jeżeli wykryto co najmniej 4 znaczniki
    if ids is not None and len(ids) >= 4:
        object_points = [] # Pozycje 3D znaczników (świat rzeczywisty)
        image_points = [] # Pozycje 2D znaczników na obrazie

        # Iteracja po wszystkich wykrytych znacznikach
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in marker_dict:
                obj_pt = marker_dict[marker_id] # Znana pozycja 3D
                img_pt = corners[i][0][1]  # Pozycja 2D górnego prawego rogu znacznika

                object_points.append(obj_pt)
                image_points.append(img_pt)

        # Konwersja do tablic NumPy
        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)

        # Jeśli nadal mamy co najmniej 4 punkty
        if len(object_points) >= 4:
            # Oblicz pozycję i orientację kamery (solvePnP = Perspective-n-Point)
            success, rvec, tvec = cv2.solvePnP(
                object_points, image_points,
                camera_matrix, dist_coeffs
            )

            if success:
                    tx, ty, tz = tvec.flatten()

                    # Korekta układu współrzędnych z OpenCV na system referencyjny (np. zgodny z Vicon)
                    corrected_x = -tx
                    corrected_y = -tz
                    corrected_z = ty

                    # Zapisz wyniki pozycji i orientacji kamery
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
