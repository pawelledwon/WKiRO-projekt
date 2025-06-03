import pandas as pd
import numpy as np

def load_aruco_topright_positions(path):
    df = pd.read_excel(path) # Wczytanie pliku Excel z pozycjami znaczników
    marker_dict = {}

    for _, row in df.iterrows():
        marker_id = int(row['ID'])
        x, y, z = row['x'], row['y'], row['z']
        marker_dict[marker_id] = np.array([x, y, z], dtype=np.float32)

    return marker_dict # Zwróć słownik {ID: [x, y, z]}

def load_fpv_camera_parameters(csv_path, distortion_coeffs):
    # Wczytaj CSV z danymi kamery
    df = pd.read_csv(csv_path, sep=';')
    
    # Pobierz potrzebne parametry
    fov_x_deg = df.at[0, 'fov x']
    fov_y_deg = df.at[0, 'fov y']
    width = df.at[0, 'width']
    height = df.at[0, 'height']

    # Zamień kąty FOV na radiany
    fov_x = np.deg2rad(fov_x_deg)
    fov_y = np.deg2rad(fov_y_deg)

    # Oblicz ogniskowe na podstawie FOV i rozdzielczości
    f_x = width / (2 * np.tan(fov_x / 2))
    f_y = height / (2 * np.tan(fov_y / 2))

    # Wyznacz środek obrazu (punkt główny)
    c_x = width / 2
    c_y = height / 2

    # Zbuduj macierz wewnętrzną kamery
    camera_matrix = np.array([
        [f_x, 0,   c_x],
        [0,   f_y, c_y],
        [0,   0,   1]
    ], dtype=np.float32)

    # Jeśli podano współczynniki dystorsji – użyj ich, w przeciwnym razie przyjmij zerowe
    if distortion_coeffs is not None:
        k1, k2, p1, p2 = distortion_coeffs
        dist_coeffs = np.array([k1, k2, p1, p2, 0.0], dtype=np.float32)
    else:
        dist_coeffs = np.zeros(5, dtype=np.float32)

    return camera_matrix, dist_coeffs # Zwróć macierz kamery i dystorsję

