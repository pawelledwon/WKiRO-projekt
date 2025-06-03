import pandas as pd
import numpy as np

def load_aruco_topright_positions(path):
    df = pd.read_excel(path)
    marker_dict = {}

    for _, row in df.iterrows():
        marker_id = int(row['ID'])
        x, y, z = row['x'], row['y'], row['z']
        marker_dict[marker_id] = np.array([x, y, z], dtype=np.float32)

    return marker_dict

def load_fpv_camera_parameters(csv_path, distortion_coeffs):
    # Wczytanie danych z CSV
    df = pd.read_csv(csv_path, sep=';')
    
    # Pobranie wartości
    fov_x_deg = df.at[0, 'fov x']
    fov_y_deg = df.at[0, 'fov y']
    width = df.at[0, 'width']
    height = df.at[0, 'height']

    # Konwersja FOV na radiany
    fov_x = np.deg2rad(fov_x_deg)
    fov_y = np.deg2rad(fov_y_deg)

    # Obliczenie ogniskowych w pikselach
    f_x = width / (2 * np.tan(fov_x / 2))
    f_y = height / (2 * np.tan(fov_y / 2))

    # Środek obrazu
    c_x = width / 2
    c_y = height / 2

    # Macierz wewnętrzna kamery
    camera_matrix = np.array([
        [f_x, 0,   c_x],
        [0,   f_y, c_y],
        [0,   0,   1]
    ], dtype=np.float32)

    # Współczynniki dystorsji
    if distortion_coeffs is not None:
        k1, k2, p1, p2 = distortion_coeffs
        dist_coeffs = np.array([k1, k2, p1, p2, 0.0], dtype=np.float32)
    else:
        dist_coeffs = np.zeros(5, dtype=np.float32)

    return camera_matrix, dist_coeffs

