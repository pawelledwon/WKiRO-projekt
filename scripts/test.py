import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Wczytaj dane
ref_df = pd.read_csv("DPJAIT/Real_Data/R18_D1_A/GX010280_REF_POS.csv")
our_df = pd.read_csv("my_camera_poses.csv")

# Sprawdź ile jest wspólnych ramek
min_len = min(len(ref_df), len(our_df))
ref_x = ref_df['x'][:min_len].values
est_x = our_df['tvec_x'][:min_len].values

# Oblicz błąd bezwzględny
abs_error = np.abs(ref_x - est_x)
mean_error = np.mean(abs_error)
max_error = np.max(abs_error)

# Wykres pozycji
plt.figure(figsize=(14, 6))
plt.plot(ref_x, label='Reference X', linestyle='--')
plt.plot(est_x, label='Estimated X (tvec_x)', linestyle='-')
plt.fill_between(range(min_len), ref_x, est_x, color='red', alpha=0.2, label='Absolute Error')

plt.title("Porównanie pozycji X: Referencja vs Estymacja\n"
          f"Średni błąd: {mean_error:.2f} mm, Maksymalny błąd: {max_error:.2f} mm")
plt.xlabel("Numer klatki")
plt.ylabel("Pozycja X [mm]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
