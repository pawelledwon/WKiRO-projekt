import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt

# Wczytaj dane
ref_df = pd.read_csv("DPJAIT/Real_Data/R18_D1_A/GX010280_REF_POS.csv")
our_df = pd.read_csv("my_camera_poses.csv")

# Dopasuj długość
min_len = min(len(ref_df), len(our_df))
ref_df = ref_df[:min_len]
our_df = our_df[:min_len]

# Osie do porównania
axes = ['x', ' y', ' z']
tvec_keys = ['tvec_x', 'tvec_y', 'tvec_z']

# Utwórz 3 wykresy (jeden na oś)
fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

for i, (axis, tvec_key) in enumerate(zip(axes, tvec_keys)):
    ref_vals = medfilt(ref_df[axis].values[:min_len], kernel_size=5)
    est_vals = medfilt(our_df[tvec_key].values[:min_len], kernel_size=5)

    abs_error = np.abs(ref_vals - est_vals)
    mean_error = np.mean(abs_error)
    max_error = np.max(abs_error)

    axs[i].plot(ref_vals, label=f'Reference {axis.upper()}', linestyle='--')
    axs[i].plot(est_vals, label=f'Estimated {axis.upper()} ({tvec_key})', linestyle='-')
    axs[i].fill_between(range(min_len), ref_vals, est_vals, color='red', alpha=0.2, label='Absolute Error')

    axs[i].set_ylabel(f'Pozycja {axis.upper()} [mm]')
    axs[i].legend()
    axs[i].grid(True)
    axs[i].set_title(
        f'Oś {axis.upper()}: Średni błąd = {mean_error:.2f} mm, Maksymalny błąd = {max_error:.2f} mm'
    )

axs[-1].set_xlabel("Numer klatki")
plt.suptitle("Porównanie pozycji XYZ: Referencja vs Estymacja", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
