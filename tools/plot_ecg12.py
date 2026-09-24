"""
Grafica un registro de PTB-XL (formato WFDB: .hea + .dat) en la
distribución clínica estándar de 12 derivaciones:

    I    aVR   V1   V4
    II   aVL   V2   V5
    III  aVF   V3   V6
    ------- tira de ritmo (II) -------

Requiere: pip install wfdb matplotlib numpy
Uso:      python plot_ecg12.py ruta/al/registro   (sin extensión, ej: records100/00000/00001_lr)
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import wfdb

# Orden clínico estándar (no es el orden en que wfdb suele devolver las señales)
LAYOUT = [
    ["I",  "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III","aVF", "V3", "V6"],
]
RHYTHM_LEAD = "II"

# Escala clínica estándar: 25 mm/s, 10 mm/mV -> con papel de 1mm por cuadro chico
MM_PER_S = 25
MM_PER_MV = 10


def plot_ecg12(record_path: str, out_path: str | None = None, title: str | None = None):
    record = wfdb.rdrecord(record_path)
    fs = record.fs                     # frecuencia de muestreo (100 o 500 Hz en PTB-XL)
    sig = record.p_signal               # (n_muestras, 12) en mV
    names = record.sig_name             # nombres de derivación tal como vienen en el .hea

    idx = {name.upper(): i for i, name in enumerate(names)}

    # Rango vertical común: mismo mm/mV en las 4 filas para que todas compartan escala.
    ymax = float(np.ceil(np.abs(sig).max() * 10) / 10) + 0.2
    ylim = (-ymax, ymax)

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [1, 1, 1, 1]})
    fig.suptitle(title or record_path, fontsize=12, fontweight="bold")

    # --- cuadrícula tipo papel de EKG en cada fila de 4 derivaciones ---
    def draw_grid(ax, n_samples, fs):
        duration_s = n_samples / fs
        ax.set_xlim(0, duration_s)
        ax.set_ylim(ylim)
        # líneas finas cada 0.04s (1mm) y gruesas cada 0.2s (5mm) — versión simplificada
        for t in np.arange(0, duration_s, 0.2):
            ax.axvline(t, color="mistyrose", linewidth=0.6, zorder=0)
        ax.axhline(0, color="mistyrose", linewidth=0.6, zorder=0)
        ax.set_yticks([])
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # --- filas 1-3: 4 derivaciones cortas en paralelo, cada una ocupando 1/4 del ancho ---
    seg_len = sig.shape[0] // 4  # cada derivación de la fila muestra 1/4 del trazado total
    for row, leads in enumerate(LAYOUT):
        ax = axes[row]
        draw_grid(ax, sig.shape[0], fs)
        for col, lead in enumerate(leads):
            start, end = col * seg_len, (col + 1) * seg_len
            t = np.arange(start, end) / fs
            y = sig[start:end, idx[lead.upper()]]
            ax.plot(t, y, color="black", linewidth=0.9)
            ax.text(t[0], ylim[1] * 0.95, lead, fontsize=9, fontweight="bold", va="top")
        ax.set_ylabel("mV" if row == 0 else "")

    # --- fila 4: tira de ritmo completa (derivación II) ---
    ax = axes[3]
    draw_grid(ax, sig.shape[0], fs)
    t = np.arange(sig.shape[0]) / fs
    y = sig[:, idx[RHYTHM_LEAD.upper()]]
    ax.plot(t, y, color="black", linewidth=0.9)
    ax.text(0, ylim[1] * 0.97, f"{RHYTHM_LEAD} (ritmo)", fontsize=9, fontweight="bold", va="top")
    ax.set_xlabel("segundos")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Guardado en {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python plot_ecg12.py ruta/al/registro [salida.png]")
        sys.exit(1)
    record_path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    plot_ecg12(record_path, out)
