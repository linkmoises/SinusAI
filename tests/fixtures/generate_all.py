#!/usr/bin/env python3
"""Genera las fixtures sintéticas de señal (WFDB) e imagen (PNG/JPG) de `tests/fixtures/`.

Reproducible: todas las semillas son fijas y fijadas en el código. No usa datos del
dataset PTB-XL (el corte real se genera con `corte_real/make_cut.py`).

Dependencias: neurokit2, wfdb, matplotlib, numpy (ver `requirements.txt`).

Uso:
    python tests/fixtures/generate_all.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import neurokit2 as nk  # noqa: E402
import numpy as np  # noqa: E402
import wfdb  # noqa: E402

FIXTURES = Path(__file__).resolve().parent

FS = 100
DURATION_S = 10
N_SAMPLES = FS * DURATION_S
CANON_LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

def wfdb_kwargs(n_leads: int) -> dict:
    """Convenio WFDB idéntico a PTB-XL: int16, ganancia 1000 -> mV, baseline 0."""
    return dict(
        fmt=["16"] * n_leads, adc_gain=[1000.0] * n_leads, baseline=[0] * n_leads,
    )


def synth_ecg12(seed=7, heart_rate=72, noise=0.0) -> np.ndarray:
    """ECG sintético, 12 derivaciones (mV), orden canónico, 1000 muestras x 100 Hz."""
    rng = np.random.default_rng(seed)

    def _lead():
        return nk.ecg_simulate(
            duration=DURATION_S, sampling_rate=FS, heart_rate=heart_rate,
            noise=noise, random_state=int(rng.integers(0, 2 ** 31)), method="ecgsyn",
        )

    i, ii, iii = _lead(), _lead(), _lead()
    avr = -(i + ii) / 2.0
    avl = (i - iii) / 2.0
    avf = (ii + iii) / 2.0
    v = [_lead() * (0.8 + 0.04 * k) for k in range(6)]
    return np.column_stack([i, ii, iii, avr, avl, avf] + v)


def write_wfdb(record_name: str, sig_mv: np.ndarray, lead_names: list[str]):
    out_dir = FIXTURES / record_name
    out_dir.mkdir(exist_ok=True)
    wfdb.wrsamp(
        record_name, fs=FS, units=["mV"] * len(lead_names), sig_name=lead_names,
        p_signal=sig_mv, **wfdb_kwargs(len(lead_names)), write_dir=str(out_dir),
    )
    for ext in ("hea", "dat"):
        assert (out_dir / f"{record_name}.{ext}").exists(), out_dir


def make_synthetic_signals():
    # --- caso nominal: 12 x 100 Hz x 10 s, sin degradación --------------------
    write_wfdb("nominal", synth_ecg12(seed=7), CANON_LEADS)

    # --- caso degradada: derivación I faltante, resto nominal (11 derivaciones) ---
    sig = synth_ecg12(seed=7)
    deg_lead = [l for l in CANON_LEADS if l != "I"]
    write_wfdb("degradada", np.delete(sig, 0, axis=1), deg_lead)

    # --- caso fallo total: header válido + 12 trazados planos (extracción imposible) ---
    write_wfdb("fallo_total", np.zeros((N_SAMPLES, 12), dtype=float), CANON_LEADS)

    # --- caso no-EKG: header ilegible (rechazo), sin .dat utilizable ------------
    nog = FIXTURES / "no_ekg"
    nog.mkdir(exist_ok=True)
    (nog / "no_ekg.hea").write_text(
        "esto NO es una cabecera WFDB valida @#$%%\n"
        "curso/2026/septiembre.jpg 12 100 1000\n", encoding="utf-8",
    )
    rng = np.random.default_rng(11)
    (nog / "no_ekg.dat").write_bytes(rng.bytes(48))


def make_toy_images():
    """Imágenes de juguete para el carril de imagen: 12 trazados (png/jpg), sin 12 y no-ECG."""
    out = FIXTURES / "imagen"
    out.mkdir(exist_ok=True)

    sig = synth_ecg12(seed=7, heart_rate=70, noise=0.003)
    t = np.arange(N_SAMPLES) / FS

    def _draw(n_traces: int):
        fig, ax = plt.subplots(figsize=(5, 3), dpi=110)
        ax.set_facecolor("white")
        for c in range(4):
            for r in range(3):
                k = r * 4 + c
                if k >= n_traces:
                    break
                y = sig[:N_SAMPLES, k]
                ax.plot(t, y + y.max() * (6 - k), color="black", linewidth=0.8)
        ax.set_xlim(0, t.max())
        ax.axis("off")
        fig.tight_layout(pad=0.1)
        return fig

    fig = _draw(12)
    fig.savefig(out / "toy_ecg_12_lead.png", bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(out / "toy_ecg_12_lead.jpg", bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)

    fig = _draw(3)
    fig.savefig(out / "toy_tres_trazos.png", bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)

    rng = np.random.default_rng(3)
    noise_img = (rng.normal(128, 40, (220, 330, 3)).clip(0, 255)).astype(np.uint8)
    plt.imsave(out / "toy_no_ecg.png", noise_img)


if __name__ == "__main__":
    make_synthetic_signals()
    make_toy_images()
    print("fixtures sintéticas generadas en", FIXTURES)