"""Figuras de la tesis: arquitectura del framework y discontinuidad a los 65 años."""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

AZUL = "#1c5cab"      # paso 550 de la rampa azul de referencia
TINTA = "#0b0b0b"
TINTA2 = "#52514e"
REJILLA = "#e4e3df"
plt.rcParams.update({
    "font.family": "Liberation Serif", "font.size": 8, "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA,
    "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "savefig.dpi": 400,
})


def caja(ax, x, y, w, h, titulo, detalle, borde=AZUL, relleno="white", estilo="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.018",
                                lw=0.9, ec=borde, fc=relleno, ls=estilo))
    ax.text(x + w / 2, y + h * 0.64, titulo, ha="center", va="center", fontsize=7.6, fontweight="bold", color=TINTA)
    ax.text(x + w / 2, y + h * 0.28, detalle, ha="center", va="center", fontsize=6.6, color=TINTA2)


def flecha(ax, x0, y0, x1, y1, color=TINTA2, estilo="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=color, ls=estilo, shrinkA=0, shrinkB=0,
                                mutation_scale=7))


def etiqueta(ax, x, y, texto):
    ax.text(x, y, texto, ha="left", va="center", fontsize=6.4, style="italic", color=TINTA2)


def arquitectura(salida):
    fig = plt.figure(figsize=(3.4, 3.9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.15)
    ax.axis("off")
    caja(ax, 0.12, 1.02, 0.76, 0.10, "Fuentes heterogéneas", "microdatos, reglas y montos de programas, precios",
         borde=TINTA2, relleno="#f4f4f2")
    flecha(ax, 0.5, 1.02, 0.5, 0.93)
    caja(ax, 0.12, 0.81, 0.76, 0.12, "Módulo 1. Ingesta y armonización", "tipos, llaves, diccionario, procedencia y calidad")
    flecha(ax, 0.30, 0.81, 0.30, 0.67)
    flecha(ax, 0.72, 0.81, 0.72, 0.67)
    etiqueta(ax, 0.315, 0.74, "¿contraste viable?")
    caja(ax, 0.02, 0.53, 0.53, 0.14, "Módulo 2. Causalidad", "especificación, grafo, efecto\ny dictamen de evidencia")
    caja(ax, 0.59, 0.53, 0.39, 0.14, "Módulo 3. Predicción", "riesgo, intervalos\ny atribuciones SHAP")
    flecha(ax, 0.30, 0.53, 0.30, 0.37)
    etiqueta(ax, 0.315, 0.45, "¿efecto identificado?")
    flecha(ax, 0.72, 0.53, 0.72, 0.37)
    etiqueta(ax, 0.735, 0.45, "orden por riesgo")
    caja(ax, 0.12, 0.23, 0.76, 0.14, "Módulo 4. Escenarios con restricciones",
         "presupuesto, capacidad, piso de equidad,\nincertidumbre y robustez")
    flecha(ax, 0.5, 0.23, 0.5, 0.13)
    caja(ax, 0.12, 0.01, 0.76, 0.12, "Fichas de escenarios", "efecto, costo, supuestos y límites de cada opción",
         borde=TINTA2, relleno="#f4f4f2")
    fig.savefig(salida, facecolor="white")
    plt.close(fig)


def ajuste(edad, y):
    b = np.polyfit(edad, y, 1)
    return np.polyval(b, edad)


def discontinuidad(parquet, salida):
    df = pd.read_parquet(parquet, columns=["edad", "pam", "ia_mod_sev", "log_ing_pc"])
    df = df[(df["edad"] >= 58) & (df["edad"] <= 72)]
    medias = df.groupby("edad").agg(pam=("pam", "mean"), ia=("ia_mod_sev", "mean"), ing=("log_ing_pc", "mean"),
                                    n=("pam", "size"))
    print("personas:", int(medias["n"].sum()))
    paneles = [("pam", 100, "Recibe la pensión (%)", "a"), ("ia", 100, "Inseguridad alimentaria (%)", "b"),
               ("ing", 1, "Ingreso per cápita (log)", "c")]
    fig, axes = plt.subplots(3, 1, figsize=(3.4, 4.7), sharex=True)
    for ax, (col, esc, rotulo, letra) in zip(axes, paneles):
        y = medias[col] * esc
        izq, der = medias.index < 65, medias.index >= 65
        ax.axvline(64.5, color=TINTA2, lw=0.7, ls=(0, (3, 2)))
        for m in (izq, der):
            e = medias.index[m].to_numpy(dtype=float)
            ax.plot(e, ajuste(e, y[m].to_numpy()), color=AZUL, lw=1.2)
        ax.plot(medias.index, y, "o", ms=3.8, mfc=AZUL, mec="white", mew=0.5)
        ax.set_ylabel(rotulo, fontsize=8.6)
        ax.grid(axis="y", color=REJILLA, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8.2)
        ax.text(0.01, 0.97, f"({letra})", transform=ax.transAxes, ha="left", va="top", fontsize=8.6, color=TINTA)
    axes[-1].set_xlabel("Edad (años cumplidos)", fontsize=8.6)
    axes[-1].set_xticks(range(58, 73, 2))
    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=0.6)
    fig.savefig(salida, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    destino = sys.argv[1]
    arquitectura(f"{destino}/fig_arquitectura.png")
    discontinuidad(sys.argv[2], f"{destino}/fig_discontinuidad.png")
