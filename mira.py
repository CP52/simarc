import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simulatore linea di mira", layout="centered")

st.title("Proiezione sul riser della linea di mira")

# Illustrazione
st.image("mira.png", caption="Schema concettuale: occhio, cocca, punta e proiezione sul riser", use_column_width=True)

# Parametri
o = st.number_input("Distanza occhio–cocca (m)", min_value=0.01, max_value=1.0, value=0.11, step=0.01)
t = st.number_input("Distanza cocca–punta (m)", min_value=0.01, max_value=2.0, value=0.70, step=0.01)

# Funzione y(x) in cm
def y_cm(x, o, t):
    u = o / (t + x)
    return 100 * x * (u / np.sqrt(1 + u**2))

# Grafico
x_vals = np.linspace(0, 50, 1000)
y_vals = y_cm(x_vals, o, t)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_vals, y_vals, label="Proiezione sul riser")
ax.set_xlim(0, 50)
ax.set_ylim(8, 12)
ax.set_xlabel("Distanza (m)")
ax.set_ylabel("Proiezione sul riser (cm)")
ax.set_title("Curva della proiezione sul riser della linea di mira")
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend()
st.pyplot(fig)

# Tabella con step di 5 m
x_table = np.arange(5, 51, 5)
y_table = y_cm(x_table, o, t)
df = pd.DataFrame({"Distanza (m)": x_table, "Proiezione sul riser (cm)": y_table})
st.subheader("Tabella valori")
st.dataframe(df, use_container_width=True)


