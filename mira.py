import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simulatore linea di mira", layout="centered")

st.title("Proiezione sul riser del punto di impatto")

# Immagine illustrativa
st.image("mira.png", caption="Schema: occhio, cocca, riser e proiezione sul riser", use_container_width=True)

# Parametri
o = st.number_input("Distanza occhio–cocca o-c (m)", min_value=0.01, max_value=1.0, value=0.10, step=0.005)
t = st.number_input("Distanza cocca–riser c-r (m)", min_value=0.01, max_value=2.0, value=0.70, step=0.01)
d = st.number_input("Drop (m)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
x_drop = st.number_input("Distanza bersaglio (m)", min_value=0.0, max_value=50.0, value=20.0, step=1.0)

# Funzione y(x) in cm
def y_cm(x, o, t, d=0.0):
    u = (o + d) / (t + x)
    y_calc = 100 * x * (u / np.sqrt(1 + u**2))
    return y_calc - d*100   # sottraggo il drop espresso in cm
    
# Curva laser (d=0)
x_vals = np.linspace(0, 50, 1000)
y_vals_laser = y_cm(x_vals, o, t, 0.0)

# Punto rosso (d>0 a distanza x_drop)
y_drop_point = y_cm(x_drop, o, t, d)

# Limiti dinamici Y
ymin = min(np.min(y_vals_laser), y_drop_point)
ymax = max(np.max(y_vals_laser), y_drop_point) * 1.05

# Grafico
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_vals, y_vals_laser, label="Proiezione laser (d=0)", color="blue")
ax.scatter([x_drop], [y_drop_point], color="red", s=60, label="Proiezione impatto con drop")
ax.set_xlim(0, 50)
ax.set_ylim(ymin, ymax)
ax.set_xlabel("Distanza bersaglio (m)")
ax.set_ylabel("Proiezione sul riser (cm)")
ax.set_title("Curva della proiezione sul riser della linea di mira-tiro teso")
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend()
st.pyplot(fig)

# Testo con Y calcolata al drop
st.subheader("Proiezione del punto di impatto con drop")
st.write(f"A distanza **{x_drop:.1f} m**, con drop = {d:.2f} m → proiezione sul riser = **{y_drop_point:.2f} cm**")

# Tabella valori per la curva laser (d=0), da 0 a 50 m passo 5
x_table = np.arange(0, 51, 5)
y_table = y_cm(x_table, o, t, 0.0)
df = pd.DataFrame({"Distanza (m)": x_table, "Proiezione sul riser (cm)": y_table})

# Rimuovere la colonna indice
df.index = [''] * len(df)

st.subheader("Tabella valori (tiro teso, d=0)")
st.dataframe(df, use_container_width=True)




