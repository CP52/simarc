import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
import math

# --- Costanti ---
G = 9.81
air_density = 1.204
mu_air = 1.81e-5

TIPO_PUNTA_CD_FACTOR = {
    "Slanciata (field/bullet)": 0.95,
    "Standard": 1.0,
    "Broadhead (larga)": 1.15
}

BOW_TYPE_DEFAULT_EFF = {
    "longbow": 0.75,
    "ricurvo": 0.82,
    "compound": 0.87,
    "takedown": 0.80,
}

# --- Funzioni di fisica/utility ---

def reynolds_number(v, diameter):
    d_m = diameter / 1000
    return air_density * v * d_m / mu_air

def realistic_drag_coefficient(v, diameter, angle_of_attack, tip_type, params):
    Re = reynolds_number(v, diameter)
    if Re < 1.2e4:
        Cd = 1.5
    elif Re < 2.0e4:
        Cd = 1.5 + (Re - 1.2e4) / (8e3) * (2.6 - 1.5)
    else:
        Cd = 2.6
    gamma_rad = np.radians(angle_of_attack)
    Cd *= (1 + 4 * gamma_rad ** 2)
    Cd *= TIPO_PUNTA_CD_FACTOR.get(tip_type, 1.0)
    return Cd

def calculate_velocity(params):
    if params['use_measured_v0']:
        return params['v0']
    mass = params['mass'] / 1000.0
    F = params['draw_force'] * 4.44822
    elong = max(0.0, params['draw_length'] - params['brace_height'])
    efficiency = params['efficiency']
    E = efficiency * F * elong
    return np.sqrt(max(0.0, 2 * E / mass))

def simulate_trajectory(angle_deg, params, include_drag=True):
    angle = np.radians(angle_deg)
    mass = params['mass'] / 1000.0
    v0 = calculate_velocity(params)

    A = np.pi * (params['diameter'] / 1000 / 2) ** 2
    dt = 0.001
    x, y = 0, params['launch_height']
    vx, vy = v0 * np.cos(angle), v0 * np.sin(angle)

    X, Y = [x], [y]
    t = 0
    while x <= params['target_distance']:
        v = np.sqrt(vx ** 2 + vy ** 2)
        gamma = np.degrees(np.arctan2(vy, vx)) - angle_deg
        if include_drag:
            Cd = realistic_drag_coefficient(v, params['diameter'], gamma, params['tip_type'], params)
            k = 0.5 * air_density * Cd * A / mass
        else:
            k = 0
        ax = -k * v * vx
        ay = -G - k * v * vy
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        t += dt
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y), v0, t

def get_y_at_x(X, Y, x_target):
    f_interp = interp1d(X, Y, kind='linear', fill_value='extrapolate')
    return float(f_interp(x_target))

def find_optimal_angle(params):
    target_y = params['target_height']
    distance = params['target_distance']
    def objective(angle):
        X, Y, _, _ = simulate_trajectory(angle, params, include_drag=True)
        if len(X) < 2:
            return 1e6
        y_at_target = get_y_at_x(X, Y, distance)
        return abs(y_at_target - target_y)
    res = minimize_scalar(objective, bounds=(-20, 45), method='bounded')
    return res.x

def calcola_quota_uscita_posturale(params, angle_rad):
    anchor_length = params['anchor_length']
    quota_uscita_neutra = params['launch_height_neutral']
    pelvis_height = params['pelvis_height']
    y_pivot = quota_uscita_neutra - pelvis_height
    dx = anchor_length
    dy = pelvis_height
    raggio_rot = np.sqrt(dx**2 + dy**2)
    phi0 = np.arcsin(dy / raggio_rot)
    y_launch = y_pivot + raggio_rot * np.sin(angle_rad + phi0)
    return y_launch

def trova_angolo_convergenza(params, tol_angle=0.01, tol_height=0.001, max_iter=20):
    quota_uscita = params['launch_height_neutral']
    angolo = 0
    for _ in range(max_iter):
        params['launch_height'] = quota_uscita
        angolo_nuovo = find_optimal_angle(params)
        y_launch_new = calcola_quota_uscita_posturale(params, np.radians(angolo_nuovo))
        if abs(angolo_nuovo - angolo) < tol_angle and abs(y_launch_new - quota_uscita) < tol_height:
            return angolo_nuovo, y_launch_new
        angolo, quota_uscita = angolo_nuovo, y_launch_new
    return angolo, quota_uscita

# --- INTERFACCIA STREAMLIT ---

st.title("🏹 Simulatore avanzato tiro con l'arco")

# Sidebar per parametri extra
st.sidebar.header("Curva Drop")
d_min = st.sidebar.number_input("Distanza minima (m)", 5, 100, 10)
d_max = st.sidebar.number_input("Distanza massima (m)", 10, 100, 50)
d_step = st.sidebar.number_input("Passo (m)", 1, 10, 1)
ref_distance = st.sidebar.number_input("Distanza di taratura (m)", 10, 100, 40)

# Freccia
st.header("Freccia")
mass = st.number_input("Peso (g)", 10.0, 50.0, 24.0)
length = st.number_input("Lunghezza (m)", 0.5, 1.0, 0.75)
diameter = st.number_input("Diametro (mm)", 4.0, 8.0, 6.2)
spine = st.number_input("Spine", 200, 1200, 700)
balance_point = st.number_input("Punto di bilanciamento (m)", 0.2, 0.8, 0.4)
tip_type = st.selectbox("Tipo di punta", list(TIPO_PUNTA_CD_FACTOR.keys()))

# Arco
st.header("Arco")
draw_force = st.number_input("Forza (lb)", 10.0, 80.0, 36.0)
draw_length = st.number_input("Allungo (m)", 0.5, 1.1, 0.70)
brace_height = st.number_input("Brace (m)", 0.05, 0.30, 0.18)
efficiency = st.number_input("Efficienza", 0.5, 0.95, 0.82)
bow_type = st.selectbox("Tipo di arco", list(BOW_TYPE_DEFAULT_EFF.keys()))

# Arciere
st.header("Arciere")
launch_height_neutral = st.number_input("Quota uscita freccia neutra (m)", 0.8, 2.2, 1.5)
anchor_length = st.number_input("Lunghezza spalla-punto aggancio (m)", 0.4, 1.0, 0.75)
pelvis_height = st.number_input("Quota bacino (m)", 0.3, 1.5, 1.0)
eye_offset_v = st.number_input("Offset verticale occhio (m)", 0.01, 0.25, 0.09)
posture = st.selectbox("Postura", ["in piedi", "inginocchiato"])

# Bersaglio
st.header("Bersaglio")
target_distance = st.number_input("Distanza bersaglio (m)", 1.0, 150.0, 40.0)
target_height = st.number_input("Quota bersaglio (m)", -2.0, 3.0, 1.5)

# Opzioni
st.header("Opzioni")
use_measured_v0 = st.checkbox("Usa velocità misurata")
v0 = st.number_input("v₀ misurata (m/s)", 5.0, 120.0, 55.0, disabled=not use_measured_v0)

if st.button("Calcola"):
    params = {
        'mass': mass,
        'length': length,
        'spine': spine,
        'diameter': diameter,
        'balance_point': balance_point,
        'tip_type': tip_type,
        'draw_force': draw_force,
        'draw_length': draw_length,
        'brace_height': brace_height,
        'efficiency': efficiency,
        'bow_type': bow_type,
        'launch_height_neutral': launch_height_neutral,
        'anchor_length': anchor_length,
        'pelvis_height': pelvis_height,
        'eye_offset_v': eye_offset_v,
        'posture': posture,
        'target_distance': target_distance,
        'target_height': target_height,
        'use_measured_v0': use_measured_v0,
        'v0': v0 if use_measured_v0 else 0.0,
    }

    angolo_finale, quota_finale = trova_angolo_convergenza(params)
    params['launch_height'] = quota_finale

    # Calcolo angolo di mira e scarto
    dx = target_distance
    dy = target_height - quota_finale
    angolo_mira = np.degrees(np.arctan2(dy, dx))
    rel_angle = angolo_finale - angolo_mira

    # Traiettoria
    X1, Y1, v0_calc, t1 = simulate_trajectory(angolo_finale, params, include_drag=True)

    fig1, ax1 = plt.subplots()
    ax1.plot(X1, Y1, label="Con drag (realistico)")
    ax1.plot(target_distance, target_height, 'go', label="Bersaglio")
    ax1.set_xlabel("Distanza (m)")
    ax1.set_ylabel("Altezza (m)")
    ax1.set_title("Traiettoria")
    ax1.grid(True)
    ax1.legend()
    st.pyplot(fig1)

    # --- Curva drop vs distanza ---
    distanze = np.arange(d_min, d_max+1, d_step)
    drops = []
    for d in distanze:
        if d <= X1.max():
            y_freccia = get_y_at_x(X1, Y1, d)
            y_mira = quota_finale + np.tan(np.radians(angolo_finale)) * d
            drop_cm = (y_mira - y_freccia) * 100
            drops.append(drop_cm)
        else:
            drops.append(None)

    fig2, axd = plt.subplots()
    axd.plot(distanze, drops, marker='o')
    axd.set_xlabel("Distanza (m)")
    axd.set_ylabel("Drop (cm)")
    axd.set_title(f"Curva drop vs distanza (taratura a {ref_distance} m)")
    axd.grid(True)
    st.pyplot(fig2)

    # Risultati
    st.success(
        f"**Angolo ottimale (orizzontale):** {angolo_finale:.2f}°\n"
        f"**Angolo di mira (punta→bersaglio):** {angolo_mira:.2f}°\n"
        f"**Scarto rispetto mira:** {rel_angle:+.2f}°\n"
        f"**Quota uscita:** {quota_finale:.2f} m\n"
        f"**v₀:** {v0_calc:.2f} m/s\n"
        f"**Tempo volo:** {t1:.2f} s"
    )

