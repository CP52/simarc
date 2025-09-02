# simarc_mira_streamlit.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d, UnivariateSpline
import io
import pandas as pd
import math

# ------------------ COSTANTI ------------------
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

# ------------------ FISICA / MODELLI ------------------
def reynolds_number(v, diameter_mm):
    d_m = diameter_mm / 1000.0
    return air_density * v * d_m / mu_air

def realistic_drag_coefficient(v, diameter_mm, angle_of_attack_deg, tip_type, params):
    Re = reynolds_number(v, diameter_mm)
    if Re < 1.2e4:
        Cd = 1.5
    elif Re < 2.0e4:
        Cd = 1.5 + (Re - 1.2e4) / (8e3) * (2.6 - 1.5)
    else:
        Cd = 2.6
    gamma_rad = np.radians(angle_of_attack_deg)
    Cd *= (1 + 4 * gamma_rad**2)
    Cd *= TIPO_PUNTA_CD_FACTOR.get(tip_type, 1.0)
    # (light) correttivi empirici opzionali:
    try:
        spine = params['spine']
        Cd *= (1 + 0.001 * (spine - 500))
    except Exception:
        pass
    return Cd

def calculate_velocity(params):
    # v0 misurata -> uso diretto
    if params['use_measured_v0']:
        return params['v0']
    # v0 calcolata da energia elastica utile: eff * F * (draw_length - brace)
    mass = params['mass'] / 1000.0
    F = params['draw_force'] * 4.44822
    elong = max(0.0, params['draw_length'] - params['brace_height'])
    E = params['efficiency'] * F * elong
    return np.sqrt(max(0.0, 2 * E / mass))

def simulate_trajectory(angle_deg, params, include_drag=True):
    angle = np.radians(angle_deg)
    mass = params['mass'] / 1000.0
    v0 = calculate_velocity(params)
    A = np.pi * (params['diameter'] / 1000.0 / 2.0) ** 2
    dt = 0.001

    x, y = 0.0, params['launch_height']
    vx, vy = v0 * np.cos(angle), v0 * np.sin(angle)
    X, Y = [x], [y]
    t = 0.0

    while x <= params['target_distance']:
        v = np.sqrt(vx**2 + vy**2)
        gamma = np.degrees(np.arctan2(vy, vx)) - angle_deg
        k = 0.0
        if include_drag:
            Cd = realistic_drag_coefficient(v, params['diameter'], gamma, params['tip_type'], params)
            k = 0.5 * air_density * Cd * A / mass
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
    f = interp1d(X, Y, kind='linear', fill_value='extrapolate')
    return float(f(x_target))

def find_optimal_angle(params):
    target_y = params['target_height']
    distance = params['target_distance']
    def objective(angle):
        X, Y, _, _ = simulate_trajectory(angle, params, include_drag=True)
        if len(X) < 2:
            return 1e9
        y_at_target = get_y_at_x(X, Y, distance)
        return abs(y_at_target - target_y)
    res = minimize_scalar(objective, bounds=(-20, 45), method='bounded')
    return res.x

def calcola_quota_uscita_posturale(params, angle_rad):
    # Rotazione del vettore pivot->punta (pivot a quota bacino)
    anchor_length = params['anchor_length']
    y_neutral = params['launch_height_neutral']
    pelvis = params['pelvis_height']
    y_pivot = y_neutral - pelvis
    dx, dy = anchor_length, pelvis
    r = np.sqrt(dx**2 + dy**2)
    phi0 = np.arcsin(dy / r)
    y_launch = y_pivot + r * np.sin(angle_rad + phi0)
    return y_launch

def trova_angolo_convergenza(params, tol_angle=0.01, tol_height=0.001, max_iter=25):
    quota_uscita = params['launch_height_neutral']
    angolo = 0.0
    for _ in range(max_iter):
        params['launch_height'] = quota_uscita
        angolo_nuovo = find_optimal_angle(params)
        y_launch_new = calcola_quota_uscita_posturale(params, np.radians(angolo_nuovo))
        if abs(angolo_nuovo - angolo) < tol_angle and abs(y_launch_new - quota_uscita) < tol_height:
            return angolo_nuovo, y_launch_new
        angolo, quota_uscita = angolo_nuovo, y_launch_new
    return angolo, quota_uscita

# ------------------ PLOT TRAIETTORIA ------------------
def plot_trajectory(X1, Y1, X2=None, Y2=None, params=None, angle=None, v0=None, tflight=None, show_mira=True):
    fig, ax = plt.subplots()
    ax.plot(X1, Y1, label="Con drag (realistico)")
    if X2 is not None and Y2 is not None:
        ax.plot(X2, Y2, label="Senza drag", linestyle='--')
        y1_at_target = get_y_at_x(X1, Y1, params['target_distance'])
        y2_at_target = get_y_at_x(X2, Y2, params['target_distance'])
        delta_cm = abs(y1_at_target - y2_at_target) * 100
        ax.annotate(
            f"Δ: {delta_cm:.1f} cm",
            xy=(params['target_distance'], y1_at_target),
            xytext=(params['target_distance'] - 8, max(y1_at_target, y2_at_target) + 0.5),
            arrowprops=dict(arrowstyle="->"),
            bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="gray", alpha=0.8)
        )

    ax.plot(params['target_distance'], params['target_height'], 'go', label="Bersaglio")

    dx = params['target_distance']
    dy = params['target_height'] - params['launch_height']
    mira_angle = np.degrees(np.arctan2(dy, dx))
    rel_angle = angle - mira_angle

    y0 = params['launch_height']
    th = np.radians(angle)
    if show_mira:
        x_mira = np.array([0, params['target_distance']])
        y_mira = y0 + np.tan(th) * x_mira
        ax.plot(x_mira, y_mira, 'k--', label="Linea di mira")

    y_freccia = get_y_at_x(X1, Y1, params['target_distance'])
    y_mira_finale = y0 + np.tan(th) * params['target_distance']
    drop_cm = (y_mira_finale - y_freccia) * 100.0
    ax.annotate(
        f"Drop: {drop_cm:.1f} cm",
        xy=(params['target_distance'], y_freccia),
        xytext=(params['target_distance'] - 5, y_freccia - 0.4),
        arrowprops=dict(arrowstyle="->"),
        bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="gray", alpha=0.8)
    )

    title = (
        f"Traiettoria\n"
        f"Angolo ottimale: {angle:.2f}° (rel. mira: {rel_angle:.2f}°)\n"
        f"v₀: {v0:.1f} m/s, Tvolo: {tflight:.2f} s"
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Distanza (m)")
    ax.set_ylabel("Altezza (m)")
    ax.set_xlim(0, params['target_distance'] + 1)
    y_max = max(Y1.max(), (Y2.max() if Y2 is not None else 0), params['target_height'])
    if show_mira:
        y_mira_end = params['launch_height'] + np.tan(th) * params['target_distance']
        y_max = max(y_max, y_mira_end)
    ax.set_ylim(min(Y1.min(), params['target_height'], 0), math.ceil(y_max) + 0.5)
    ax.grid(True)
    ax.legend()
    return fig

# ------------------ GEOMETRIA RISER (dal tuo mira.py) ------------------
# y_cm: proiezione sul riser (cm) di un punto a distanza x, con drop d (m)
def y_cm(x, o, t, d=0.0):
    # o = distanza occhio–cocca [m], t = distanza cocca–riser [m]
    u = (o + d) / (t + x)
    y_calc = 100.0 * x * (u / np.sqrt(1 + u**2))
    return y_calc - d * 100.0  # tolgo il drop espresso in cm

# ============================================================
#                      INTERFACCIA
# ============================================================
st.set_page_config(page_title="Simulatore balistico + Mirino", layout="wide")
st.title("🏹 Simulatore + Generatore di Mirino (riser)")

with st.sidebar:
    st.header("Curva Drop")
    d_min = st.number_input("Distanza minima (m)", 5, 100, 10)
    d_max = st.number_input("Distanza massima (m)", 10, 100, 60)
    d_step = st.number_input("Passo (m)", 1, 10, 10)
    fit_degree = st.slider("Grado polinomio per drop(x)", 1, 5, 2)
    d_query = st.number_input("Distanza per query drop(x) (m)", 5, 100, 30)

# ------------------ INPUT PRINCIPALI ------------------
colA, colB = st.columns(2)

with colA:
    st.subheader("Freccia")
    mass = st.number_input("Peso (g)", 10.0, 50.0, 24.0)
    length = st.number_input("Lunghezza (m)", 0.5, 1.0, 0.75)
    diameter = st.number_input("Diametro (mm)", 4.0, 8.0, 6.2)
    spine = st.number_input("Spine", 200, 1200, 700)
    balance_point = st.number_input("Punto bilanciamento (m)", 0.2, 0.8, 0.4)
    tip_type = st.selectbox("Tipo di punta", list(TIPO_PUNTA_CD_FACTOR.keys()))

    st.subheader("Arco")
    draw_force = st.number_input("Forza (lb)", 10.0, 80.0, 36.0)
    draw_length = st.number_input("Allungo (m)", 0.5, 1.1, 0.70)
    brace_height = st.number_input("Brace (m)", 0.05, 0.30, 0.18)
    efficiency = st.number_input("Efficienza", 0.5, 0.95, 0.82)
    bow_type = st.selectbox("Tipo di arco", list(BOW_TYPE_DEFAULT_EFF.keys()))

with colB:
    st.subheader("Arciere (biometria e postura)")
    launch_height_neutral = st.number_input("Quota uscita freccia neutra (m)", 0.8, 2.2, 1.5)
    anchor_length = st.number_input("Lunghezza spalla-punto aggancio (m)", 0.4, 1.0, 0.75)
    pelvis_height = st.number_input("Quota bacino (m)", 0.3, 1.5, 1.0)
    eye_offset_v = st.number_input("Offset verticale occhio (m)", 0.01, 0.25, 0.09)
    posture = st.selectbox("Postura", ["in piedi", "inginocchiato"])

    st.subheader("Bersaglio")
    target_distance = st.number_input("Distanza bersaglio (m)", 1.0, 150.0, 40.0)
    target_height = st.number_input("Quota bersaglio (m)", -2.0, 3.0, 1.5)

    st.subheader("Opzioni balistiche")
    use_measured_v0 = st.checkbox("Usa v₀ misurata")
    v0 = st.number_input("v₀ misurata (m/s)", 5.0, 120.0, 55.0, disabled=not use_measured_v0)
    show_mira = st.checkbox("Mostra linea di mira", value=True)
    show_compare = st.checkbox("Confronta con traiettoria ideale", value=False)

st.subheader("Geometria visiva mirino (riser)")
colG1, colG2 = st.columns(2)
with colG1:
    o_eye_cock = st.number_input("Distanza occhio–cocca o–c (m)", 0.05, 0.40, 0.11, step=0.01)
with colG2:
    t_cock_riser = st.number_input("Distanza cocca–riser c–r (m)", 0.2, 1.5, 0.70, step=0.01)

# ------------------ CALCOLO ------------------
if st.button("Calcola e genera mirino"):
    params = {
        'mass': mass, 'length': length, 'spine': spine, 'diameter': diameter,
        'balance_point': balance_point, 'tip_type': tip_type,
        'draw_force': draw_force, 'draw_length': draw_length, 'brace_height': brace_height,
        'efficiency': efficiency, 'bow_type': bow_type,
        'launch_height_neutral': launch_height_neutral, 'anchor_length': anchor_length,
        'pelvis_height': pelvis_height, 'eye_offset_v': eye_offset_v,
        'posture': posture, 'target_distance': target_distance, 'target_height': target_height,
        'use_measured_v0': use_measured_v0, 'v0': v0 if use_measured_v0 else 0.0
    }

    # Se non v0 misurata: preset efficienza per tipo arco SOLO se l'utente non ha messo un valore "suo"
    if not use_measured_v0 and abs(efficiency - 0.82) < 1e-6:
        params['efficiency'] = BOW_TYPE_DEFAULT_EFF.get(bow_type, efficiency)

    # --- Iterazione postura -> angolo ---
    ang_opt, y_launch = trova_angolo_convergenza(params)
    params['launch_height'] = y_launch

    # Angolo di mira e scarto
    dx = target_distance
    dy = target_height - y_launch
    ang_mira = np.degrees(np.arctan2(dy, dx))
    rel_angle = ang_opt - ang_mira

    # Traiettoria (per mostrare grafico completo)
    X1, Y1, v0_calc, t1 = simulate_trajectory(ang_opt, params, include_drag=True)
    if show_compare:
        X2, Y2, _, _ = simulate_trajectory(ang_opt, params, include_drag=False)
    else:
        X2, Y2 = None, None

    # ---- Grafico 1: traiettoria
    st.markdown("### Traiettoria della freccia")
    fig_traj = plot_trajectory(X1, Y1, X2, Y2, params=params, angle=ang_opt, v0=v0_calc, tflight=t1, show_mira=show_mira)
    st.pyplot(fig_traj, use_container_width=True)

    # ---- Curva drop vs distanza (usando SEMPRE l'angolo ottimale trovato)
    st.markdown("### Curva del drop vs distanza")
    distanze = np.arange(d_min, d_max + 1, d_step)
    drops_cm = []
    drops_m = []
    for d in distanze:
        if d <= X1.max():
            y_freccia = get_y_at_x(X1, Y1, d)
            y_mira = y_launch + np.tan(np.radians(ang_opt)) * d
            drop_m = (y_mira - y_freccia)
            drops_m.append(drop_m)
            drops_cm.append(drop_m * 100.0)
        else:
            drops_m.append(None)
            drops_cm.append(None)

    # fit spline (se sufficiente numero di punti)
    valid_x = np.array([d for d, dr in zip(distanze, drops_cm) if dr is not None])
    valid_y = np.array([dr for dr in drops_cm if dr is not None])

    spline = None
    if len(valid_x) > 3:
        spline = UnivariateSpline(valid_x, valid_y, s=0)

    # fit polinomiale
    poly = None
    poly_coeffs = None
    if len(valid_x) >= (fit_degree + 1):
        poly_coeffs = np.polyfit(valid_x, valid_y, deg=fit_degree)
        poly = np.poly1d(poly_coeffs)

    # Plot drop + fit
    fig_drop, axd = plt.subplots()
    axd.plot(distanze, drops_cm, "o", label="Simulazione")
    if spline is not None:
        x_fit = np.linspace(valid_x.min(), valid_x.max(), 300)
        y_fit = spline(x_fit)
        axd.plot(x_fit, y_fit, "-", label="Fit spline")
    if poly is not None:
        x_fit2 = np.linspace(valid_x.min(), valid_x.max(), 300)
        y_fit2 = poly(x_fit2)
        axd.plot(x_fit2, y_fit2, "--", label=f"Polinomio grado {fit_degree}")
    axd.set_xlabel("Distanza (m)")
    axd.set_ylabel("Drop (cm)")
    axd.grid(True)
    axd.legend()
    st.pyplot(fig_drop, use_container_width=True)

    # Query drop(x) alla distanza d_query
    if spline is not None:
        dq = float(np.clip(d_query, valid_x.min(), valid_x.max()))
        drop_q_spline = float(spline(dq))
        st.info(f"**drop_spline({dq:.1f} m) ≈ {drop_q_spline:.1f} cm**")
    if poly is not None:
        drop_q_poly = float(poly(d_query))
        st.info(f"**drop_poly (grado {fit_degree})({d_query:.1f} m) ≈ {drop_q_poly:.1f} cm**")

    # Stampa formula polinomiale leggibile
    if poly is not None:
        terms = []
        deg = len(poly_coeffs) - 1
        for i, c in enumerate(poly_coeffs):
            p = deg - i
            if p == 0:
                terms.append(f"{c:+.6f}")
            elif p == 1:
                terms.append(f"{c:+.6f}·x")
            else:
                terms.append(f"{c:+.6f}·x^{p}")
        st.code("drop(x) [cm] ≈ " + " ".join(terms), language="text")

    # ---- Proiezione sul riser: genera il "mirino"
    st.markdown("### Proiezione sul riser (mirino)")

    # preferisco usare la spline se disponibile, altrimenti il polinomio, altrimenti i punti raw (interp lineare)
    def drop_cm_at(x):
        if spline is not None and valid_x.min() <= x <= valid_x.max():
            return float(spline(x))
        elif poly is not None:
            return float(poly(x))
        else:
            # fallback: interp lineare sui punti validi
            if len(valid_x) >= 2:
                f = interp1d(valid_x, valid_y, kind='linear', fill_value='extrapolate')
                return float(f(x))
            return np.nan

    proj_rows = []
    for d in distanze:
        if d < (valid_x.min() if len(valid_x) else d_min) or d > (valid_x.max() if len(valid_x) else d_max):
            proj_rows.append((d, np.nan, np.nan))
            continue
        drop_cm_val = drop_cm_at(d)  # cm
        drop_m_val = drop_cm_val / 100.0
        yproj = y_cm(d, o_eye_cock, t_cock_riser, d=drop_m_val)  # cm
        proj_rows.append((d, drop_cm_val, yproj))

    df_proj = pd.DataFrame(proj_rows, columns=["Distanza (m)", "Drop (cm)", "Proiezione riser (cm)"])
    st.dataframe(df_proj, use_container_width=True)

    # Scarica CSV
    csv_buf = io.StringIO()
    df_proj.to_csv(csv_buf, index=False)
    st.download_button("⬇️ Scarica mirino (CSV)", data=csv_buf.getvalue(), file_name="mirino_riser.csv", mime="text/csv")

    # ---- Grafico "mirino" come scala verticale con tacche
    st.markdown("### Scala del mirino (tacche sul riser)")
    # Costruisco una scala verticale: y = proiezione in cm; per estetica, sposto a x=0
    valid_proj = df_proj.dropna()
    if len(valid_proj) >= 2:
        y_marks = valid_proj["Proiezione riser (cm)"].values
        dist_marks = valid_proj["Distanza (m)"].values

        # calcolo range con margine
        y_min = float(np.nanmin(y_marks))
        y_max = float(np.nanmax(y_marks))
        pad = max(2.0, 0.05 * (y_max - y_min))
        y_min_plot = y_min - pad
        y_max_plot = y_max + pad

        fig_m, axm = plt.subplots(figsize=(3, 8))
        axm.vlines(0, y_min_plot, y_max_plot, colors="black", linewidth=2)  # "riser"
        axm.hlines(y_marks, xmin=-0.5, xmax=0.5, colors="tab:blue")          # tacche
        for yv, dv in zip(y_marks, dist_marks):
            axm.text(0.55, yv, f"{int(dv)} m", va="center", fontsize=9)

        axm.set_ylim(y_min_plot, y_max_plot)
        axm.set_xlim(-1.0, 3.0)
        axm.set_yticks(np.linspace(round(y_min_plot), round(y_max_plot), 9))
        axm.set_xticks([])
        axm.set_title("Mirino – proiezione tacche sul riser (cm)")
        axm.set_ylabel("Quota (cm) sulla colonna del riser")
        axm.grid(True, axis='y', linestyle='--', alpha=0.3)
        st.pyplot(fig_m, use_container_width=False)
    else:
        st.warning("Non ci sono abbastanza punti per disegnare la scala del mirino.")

    # ---- Riepilogo risultati principali
    st.success(
        f"**Angolo ottimale (orizzontale):** {ang_opt:.2f}°\n"
        f"**Angolo di mira (punta→bersaglio):** {ang_mira:.2f}°\n"
        f"**Scarto rispetto mira:** {rel_angle:+.2f}°\n"
        f"**Quota uscita:** {y_launch:.2f} m\n"
        f"**v₀:** {v0_calc:.2f} m/s\n"
        f"**Tempo volo:** {t1:.2f} s"
    )

