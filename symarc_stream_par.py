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
    try:
        foc = calcola_foc(params)
        if foc > 10:
            Cd *= (1 + 0.02 * (foc - 10))
    except:
        pass
    try:
        spine = params['spine']
        Cd *= (1 + 0.001 * (spine - 500))
    except:
        pass
    return Cd

def calcola_foc(params):
    lunghezza = params['length']
    punto_equilibrio = params['balance_point']
    centro_geometrico = lunghezza / 2
    foc = ((punto_equilibrio - centro_geometrico) / lunghezza) * 100
    return foc

def calculate_velocity(params):
    if params['use_measured_v0']:
        return params['v0']
    mass = params['mass'] / 1000.0
    F = params['draw_force'] * 4.44822
    draw_length = params['draw_length']
    efficiency = params['efficiency']
    E = efficiency * F * draw_length
    return np.sqrt(2 * E / mass)

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
    X = np.array(X)
    Y = np.array(Y)
    return X, Y, v0, t

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

def plot_trajectory(X1, Y1, X2=None, Y2=None, params=None, angle=None, v0=None, tflight=None, show_mira=False):
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
    theta_rad = np.radians(angle)
    if show_mira:
        x_mira = np.array([0, params['target_distance']])
        y0 = params['launch_height']
        theta_rad = np.radians(angle)
        y_mira = y0 + np.tan(theta_rad) * x_mira
        ax.plot(x_mira, y_mira, 'k--', label="Linea di mira (tangente all'uscita)")

    # Drop rispetto alla linea di mira al bersaglio
    y_freccia = get_y_at_x(X1, Y1, params['target_distance'])
    y_mira_finale = y0 + np.tan(theta_rad) * params['target_distance']
    drop_cm = (y_mira_finale - y_freccia) * 100
    ax.annotate(
        f"Drop: {drop_cm:.1f} cm",
        xy=(params['target_distance'], y_freccia),
        xytext=(params['target_distance'] - 5, y_freccia - 0.4),
        arrowprops=dict(arrowstyle="->"),
        bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="gray", alpha=0.8)
    )
    title = (
        f"Traiettoria della freccia\n"
        f"Angolo ottimale rispetto orizzontale: {angle:.2f}°, rispetto alla mira: {rel_angle:.2f}°\n"
        f"v₀: {v0:.1f} m/s, Tempo volo: {tflight:.2f} s"
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Distanza orizzontale (m)")
    ax.set_ylabel("Altezza (m)")
    ax.set_xlim(0, params['target_distance'] + 1)
    y_max_traject = max(Y1.max(), (Y2.max() if Y2 is not None else 0), params['target_height'])
    if show_mira:
        theta_rad = np.radians(angle)
        y_mira_end = params['launch_height'] + np.tan(theta_rad) * params['target_distance']
        y_max_traject = max(y_max_traject, y_mira_end)
    ax.set_ylim(min(Y1.min(), params['target_height'], 0), math.ceil(y_max_traject) + 0.5)
    ax.grid(True)
    ax.legend()
    return fig

def calcola_quota_uscita_posturale(params, angle_rad):
    anchor_length = params['anchor_length']
    quota_uscita_neutra = params['launch_height_neutral']
    pelvis_height = params['pelvis_height']
    x_pivot = -anchor_length
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
    for iterazione in range(max_iter):
        params['launch_height'] = quota_uscita
        angolo_nuovo = find_optimal_angle(params)
        y_launch_new = calcola_quota_uscita_posturale(params, np.radians(angolo_nuovo))
        if abs(angolo_nuovo - angolo) < tol_angle and abs(y_launch_new - quota_uscita) < tol_height:
            return angolo_nuovo, y_launch_new
        angolo = angolo_nuovo
        quota_uscita = y_launch_new
    return angolo, quota_uscita

def plot_parallax_scenario(
    target_distance=40.0, target_height=1.5,
    riser_height=0.43, riser_width=0.03, window_height=0.12, window_offset=0.05,
    anchor_length=0.75, launch_height=1.5, eye_offset_v=0.09
):
    x_eye = -anchor_length
    y_eye = launch_height + eye_offset_v
    x_riser = 0
    y_riser_base = launch_height - riser_height/2
    y_riser_top = launch_height + riser_height/2
    x_target = target_distance
    y_target = target_height

    # Finestra arco
    y_window_bottom = launch_height - window_height/2
    y_window_top = launch_height + window_height/2

    # Punto di mira apparente
    m = (y_target - y_eye) / (x_target - x_eye)
    y_mira = y_eye + m * (x_riser - x_eye)
    delta_mm = (y_mira - launch_height) * 1000

    fig, ax = plt.subplots(figsize=(10, 3))

    # Bersaglio olimpico
    circle = plt.Circle((x_target, y_target), 0.3, fill=False, color="blue", lw=3)
    ax.add_patch(circle)
    ax.plot(x_target, y_target, 'yo', ms=15, label="Centro bersaglio")

    # Riser
    ax.add_patch(plt.Rectangle(
        (x_riser - riser_width/2, y_riser_base),
        riser_width, riser_height, color="gray", alpha=0.5, label="Riser"
    ))
    ax.add_patch(plt.Rectangle(
        (x_riser, y_window_bottom),
        window_offset, window_height, color="white", alpha=1, zorder=5
    ))

    # Freccia
    ax.plot(x_riser, launch_height, 'go', ms=10, label="Freccia (uscita)")

    # Occhio
    ax.plot(x_eye, y_eye, 'o', color='orange', ms=10, label="Occhio")

    # Linea di mira
    ax.plot([x_eye, x_target], [y_eye, y_target], 'k--', lw=2, label="Linea di mira")

    # Punto di mira apparente
    ax.plot(x_riser, y_mira, 'ro', ms=10, label="Punto di mira (parallasse)")
    ax.annotate(
        f"{delta_mm:+.1f} mm rispetto freccia",
        xy=(x_riser, y_mira), xytext=(x_riser+1, y_mira+0.2),
        arrowprops=dict(arrowstyle="->", color='red'),
        fontsize=10, color='red'
    )

    ax.set_xlim(x_eye - 0.2, x_target + 1.5)
    ax.set_ylim(min(y_riser_base, y_eye, y_target) - 0.5, max(y_riser_top, y_eye, y_target) + 0.5)
    ax.set_aspect('auto')
    ax.set_xlabel("Distanza orizzontale (m)")
    ax.set_ylabel("Quota (m)")
    ax.set_title("Effetto parallasse: punto di mira apparente sul riser")
    ax.legend()
    ax.grid(True, linestyle=':')
    plt.tight_layout()
    return fig
    
def plot_parallax_fpv(
    target_distance=40.0, target_height=1.5,
    riser_height=0.43, riser_width=0.03, window_height=0.12, window_offset=0.05,
    anchor_length=0.75, launch_height=1.5, eye_offset_v=0.09
):
    # Sistema di riferimento: occhio in (0,0)
    D = target_distance
    d = anchor_length
    y_eye = launch_height + eye_offset_v
    y_target_rel = target_height - y_eye

    # Riser davanti all'occhio (a destra)
    x_riser = d
    y_riser_base = -eye_offset_v - riser_height/2
    y_riser_top = -eye_offset_v + riser_height/2

    # Finestra (sul lato destro del riser, per arciere destrimano)
    x_window_left = x_riser + riser_width/2 - window_offset
    x_window_right = x_riser + riser_width/2
    y_window_bottom = -eye_offset_v - window_height/2
    y_window_top = -eye_offset_v + window_height/2

    # Freccia: cocca nell'angolo in basso a destra della finestra
    x_arrow = x_window_right - 0.004  # piccolo margine per vedere il verde
    y_arrow = y_window_bottom + 0.005

    # Punto di mira apparente (intersezione retta visuale col riser)
    m = y_target_rel / (D + d)
    y_mira = m * d
    delta_mm = (y_mira - (-eye_offset_v)) * 1000

    # Bersaglio - dimensione apparente e posizione
    bersaglio_diametro_reale = 1.22  # metri
    x_bersaglio = x_riser + 0.2      # sempre centrato dietro il riser
    y_bersaglio = y_mira

    # Dimensione apparente (diametro angolare -> diametro a questa distanza nel disegno)
    diametro_apparente_rad = bersaglio_diametro_reale / (D)
    # Scala: proiettiamo la "dimensione" a una distanza fissa dietro il riser (es. 0.2 m)
    scale = 0.2 / D
    bersaglio_draw_diam = bersaglio_diametro_reale * scale

    fig, ax = plt.subplots(figsize=(4, 6))

    # Riser (rettangolo verticale)
    ax.add_patch(plt.Rectangle(
        (x_riser - riser_width/2, y_riser_base),
        riser_width, riser_height, color="gray", alpha=0.7, label="Riser"
    ))

    # Finestra (sul lato destro)
    ax.add_patch(plt.Rectangle(
        (x_window_left, y_window_bottom),
        window_offset, window_height, color="white", alpha=1, zorder=5
    ))

    # Freccia: cocca verde, angolo in basso a destra della finestra
    ax.plot(x_arrow, y_arrow, 'go', ms=13, label="Freccia (cocca)")

    # Punto di mira apparente (rosso, dove il bersaglio cade sul riser)
    ax.plot(x_riser, y_mira, 'ro', ms=13, label="Punto di mira apparente")

    # Bersaglio olimpico (10 cerchi concentrici, colori regolamentari semplificati)
    bersaglio_color = ['#FFF', '#000', '#00F', '#F00', '#FFD700']  # bianco, nero, blu, rosso, oro
    rings_cm = [122, 100, 80, 60, 40, 20, 12.2]  # diametri anelli principali (cm)
    rings = [r/100 * scale for r in rings_cm]    # scala e conversione metri

    for i, (radius, col) in enumerate(zip(reversed(rings), reversed(bersaglio_color*2))):
        ring = plt.Circle((x_bersaglio, y_bersaglio), radius/2, fill=True, color=col, alpha=0.33, zorder=1)
        ax.add_patch(ring)

    # Legenda in alto a destra
    ax.legend(loc='upper right', fontsize=9)

    # Annotazione delta
    ax.annotate(
        f"{delta_mm:+.1f} mm rispetto freccia",
        xy=(x_riser, y_mira), xytext=(x_riser + 0.07, y_mira + 0.09),
        arrowprops=dict(arrowstyle="->", color='red'),
        fontsize=10, color='red'
    )

    ax.set_xlim(x_riser - 0.04, x_riser + 0.28)
    ax.set_ylim(y_riser_base - 0.08, y_riser_top + 0.18)
    ax.set_xlabel("Vista soggettiva (orizzontale)")
    ax.set_ylabel("Quota relativa all'occhio (m)")
    ax.set_title("Vista soggettiva: punto di mira apparente su riser", fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    return fig


# --- INTERFACCIA STREAMLIT ---

st.title("Simulatore avanzato tiro con l'arco")

# FRECCIA
st.header("Freccia")
col1, col2 = st.columns(2)
with col1:
    mass = st.number_input("Peso (g)", 10.0, 50.0, 24.0)
    length = st.number_input("Lunghezza (m)", 0.5, 1.0, 0.75)
    diameter = st.number_input("Diametro (mm)", 4.0, 8.0, 6.2)
    spine = st.number_input("Spine", 200, 1200, 700)
with col2:
    balance_point = st.number_input("Punto di bilanciamento (m)", 0.2, 0.8, 0.4)
    tip_type = st.selectbox("Tipo di punta", ["Standard", "Slanciata (field/bullet)", "Broadhead (larga)"])

# ARCO
st.header("Arco")
col3, col4 = st.columns(2)
with col3:
    draw_force = st.number_input("Forza (lb)", 10.0, 80.0, 36.0)
    draw_length = st.number_input("Allungo (m)", 0.5, 1.1, 0.70)
with col4:
    efficiency = st.number_input("Efficienza", 0.7, 0.9, 0.82)
    bow_type = st.selectbox("Tipo di arco", ["longbow", "ricurvo", "compound", "takedown"])

# ARCIERE
st.header("Arciere (biometria e postura)")
col5, col6 = st.columns(2)
with col5:
    launch_height_neutral = st.number_input("Quota uscita freccia neutra (m)", 0.8, 2.2, 1.5)
    anchor_length = st.number_input("Lunghezza spalla-punto aggancio (m)", 0.4, 1.0, 0.75)
with col6:
    pelvis_height = st.number_input("Quota bacino (m)", 0.3, 1.5, 1.0)
    eye_offset_v = st.number_input("Offset verticale occhio (m)", 0.01, 0.25, 0.09)

posture = st.selectbox("Postura", ["in piedi", "inginocchiato"])

# BERSAGLIO
st.header("Bersaglio")
col7, col8 = st.columns(2)
with col7:
    target_distance = st.number_input("Distanza bersaglio (m)", 1.0, 150.0, 40.0)
with col8:
    target_height = st.number_input("Quota bersaglio (m)", -2.0, 3.0, 1.5)

# OPZIONI
st.header("Opzioni")
use_measured_v0 = st.checkbox("Usa velocità misurata")
v0 = st.number_input("v₀ misurata (m/s)", 5.0, 120.0, 55.0, disabled=not use_measured_v0)
show_mira = st.checkbox("Mostra linea di mira", value=True)
show_compare = st.checkbox("Confronta con traiettoria ideale (senza drag)", value=False)

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
        'efficiency': efficiency if not use_measured_v0 else 0.82,  # sarà ricalcolata sotto
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
        'show_mira': show_mira,
        'show_compare': show_compare,
    }

    bow_eff = {
        "longbow": 0.75,
        "ricurvo": 0.82,
        "compound": 0.87,
        "takedown": 0.80
    }
    if not use_measured_v0:
        params['efficiency'] = bow_eff.get(bow_type, 0.8)
    else:
        mass_kg = mass / 1000.0
        F = draw_force * 4.44822
        E_cinetica = 0.5 * mass_kg * v0 ** 2
        E_elastica = F * draw_length
        eff_calc = E_cinetica / E_elastica if E_elastica > 0 else 0
        params['efficiency'] = eff_calc

    angolo_finale, quota_finale = trova_angolo_convergenza(params)
    params['launch_height'] = quota_finale
    X1, Y1, v0_calc, t1 = simulate_trajectory(angolo_finale, params, include_drag=True)
    if show_compare:
        X2, Y2, _, _ = simulate_trajectory(angolo_finale, params, include_drag=False)
    else:
        X2, Y2 = None, None

    fig = plot_trajectory(X1, Y1, X2, Y2, params=params, angle=angolo_finale, v0=v0_calc, tflight=t1, show_mira=show_mira)
    st.pyplot(fig)

    st.success(
        f"**Angolo ottimale:** {angolo_finale:.2f}°\n"
        f"**Quota uscita:** {quota_finale:.2f} m\n"
        f"**v₀ calcolata:** {v0_calc:.2f} m/s\n"
        f"**Tempo volo:** {t1:.2f} s\n"
        f"**Efficienza stimata:** {params['efficiency']:.3f}"
    )

    # Grafico parallasse: riser/occhio/bersaglio
    fig_fpv = plot_parallax_fpv(
        target_distance=target_distance,
        target_height=target_height,
        riser_height=0.43,
        riser_width=0.03,
        window_height=0.12,
        window_offset=0.05,
        anchor_length=anchor_length,
        launch_height=quota_finale,
        eye_offset_v=eye_offset_v
    )
    st.pyplot(fig_fpv)
