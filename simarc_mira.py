# simarc_mira.py
# App Streamlit completa: simulazione balistica + generatore mirino (riser)
# Include PDF A4 1:1 con y=0, Laser 30 m, barre 5 cm.

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d, UnivariateSpline
import io
import pandas as pd
import math
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ==============================
# COSTANTI
# ==============================
G = 9.81
air_density = 1.204     # kg/m^3
mu_air = 1.81e-5        # Pa*s

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

# ==============================
# FUNZIONI BALISTICHE
# ==============================
def reynolds_number(v, diameter_mm):
    d_m = diameter_mm / 1000.0
    return air_density * v * d_m / mu_air

def realistic_drag_coefficient(v, diameter_mm, angle_of_attack_deg, tip_type, params):
    # Modello semplice ma regolare per Cd, con dipendenza da Re e (leggera) dall'assetto
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
    return Cd

def calculate_velocity(params):
    """
    Se v0 misurata è fornita la usa; altrimenti calcola da energia elastica:
    E = eff * F * (draw_length - brace_height)
    """
    if params['use_measured_v0']:
        return params['v0']
    mass = params['mass'] / 1000.0  # g -> kg
    F = params['draw_force'] * 4.44822  # lb -> N
    elong = max(0.0, params['draw_length'] - params['brace_height'])
    E = params['efficiency'] * F * elong
    return np.sqrt(max(0.0, 2 * E / mass))

def simulate_trajectory(angle_deg, params, include_drag=True, end_x=None):
    """
    Integrazione (Euler) fino a end_x (default: max distanza utile).
    """
    angle = np.radians(angle_deg)
    mass = params['mass'] / 1000.0
    v0 = calculate_velocity(params)
    A = np.pi * (params['diameter'] / 1000.0 / 2.0) ** 2
    dt = 0.001

    # distanza massima da simulare: serve coprire sia target_distance sia drop_range_max
    end_x = float(end_x if end_x is not None else params['target_distance'])

    x, y = 0.0, params['launch_height']
    vx, vy = v0 * np.cos(angle), v0 * np.sin(angle)
    X, Y = [x], [y]
    t = 0.0

    while x <= end_x:
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
        # Se la freccia va molto sotto zero e oltre end_x di parecchio, potremmo fermare.
        if x > end_x + 5:
            break
    return np.array(X), np.array(Y), v0, t

def get_y_at_x(X, Y, x_target):
    f = interp1d(X, Y, kind='linear', fill_value='extrapolate')
    return float(f(x_target))

def find_optimal_angle(params):
    """
    Trova l'angolo che porta l'impatto esattamente alla quota bersaglio
    alla distanza params['target_distance'].
    """
    target_y = params['target_height']
    distance = params['target_distance']

    def objective(angle):
        X, Y, _, _ = simulate_trajectory(angle, params, include_drag=True, end_x=distance)
        if len(X) < 2:
            return 1e9
        y_at_target = get_y_at_x(X, Y, distance)
        return abs(y_at_target - target_y)

    res = minimize_scalar(objective, bounds=(-20, 45), method='bounded')
    return res.x

def calcola_quota_uscita_posturale(params, angle_rad):
    """
    Ruota il vettore pivot(bacino)->punta freccia (x=0) mantenendo la T.
    Il pivot è a quota (launch_height_neutral - pelvis_height).
    """
    anchor_length = params['anchor_length']
    y_neutral = params['launch_height_neutral']
    pelvis = params['pelvis_height']

    y_pivot = y_neutral - pelvis
    dx, dy = anchor_length, pelvis
    r = np.sqrt(dx**2 + dy**2)
    phi0 = np.arcsin(dy / r)
    return y_pivot + r * np.sin(angle_rad + phi0)

def trova_angolo_convergenza(params, tol_angle=0.01, tol_height=0.001, max_iter=25):
    """
    Itera: angolo ottimale -> nuova quota di uscita da postura a T -> finché converge.
    """
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

# ==============================
# GEOMETRIA RISER
# ==============================
def y_cm(x, o, t, d=0.0):
    """
    Proiezione (cm) sul riser di un punto a distanza x (m) con drop d (m).
    o = distanza occhio–cocca [m], t = distanza cocca–riser [m].
    """
    u = (o + d) / (t + x)
    y_calc = 100.0 * x * (u / np.sqrt(1 + u**2))
    return y_calc - d * 100.0

# ==============================
# PLOT TRAIETTORIA
# ==============================
def plot_trajectory(X1, Y1, params, angle, v0, tflight, X2=None, Y2=None, show_mira=True):
    fig, ax = plt.subplots()
    ax.plot(X1, Y1, label="Con drag")
    if X2 is not None and Y2 is not None:
        ax.plot(X2, Y2, "--", label="Senza drag")

    # Bersaglio
    ax.plot(params['target_distance'], params['target_height'], 'go', label="Bersaglio")

    # Linea di mira (all'angolo di lancio) – utile per definire il drop
    y0 = params['launch_height']
    th = np.radians(angle)
    if show_mira:
        x_mira = np.array([0.0, params['target_distance']])
        y_mira = y0 + np.tan(th) * x_mira
        ax.plot(x_mira, y_mira, 'k--', label="Linea di mira")

    # Annotazioni
    dx = params['target_distance']
    dy = params['target_height'] - y0
    mira_angle = np.degrees(np.arctan2(dy, dx))
    rel_angle = angle - mira_angle

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
    xmax = max(params['target_distance'], X1.max(), (X2.max() if X2 is not None else 0.0))
    ax.set_xlim(0, xmax + 1)
    y_max = max(Y1.max(), (Y2.max() if Y2 is not None else -1e9), params['target_height'])
    if show_mira:
        y_mira_end = y0 + np.tan(th) * xmax
        y_max = max(y_max, y_mira_end)
    ax.set_ylim(min(Y1.min(), params['target_height'], 0), math.ceil(y_max) + 0.5)
    ax.grid(True)
    ax.legend()
    return fig

# ==============================
# PDF MIRINO (A4, 1:1, con y=0 e Laser 30 m)
# ==============================
def esporta_mirino_pdf_bytes(df_proj, o_eye_cock, t_cock_riser, filename="mirino_riser.pdf"):
    """
    PDF in scala 1:1 (1 cm = 28.346 pt) con:
      - colonna riser,
      - tacche delle distanze (df_proj),
      - tacca aggiuntiva a y=0 cm,
      - punto Laser @30 m,
      - barre di controllo 5 cm.
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    def cm2pt(x_cm):  # scala fissa 1:1
        return x_cm * 28.346

    # Tacche del mirino
    y_vals = df_proj["Proiezione riser (cm)"].dropna().values.tolist()

    # Aggiungi il riferimento y=0
    y_vals.append(0.0)
    # Aggiungi il punto laser 30 m
    y_laser_30_cm = y_cm(30.0, o_eye_cock, t_cock_riser, d=0.0)
    y_vals.append(y_laser_30_cm)

    y_min, y_max = float(min(y_vals)), float(max(y_vals))

    x_center = width / 2.0
    margin_bottom = 50  # pt
    y0_pt = margin_bottom - cm2pt(y_min)

    # Colonna verticale riser
    c.setLineWidth(2)
    c.line(x_center, y0_pt + cm2pt(y_min), x_center, y0_pt + cm2pt(y_max))

    # Tacche normali da df_proj
    c.setFont("Helvetica", 8)
    for _, row in df_proj.dropna().iterrows():
        y_pt = y0_pt + cm2pt(float(row["Proiezione riser (cm)"]))
        c.line(x_center - 20, y_pt, x_center + 20, y_pt)
        c.drawString(x_center + 30, y_pt - 3, f"{int(row['Distanza (m)'])} m")

    # Tacca a y=0 cm
    y_zero_pt = y0_pt + cm2pt(0.0)
    c.setStrokeColorRGB(1, 0, 0)
    c.line(x_center - 25, y_zero_pt, x_center + 25, y_zero_pt)
    c.setFont("Helvetica-Bold", 8)
    c.drawString(x_center + 30, y_zero_pt - 3, "0 cm (base)")
    c.setStrokeColorRGB(0, 0, 0)

    # Punto Laser @30 m
    y_laser_pt = y0_pt + cm2pt(y_laser_30_cm)
    c.setFillColorRGB(0, 1, 0)
    c.circle(x_center, y_laser_pt, 2.5, fill=1, stroke=0)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(x_center + 30, y_laser_pt - 4, "Laser 30 m")

    # Barre di controllo 5 cm
    c.setLineWidth(3)
    y_bar = y0_pt + cm2pt(y_min) - 40
    # Orizzontale
    c.line(x_center - cm2pt(2.5), y_bar, x_center + cm2pt(2.5), y_bar)
    c.setFont("Helvetica", 9)
    c.drawCentredString(x_center, y_bar - 12, "Oriz. 5 cm")
    # Verticale
    x_bar = x_center + 80
    c.line(x_bar, y_bar, x_bar, y_bar + cm2pt(5.0))
    c.drawCentredString(x_bar, y_bar + 12, "                 Vert. 5 cm")

    c.showPage()
    c.save()
    buf.seek(0)
    return buf, filename


# ==============================
# INTERFACCIA STREAMLIT
# ==============================
st.set_page_config(page_title="Simulatore + Mirino", layout="wide")
st.title("🏹 Simulatore + Generatore di Mirino (riser)")

# --- Sidebar: impostazioni drop/fitting ---
with st.sidebar:
    st.header("Curva Drop")
    d_min = st.number_input("Distanza minima (m)", 5, 100, 10)
    d_max = st.number_input("Distanza massima (m)", 10, 100, 50)
    d_step = st.number_input("Passo (m)", 1, 10, 5)
    fit_degree = st.slider("Grado polinomio drop(x)", 1, 5, 2)
    d_query = st.number_input("Query drop(x) (m)", 5, 100, 30)

# --- Input principali (gruppi) ---
colA, colB = st.columns(2)

with colA:
    st.subheader("Freccia")
    mass = st.number_input("Peso (g)", 10.0, 50.0, 24.0)
    length = st.number_input("Lunghezza (m)", 0.5, 1.0, 0.75)
    diameter = st.number_input("Diametro (mm)", 4.0, 8.0, 6.2)
    spine = st.number_input("Spine", 200, 1200, 700)
    balance_point = st.number_input("Punto di bilanciamento (m)", 0.2, 0.8, 0.4)
    tip_type = st.selectbox("Tipo di punta", list(TIPO_PUNTA_CD_FACTOR.keys()))

    st.subheader("Arco")
    draw_force = st.number_input("Forza (lb)", 10.0, 80.0, 36.0)
    draw_length = st.number_input("Allungo (m)", 0.5, 1.1, 0.70)
    brace_height = st.number_input("Brace (m)", 0.05, 0.30, 0.18)
    efficiency = st.number_input("Efficienza", 0.5, 0.95, 0.82)
    bow_type = st.selectbox("Tipo di arco", list(BOW_TYPE_DEFAULT_EFF.keys()))

with colB:
    st.subheader("Arciere (biometria/postura)")
    launch_height_neutral = st.number_input("Quota uscita freccia neutra (m)", 0.8, 2.2, 1.5)
    anchor_length = st.number_input("Lunghezza spalla–aggancio (m)", 0.4, 1.0, 0.75)
    pelvis_height = st.number_input("Quota bacino (m)", 0.3, 1.5, 1.0)
    eye_offset_v = st.number_input("Offset verticale occhio (m)", 0.01, 0.25, 0.09)

    st.subheader("Bersaglio")
    target_distance = st.number_input("Distanza bersaglio (m)", 1.0, 150.0, 40.0)
    target_height = st.number_input("Quota bersaglio (m)", -2.0, 3.0, 1.5)

    st.subheader("Opzioni")
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

# ==============================
# CALCOLO
# ==============================
if st.button("Calcola e genera mirino"):
    # Parametri
    params = {
        'mass': mass, 'length': length, 'spine': spine, 'diameter': diameter,
        'balance_point': balance_point, 'tip_type': tip_type,
        'draw_force': draw_force, 'draw_length': draw_length, 'brace_height': brace_height,
        'efficiency': efficiency, 'bow_type': bow_type,
        'launch_height_neutral': launch_height_neutral, 'anchor_length': anchor_length,
        'pelvis_height': pelvis_height, 'eye_offset_v': eye_offset_v,
        'target_distance': target_distance, 'target_height': target_height,
        'use_measured_v0': use_measured_v0, 'v0': v0
    }

    # Se non v0 misurata e l’efficienza è quella “di default” inserita, usa un preset dal tipo di arco
    if not use_measured_v0 and abs(efficiency - 0.82) < 1e-6:
        params['efficiency'] = BOW_TYPE_DEFAULT_EFF.get(bow_type, efficiency)

    # 1) Iterazione postura -> angolo ottimale
    ang_opt, y_launch = trova_angolo_convergenza(params)
    params['launch_height'] = y_launch

    # 2) Traiettoria (simuliamo fino a max(d_max, target_distance))
    sim_end = max(float(d_max), float(target_distance))
    X1, Y1, v0_calc, t1 = simulate_trajectory(ang_opt, params, include_drag=True, end_x=sim_end)
    if show_compare:
        X2, Y2, _, _ = simulate_trajectory(ang_opt, params, include_drag=False, end_x=sim_end)
    else:
        X2, Y2 = None, None

    # 3) Plot traiettoria
    st.markdown("### Traiettoria")
    fig_traj = plot_trajectory(X1, Y1, params, ang_opt, v0_calc, t1, X2, Y2, show_mira=show_mira)
    st.pyplot(fig_traj, use_container_width=True)

    # 4) Curva drop vs distanza (usando SEMPRE l'angolo ottimale trovato)
    st.markdown("### Curva del drop vs distanza")
    distanze = np.arange(d_min, d_max + 1, d_step)
    drops_cm, drops_m = [], []
    for d in distanze:
        if d <= X1.max():
            y_freccia = get_y_at_x(X1, Y1, d)
            y_mira = y_launch + np.tan(np.radians(ang_opt)) * d
            drop_m = y_mira - y_freccia
            drops_m.append(drop_m)
            drops_cm.append(drop_m * 100.0)
        else:
            drops_m.append(None); drops_cm.append(None)

    valid_x = np.array([d for d, dr in zip(distanze, drops_cm) if dr is not None])
    valid_y = np.array([dr for dr in drops_cm if dr is not None])

    spline = UnivariateSpline(valid_x, valid_y, s=0) if len(valid_x) > 3 else None
    poly, poly_coeffs = (None, None)
    if len(valid_x) >= (fit_degree + 1):
        poly_coeffs = np.polyfit(valid_x, valid_y, deg=fit_degree)
        poly = np.poly1d(poly_coeffs)

    fig_drop, axd = plt.subplots()
    axd.plot(distanze, drops_cm, "o", label="Simulazione")
    if spline is not None:
        x_fit = np.linspace(valid_x.min(), valid_x.max(), 300)
        axd.plot(x_fit, spline(x_fit), "-", label="Fit spline")
    if poly is not None:
        x_fit2 = np.linspace(valid_x.min(), valid_x.max(), 300)
        axd.plot(x_fit2, poly(x_fit2), "--", label=f"Polinomio grado {fit_degree}")
    axd.set_xlabel("Distanza (m)")
    axd.set_ylabel("Drop (cm)")
    axd.grid(True); axd.legend()
    st.pyplot(fig_drop, use_container_width=True)

    # 5) Query drop(x)
    if spline is not None:
        dq = float(np.clip(d_query, valid_x.min(), valid_x.max()))
        st.info(f"**drop_spline({dq:.1f} m) ≈ {float(spline(dq)):.1f} cm**")
    if poly is not None:
        st.info(f"**drop_poly (grado {fit_degree})({d_query:.1f} m) ≈ {float(poly(d_query)):.1f} cm**")

    if poly is not None:
        deg = len(poly_coeffs) - 1
        terms = []
        for i, c in enumerate(poly_coeffs):
            p = deg - i
            if p == 0:
                terms.append(f"{c:+.6f}")
            elif p == 1:
                terms.append(f"{c:+.6f}·x")
            else:
                terms.append(f"{c:+.6f}·x^{p}")
        st.code("drop(x) [cm] ≈ " + " ".join(terms), language="text")

    # 6) Proiezione sul riser
    st.markdown("### Proiezione sul riser (mirino)")
    def drop_cm_at(x):
        if spline is not None and valid_x.min() <= x <= valid_x.max():
            return float(spline(x))
        elif poly is not None:
            return float(poly(x))
        else:
            if len(valid_x) >= 2:
                return float(interp1d(valid_x, valid_y, kind='linear', fill_value='extrapolate')(x))
            return np.nan

    proj_rows = []
    for d in distanze:
        if len(valid_x) and (d < valid_x.min() or d > valid_x.max()):
            proj_rows.append((d, np.nan, np.nan)); continue
        drop_cm_val = drop_cm_at(d)
        drop_m_val = drop_cm_val / 100.0 if drop_cm_val is not None and not np.isnan(drop_cm_val) else np.nan
        yproj = y_cm(d, o_eye_cock, t_cock_riser, d=drop_m_val if not np.isnan(drop_m_val) else 0.0)
        proj_rows.append((d, drop_cm_val, yproj))
    df_proj = pd.DataFrame(proj_rows, columns=["Distanza (m)", "Drop (cm)", "Proiezione riser (cm)"])
    st.dataframe(df_proj, use_container_width=True)

    # 7) CSV
    csv_buf = io.StringIO()
    df_proj.to_csv(csv_buf, index=False)
    st.download_button("⬇️ Scarica mirino (CSV)", data=csv_buf.getvalue(),
                       file_name="mirino_riser.csv", mime="text/csv")

    # 8) PDF A4 1:1
    pdf_buf_name = esporta_mirino_pdf_bytes(df_proj, o_eye_cock, t_cock_riser)
    if pdf_buf_name is not None:
        pdf_buf, pdf_name = pdf_buf_name
        st.download_button("📄 Scarica mirino stampabile (PDF)", data=pdf_buf,
                           file_name=pdf_name, mime="application/pdf")

    # 9) Anteprima "scala mirino" (solo tacche + Laser 30 m; nessuna y=0 qui)
    st.markdown("### Scala del mirino – anteprima")
    valid_proj = df_proj.dropna()
    if len(valid_proj) >= 2:
        y_marks = valid_proj["Proiezione riser (cm)"].values
        dist_marks = valid_proj["Distanza (m)"].values
        y_minp = float(np.nanmin(y_marks))
        y_maxp = float(np.nanmax(y_marks))
        pad = max(2.0, 0.05 * (y_maxp - y_minp))
        y_min_plot = y_minp - pad
        y_max_plot = y_maxp + pad

        fig_m, axm = plt.subplots(figsize=(3, 8))
        axm.vlines(0, y_min_plot, y_max_plot, colors="black", linewidth=2)  # "riser"
        axm.hlines(y_marks, xmin=-0.5, xmax=0.5, colors="tab:blue")         # tacche
        for yv, dv in zip(y_marks, dist_marks):
            axm.text(0.6, yv, f"{int(dv)} m", va="center", fontsize=8)

        # Punto Laser 30 m (d=0)
        y_laser_30 = y_cm(30.0, o_eye_cock, t_cock_riser, d=0.0)
        axm.scatter(0, y_laser_30, color="blue", zorder=5)
        axm.text(0.6, y_laser_30, "Laser 30 m", va="center", fontsize=8, color="blue")

        axm.set_ylim(y_min_plot, y_max_plot)
        axm.set_xlim(-1.0, 3.0)
        axm.set_xticks([])
        axm.set_yticks(np.linspace(round(y_min_plot), round(y_max_plot), 9))
        axm.set_title("Mirino – proiezione tacche (cm)")
        axm.set_ylabel("Quota (cm) sulla colonna del riser")
        axm.grid(True, axis='y', linestyle='--', alpha=0.3)
        st.pyplot(fig_m, use_container_width=False)
    else:
        st.warning("Pochi punti validi per l’anteprima della scala del mirino.")

    # 10) Riepilogo
    dx = target_distance
    dy = target_height - y_launch
    ang_mira = np.degrees(np.arctan2(dy, dx))
    rel_angle = ang_opt - ang_mira
    st.success(
        f"**Angolo ottimale (orizzontale):** {ang_opt:.2f}°\n"
        f"**Angolo di mira (punta→bersaglio):** {ang_mira:.2f}°\n"
        f"**Scarto rispetto mira:** {rel_angle:+.2f}°\n"
        f"**Quota uscita:** {y_launch:.2f} m\n"
        f"**v₀:** {v0_calc:.2f} m/s\n"
        f"**Tempo volo:** {t1:.2f} s"
    )



