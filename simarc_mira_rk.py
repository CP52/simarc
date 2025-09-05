# simarc_mira_rk4.py
# App Streamlit completa: simulazione balistica con RK4 adattativo + generatore mirino

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
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

# ==============================
# CLASSI E COSTANTI
# ==============================
@dataclass
class SimulationParams:
    mass: float
    length: float
    spine: int
    diameter: float
    balance_point: float
    tip_type: str
    draw_force: float
    draw_length: float
    brace_height: float
    efficiency: float
    bow_type: str
    launch_height_neutral: float
    anchor_length: float
    pelvis_height: float
    eye_offset_v: float
    target_distance: float
    target_height: float
    use_measured_v0: bool
    v0: float

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

# ==============================
# FUNZIONI AERODINAMICHE
# ==============================
def reynolds_number(v: float, diameter_mm: float) -> float:
    d_m = diameter_mm / 1000.0
    return air_density * v * d_m / mu_air

def realistic_drag_coefficient(v: float, diameter_mm: float, 
                              angle_of_attack_deg: float, 
                              tip_type: str, params: SimulationParams) -> float:
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

def calculate_velocity(params: SimulationParams) -> float:
    if params.use_measured_v0:
        return params.v0
    
    mass = params.mass / 1000.0
    F = params.draw_force * 4.44822
    elong = max(0.0, params.draw_length - params.brace_height)
    E = params.efficiency * F * elong
    return np.sqrt(max(0.0, 2 * E / mass))

# ==============================
# INTEGRAZIONE RK4 ADATTATIVA
# ==============================
class RK4AdaptiveIntegrator:
    def __init__(self, tol: float = 1e-6, max_dt: float = 0.1, min_dt: float = 1e-6):
        self.tol = tol
        self.max_dt = max_dt
        self.min_dt = min_dt
        self.stats = {'steps': 0, 'rejections': 0, 'min_dt_used': float('inf')}
    
    def derivatives(self, t: float, state: np.ndarray, 
                   params: SimulationParams, angle_deg: float, 
                   include_drag: bool) -> np.ndarray:
        x, y, vx, vy = state
        mass = params.mass / 1000.0
        A = np.pi * (params.diameter / 1000.0 / 2.0) ** 2
        v = np.sqrt(vx**2 + vy**2)
        
        F_drag_x, F_drag_y = 0.0, 0.0
        if include_drag and v > 1e-6:
            gamma = np.degrees(np.arctan2(vy, vx)) - angle_deg
            Cd = realistic_drag_coefficient(v, params.diameter, gamma, 
                                          params.tip_type, params)
            F_drag = 0.5 * air_density * Cd * A * v**2
            F_drag_x = -F_drag * (vx / v)
            F_drag_y = -F_drag * (vy / v)
        
        return np.array([vx, vy, F_drag_x / mass, -G + F_drag_y / mass])
    
    def rk4_step(self, t: float, state: np.ndarray, dt: float,
                params: SimulationParams, angle_deg: float, 
                include_drag: bool) -> np.ndarray:
        k1 = self.derivatives(t, state, params, angle_deg, include_drag)
        k2 = self.derivatives(t + dt/2, state + dt/2 * k1, params, angle_deg, include_drag)
        k3 = self.derivatives(t + dt/2, state + dt/2 * k2, params, angle_deg, include_drag)
        k4 = self.derivatives(t + dt, state + dt * k3, params, angle_deg, include_drag)
        
        return state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    def integrate(self, angle_deg: float, params: SimulationParams, 
                 include_drag: bool = True, end_x: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, float, float]:
        
        angle = np.radians(angle_deg)
        v0 = calculate_velocity(params)
        end_x = float(end_x if end_x is not None else params.target_distance)
        
        # Stato iniziale: [x, y, vx, vy]
        state = np.array([0.0, params.launch_height, 
                         v0 * np.cos(angle), v0 * np.sin(angle)])
        
        X, Y, T = [state[0]], [state[1]], [0.0]
        t = 0.0
        dt = self.max_dt
        
        self.stats = {'steps': 0, 'rejections': 0, 'min_dt_used': float('inf')}
        
        while state[0] <= end_x and state[1] >= -10 and t < 10.0:
            # Tentativo di passo
            state_full = self.rk4_step(t, state, dt, params, angle_deg, include_drag)
            
            # Due mezzi passi per stima errore
            state_half1 = self.rk4_step(t, state, dt/2, params, angle_deg, include_drag)
            state_half2 = self.rk4_step(t + dt/2, state_half1, dt/2, params, angle_deg, include_drag)
            
            # Stima errore (relativo)
            error = np.linalg.norm(state_full - state_half2) / (np.linalg.norm(state) + 1e-12)
            
            if error < self.tol or dt <= self.min_dt:
                # Accetta il passo
                state = state_full
                t += dt
                X.append(state[0])
                Y.append(state[1])
                T.append(t)
                self.stats['steps'] += 1
                self.stats['min_dt_used'] = min(self.stats['min_dt_used'], dt)
                
                # Adatta il passo
                if error > 0:
                    dt = min(self.max_dt, 0.9 * dt * (self.tol/error)**0.2)
                else:
                    dt = min(self.max_dt, dt * 1.2)
            else:
                # Rifiuta il passo, riduci dt
                dt = max(self.min_dt, 0.9 * dt * (self.tol/error)**0.2)
                self.stats['rejections'] += 1
            
            # Condizioni di terminazione
            if state[0] > end_x + 5 or state[1] < -5:
                break
        
        return np.array(X), np.array(Y), v0, t

# ==============================
# FUNZIONI DI SUPPORTO
# ==============================
def get_y_at_x(X: np.ndarray, Y: np.ndarray, x_target: float) -> float:
    if len(X) < 2:
        return 0.0
    f = interp1d(X, Y, kind='linear', fill_value='extrapolate')
    return float(f(x_target))

def find_optimal_angle(params: SimulationParams, integrator: RK4AdaptiveIntegrator) -> float:
    target_y = params.target_height
    distance = params.target_distance

    def objective(angle):
        X, Y, _, _ = integrator.integrate(angle, params, include_drag=True, end_x=distance)
        if len(X) < 2:
            return 1e9
        y_at_target = get_y_at_x(X, Y, distance)
        return abs(y_at_target - target_y)

    res = minimize_scalar(objective, bounds=(-20, 45), method='bounded')
    return res.x

def calcola_quota_uscita_posturale(params: SimulationParams, angle_rad: float) -> float:
    anchor_length = params.anchor_length
    y_neutral = params.launch_height_neutral
    pelvis = params.pelvis_height

    y_pivot = y_neutral - pelvis
    dx, dy = anchor_length, pelvis
    r = np.sqrt(dx**2 + dy**2)
    phi0 = np.arcsin(dy / r)
    return y_pivot + r * np.sin(angle_rad + phi0)

def trova_angolo_convergenza(params: SimulationParams, integrator: RK4AdaptiveIntegrator, 
                           tol_angle: float = 0.01, tol_height: float = 0.001, 
                           max_iter: int = 25) -> Tuple[float, float]:
    quota_uscita = params.launch_height_neutral
    angolo = 0.0
    
    for i in range(max_iter):
        modified_params = SimulationParams(
            **{**params.__dict__, 'launch_height': quota_uscita}
        )
        
        angolo_nuovo = find_optimal_angle(modified_params, integrator)
        y_launch_new = calcola_quota_uscita_posturale(params, np.radians(angolo_nuovo))
        
        if (abs(angolo_nuovo - angolo) < tol_angle and 
            abs(y_launch_new - quota_uscita) < tol_height):
            return angolo_nuovo, y_launch_new
        
        angolo, quota_uscita = angolo_nuovo, y_launch_new
    
    return angolo, quota_uscita

def y_cm(x: float, o: float, t: float, d: float = 0.0) -> float:
    u = (o + d) / (t + x)
    y_calc = 100.0 * x * (u / np.sqrt(1 + u**2))
    return y_calc - d * 100.0

# ==============================
# VISUALIZZAZIONE
# ==============================
def plot_trajectory(X1: np.ndarray, Y1: np.ndarray, params: SimulationParams, 
                   angle: float, v0: float, tflight: float, 
                   X2: Optional[np.ndarray] = None, Y2: Optional[np.ndarray] = None, 
                   show_mira: bool = True) -> plt.Figure:
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(X1, Y1, label="Con drag (RK4 adattativo)", linewidth=2)
    
    if X2 is not None and Y2 is not None:
        ax.plot(X2, Y2, "--", label="Senza drag", alpha=0.7)

    ax.plot(params.target_distance, params.target_height, 'go', 
            markersize=8, label="Bersaglio")

    y0 = params.launch_height
    th = np.radians(angle)
    
    if show_mira:
        x_mira = np.array([0.0, params.target_distance])
        y_mira = y0 + np.tan(th) * x_mira
        ax.plot(x_mira, y_mira, 'k--', label="Linea di mira", alpha=0.7)

    y_freccia = get_y_at_x(X1, Y1, params.target_distance)
    y_mira_finale = y0 + np.tan(th) * params.target_distance
    drop_cm = (y_mira_finale - y_freccia) * 100.0
    
    ax.annotate(
        f"Drop: {drop_cm:.1f} cm",
        xy=(params.target_distance, y_freccia),
        xytext=(params.target_distance - 5, y_freccia - 0.4),
        arrowprops=dict(arrowstyle="->", color='red'),
        bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="gray", alpha=0.8)
    )

    dx = params.target_distance
    dy = params.target_height - y0
    mira_angle = np.degrees(np.arctan2(dy, dx))
    rel_angle = angle - mira_angle

    title = (
        f"Traiettoria - RK4 Adattativo\n"
        f"Angolo: {angle:.2f}° | v₀: {v0:.1f} m/s | Tvolo: {tflight:.2f} s\n"
        f"Scarto mira: {rel_angle:+.2f}° | Quota: {y0:.2f} m"
    )
    
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Distanza (m)")
    ax.set_ylabel("Altezza (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    xmax = max(params.target_distance, X1.max(), 
              (X2.max() if X2 is not None else 0.0)) + 2
    ymin = min(Y1.min(), params.target_height, 0) - 0.5
    ymax = max(Y1.max(), params.target_height) + 0.5
    
    ax.set_xlim(0, xmax)
    ax.set_ylim(ymin, ymax)
    
    return fig

# ==============================
# GENERAZIONE PDF MIRINO
# ==============================
def esporta_mirino_pdf_bytes(df_proj: pd.DataFrame, o_eye_cock: float, 
                            t_cock_riser: float, filename: str = "mirino_riser.pdf") -> Tuple[io.BytesIO, str]:
    
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    def cm2pt(x_cm: float) -> float:
        return x_cm * 28.346

    y_vals = df_proj["Proiezione riser (cm)"].dropna().values.tolist()
    y_vals.extend([0.0, y_cm(30.0, o_eye_cock, t_cock_riser, 0.0)])
    y_min, y_max = float(min(y_vals)), float(max(y_vals))

    x_center = width / 2.0
    margin_bottom = cm2pt(5.0)
    y0_pt = margin_bottom - cm2pt(y_min)

    c.setLineWidth(2)
    c.line(x_center, y0_pt + cm2pt(y_min), x_center, y0_pt + cm2pt(y_max))

    c.setFont("Helvetica", 8)
    for i, (_, row) in enumerate(df_proj.dropna().iterrows()):
        y_pt = y0_pt + cm2pt(float(row["Proiezione riser (cm)"]))
        c.line(x_center - 20, y_pt, x_center + 20, y_pt)
        
        if i % 2 == 0:
            c.drawString(x_center + 30, y_pt - 3, f"{int(row['Distanza (m)'])} m")
        else:
            text = f"{int(row['Distanza (m)'])} m"
            tw = c.stringWidth(text, "Helvetica", 8)
            c.drawString(x_center - 30 - tw, y_pt - 3, text)

    y_zero_pt = y0_pt + cm2pt(0.0)
    c.setStrokeColorRGB(1, 0, 0)
    c.line(x_center - 25, y_zero_pt, x_center + 25, y_zero_pt)
    c.setFont("Helvetica-Bold", 8)
    c.drawString(x_center + 30, y_zero_pt - 3, "0 cm (base)")
    c.setStrokeColorRGB(0, 0, 0)

    y_laser_pt = y0_pt + cm2pt(y_cm(30.0, o_eye_cock, t_cock_riser, 0.0))
    c.setFillColorRGB(0, 1, 0)
    c.circle(x_center, y_laser_pt, 2.5, fill=1, stroke=0)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(x_center + 30, y_laser_pt - 4, "Laser 30 m")

    c.setLineWidth(3)
    y_bar = y0_pt + cm2pt(y_min) - 40
    c.line(x_center - cm2pt(2.5), y_bar, x_center + cm2pt(2.5), y_bar)
    c.setFont("Helvetica", 9)
    c.drawCentredString(x_center, y_bar - 12, "Oriz. 5 cm")
    
    x_bar = x_center + 80
    c.line(x_bar, y_bar, x_bar, y_bar + cm2pt(5.0))
    c.drawCentredString(x_bar, y_bar + 12, "Vert. 5 cm")

    c.showPage()
    c.save()
    buf.seek(0)
    return buf, filename

# ==============================
# INTERFACCIA STREAMLIT
# ==============================
def main():
    st.set_page_config(page_title="Simulatore RK4 + Mirino", layout="wide")
    st.title("🏹 Simulatore Balistico RK4 + Generatore Mirino")
    
    st.sidebar.header("⚙️ Impostazioni Simulazione")
    tol = st.sidebar.slider("Tolleranza RK4", 1e-8, 1e-4, 1e-6, format="%.1e")
    max_dt = st.sidebar.slider("Passo massimo (s)", 0.001, 0.1, 0.01)
    min_dt = st.sidebar.slider("Passo minimo (s)", 1e-8, 0.001, 1e-6, format="%.1e")
    
    st.sidebar.header("📊 Curva Drop")
    d_min = st.sidebar.number_input("Distanza minima (m)", 5, 100, 10)
    d_max = st.sidebar.number_input("Distanza massima (m)", 10, 100, 50)
    d_step = st.sidebar.number_input("Passo (m)", 1, 10, 5)
    fit_degree = st.sidebar.slider("Grado polinomio", 1, 5, 2)
    d_query = st.sidebar.number_input("Query drop(x) (m)", 5, 100, 30)

    colA, colB = st.columns(2)

    with colA:
        st.subheader("🎯 Freccia")
        mass = st.number_input("Peso (g)", 10.0, 50.0, 24.0)
        length = st.number_input("Lunghezza (m)", 0.5, 1.0, 0.75)
        diameter = st.number_input("Diametro (mm)", 4.0, 8.0, 6.2)
        spine = st.number_input("Spine", 200, 1200, 700)
        balance_point = st.number_input("Punto di bilanciamento (m)", 0.2, 0.8, 0.4)
        tip_type = st.selectbox("Tipo di punta", list(TIPO_PUNTA_CD_FACTOR.keys()))

        st.subheader("🏹 Arco")
        draw_force = st.number_input("Forza (lb)", 10.0, 80.0, 36.0)
        draw_length = st.number_input("Allungo (m)", 0.5, 1.1, 0.70)
        brace_height = st.number_input("Brace (m)", 0.05, 0.30, 0.18)
        efficiency = st.number_input("Efficienza", 0.5, 0.95, 0.82)
        bow_type = st.selectbox("Tipo di arco", list(BOW_TYPE_DEFAULT_EFF.keys()))

    with colB:
        st.subheader("👤 Arciere")
        launch_height_neutral = st.number_input("Quota uscita neutra (m)", 0.8, 2.2, 1.5)
        anchor_length = st.number_input("Lunghezza spalla–aggancio (m)", 0.4, 1.0, 0.75)
        pelvis_height = st.number_input("Quota bacino (m)", 0.3, 1.5, 1.0)
        eye_offset_v = st.number_input("Offset verticale occhio (m)", 0.01, 0.25, 0.09)

        st.subheader("🎯 Bersaglio")
        target_distance = st.number_input("Distanza bersaglio (m)", 1.0, 150.0, 50.0)
        target_height = st.number_input("Quota bersaglio (m)", -20.0, 20.0, 1.5)

        st.subheader("⚙️ Opzioni")
        use_measured_v0 = st.checkbox("Usa v₀ misurata")
        v0 = st.number_input("v₀ misurata (m/s)", 5.0, 120.0, 55.0, disabled=not use_measured_v0)
        show_mira = st.checkbox("Mostra linea di mira", value=True)
        show_compare = st.checkbox("Confronta con traiettoria ideale", value=False)

    st.subheader("📐 Geometria visiva mirino")
    colG1, colG2 = st.columns(2)
    with colG1:
        o_eye_cock = st.number_input("Distanza occhio–cocca (m)", 0.05, 0.40, 0.11, 
                                   step=0.005, format="%.3f")
    with colG2:
        t_cock_riser = st.number_input("Distanza cocca–riser (m)", 0.2, 1.5, 0.70, step=0.01)

    if st.button("🚀 Calcola e genera mirino", type="primary"):
        params = SimulationParams(
            mass=mass, length=length, spine=spine, diameter=diameter,
            balance_point=balance_point, tip_type=tip_type,
            draw_force=draw_force, draw_length=draw_length, brace_height=brace_height,
            efficiency=efficiency, bow_type=bow_type,
            launch_height_neutral=launch_height_neutral, anchor_length=anchor_length,
            pelvis_height=pelvis_height, eye_offset_v=eye_offset_v,
            target_distance=target_distance, target_height=target_height,
            use_measured_v0=use_measured_v0, v0=v0,
            launch_height=launch_height_neutral  # Valore iniziale
        )

        if not use_measured_v0 and abs(efficiency - 0.82) < 1e-6:
            params.efficiency = BOW_TYPE_DEFAULT_EFF.get(bow_type, efficiency)

        # Inizializza integratore RK4
        integrator = RK4AdaptiveIntegrator(tol=tol, max_dt=max_dt, min_dt=min_dt)
        
        with st.spinner("🔍 Calcolo angolo ottimale..."):
            ang_opt, y_launch = trova_angolo_convergenza(params, integrator)
            params.launch_height = y_launch

        with st.spinner("📈 Simulazione traiettoria..."):
            sim_end = max(float(d_max), float(target_distance))
            X1, Y1, v0_calc, t1 = integrator.integrate(ang_opt, params, include_drag=True, end_x=sim_end)
            
            if show_compare:
                X2, Y2, _, _ = integrator.integrate(ang_opt, params, include_drag=False, end_x=sim_end)
            else:
                X2, Y2 = None, None

        # Visualizzazione risultati
        st.markdown("### 📊 Traiettoria")
        fig_traj = plot_trajectory(X1, Y1, params, ang_opt, v0_calc, t1, X2, Y2, show_mira)
        st.pyplot(fig_traj, use_container_width=True)
        
        # Statistiche integratore
        st.sidebar.info(f"**Statistiche RK4:**\n"
                       f"Passi accettati: {integrator.stats['steps']}\n"
                       f"Passi rifiutati: {integrator.stats['rejections']}\n"
                       f"Passo minimo: {integrator.stats['min_dt_used']:.2e} s")

        # [Resto del codice per drop curve, mirino, PDF... identico alla versione originale]
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
            st.warning("Pochi punti validi per l'anteprima della scala del mirino.")

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
if __name__ == "__main__":
    main()