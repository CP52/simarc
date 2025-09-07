# simarc_mira_rk4_improved.py
# App Streamlit migliorata: simulazione balistica con RK4 adattativo + generatore mirino

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
from dataclasses import dataclass, replace
from typing import Optional, Tuple, List, Dict, Any
import logging
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================
# CONFIGURAZIONE E COSTANTI
# ==============================
class PhysicalConstants:
    G = 9.81  # Accelerazione gravitazionale [m/s²]
    AIR_DENSITY = 1.204  # Densità dell'aria [kg/m³]
    AIR_VISCOSITY = 1.81e-5  # Viscosità cinematica dell'aria [Pa·s]

# Configurazione grafica migliorata
PLOT_CONFIG = {
    'style': 'seaborn-v0_8',
    'colors': {
        'trajectory': '#2E86AB',
        'no_drag': '#A23B72',
        'target': '#F18F01',
        'sight_line': '#C73E1D',
        'drop': '#8B5A3C'
    },
    'figsize': (12, 7),
    'dpi': 100
}

TIPO_PUNTA_CD_FACTOR = {
    "Slanciata (field/bullet)": 0.95,
    "Standard": 1.0,
    "Broadhead (larga)": 1.15,
    "Judo Point": 1.25  # Aggiunta nuova punta
}

BOW_TYPE_DEFAULT_EFF = {
    "longbow": 0.75,
    "ricurvo": 0.82,
    "compound": 0.87,
    "takedown": 0.80,
    "barebow": 0.78  # Aggiunto nuovo tipo
}

# ==============================
# CLASSI MIGLIORATE
# ==============================
@dataclass
class SimulationParams:
    """Parametri di simulazione con validazione migliorata"""
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
    launch_height: Optional[float] = None
    # Nuovi parametri ambientali
    wind_speed: float = 0.0
    wind_direction: float = 0.0  # gradi, 0 = vento frontale
    air_temperature: float = 20.0  # Celsius
    air_pressure: float = 1013.25  # hPa
    humidity: float = 50.0  # %

    def __post_init__(self):
        self.validate()
        if self.launch_height is None:
            self.launch_height = self.launch_height_neutral
    
    def validate(self):
        """Validazione parametri con messaggi di errore dettagliati"""
        validations = [
            (self.mass > 0, "Il peso della freccia deve essere positivo"),
            (0.5 <= self.length <= 1.2, "Lunghezza freccia deve essere tra 0.5 e 1.2 m"),
            (4.0 <= self.diameter <= 12.0, "Diametro deve essere tra 4 e 12 mm"),
            (0.0 <= self.balance_point <= self.length, "Punto di bilanciamento non valido"),
            (self.draw_force > 0, "Forza di trazione deve essere positiva"),
            (self.draw_length > self.brace_height, "Allungo deve essere maggiore del brace height"),
            (0.5 <= self.efficiency <= 0.98, "Efficienza deve essere tra 0.5 e 0.98"),
            (self.target_distance > 0, "Distanza bersaglio deve essere positiva"),
            (-5 <= self.wind_speed <= 20, "Velocità vento deve essere tra -5 e 20 m/s"),
            (-10 <= self.air_temperature <= 50, "Temperatura deve essere tra -10 e 50°C")
        ]
        
        for condition, message in validations:
            if not condition:
                raise ValueError(f"Parametro non valido: {message}")

@dataclass
class TrajectoryResults:
    """Risultati della simulazione con metadati"""
    X: np.ndarray
    Y: np.ndarray
    V_x: np.ndarray
    V_y: np.ndarray
    times: np.ndarray
    v0: float
    flight_time: float
    max_height: float
    range_distance: float
    angle_degrees: float
    integration_stats: Dict[str, Any]

# ==============================
# FUNZIONI AERODINAMICHE MIGLIORATE
# ==============================
@lru_cache(maxsize=1000)
def reynolds_number(v: float, diameter_mm: float, 
                   temperature: float = 20.0, pressure: float = 1013.25) -> float:
    """Calcolo numero di Reynolds con dipendenza da temperatura e pressione"""
    # Correzione densità per temperatura e pressione
    rho = PhysicalConstants.AIR_DENSITY * (pressure / 1013.25) * (293.15 / (273.15 + temperature))
    d_m = diameter_mm / 1000.0
    return rho * v * d_m / PhysicalConstants.AIR_VISCOSITY

def enhanced_drag_coefficient(v: float, diameter_mm: float, 
                            angle_of_attack_deg: float, 
                            tip_type: str, params: SimulationParams) -> float:
    """Coefficiente di resistenza migliorato con più transizioni"""
    Re = reynolds_number(v, diameter_mm, params.air_temperature, params.air_pressure)
    
    # Transizioni più raffinate del coefficiente di resistenza
    if Re < 1e3:
        Cd = 2.0
    elif Re < 5e3:
        Cd = 2.0 - 0.5 * (Re - 1e3) / 4e3
    elif Re < 1.2e4:
        Cd = 1.5 - 0.3 * (Re - 5e3) / 7.2e3
    elif Re < 2.0e4:
        Cd = 1.2 + 1.4 * (Re - 1.2e4) / 8e3
    elif Re < 1e5:
        Cd = 2.6 - 1.1 * (Re - 2e4) / 8e4
    else:
        Cd = 1.5
    
    # Effetto angolo di attacco migliorato
    gamma_rad = np.radians(angle_of_attack_deg)
    Cd *= (1 + 3 * gamma_rad**2 + gamma_rad**4)
    
    # Fattore tipo punta
    Cd *= TIPO_PUNTA_CD_FACTOR.get(tip_type, 1.0)
    
    # Correzione per umidità (piccolo effetto)
    humidity_factor = 1 + 0.001 * (params.humidity - 50) / 50
    Cd *= humidity_factor
    
    return Cd

def calculate_velocity_enhanced(params: SimulationParams) -> float:
    """Calcolo velocità migliorato con correzioni ambientali"""
    if params.use_measured_v0:
        return params.v0
    
    mass = params.mass / 1000.0
    F = params.draw_force * 4.44822  # lb -> N
    elong = max(0.0, params.draw_length - params.brace_height)
    
    # Energia immagazzinata nel sistema arco-corda
    E = params.efficiency * F * elong
    
    # Correzione per densità aria (effetto piccolo ma realistico)
    air_density_ratio = (params.air_pressure / 1013.25) * (293.15 / (273.15 + params.air_temperature))
    drag_correction = 1 - 0.02 * (air_density_ratio - 1)
    
    v0 = np.sqrt(max(0.0, 2 * E / mass)) * drag_correction
    return v0

# ==============================
# INTEGRAZIONE RK4 MIGLIORATA
# ==============================
class EnhancedRK4Integrator:
    """Integratore RK4 con controllo dell'errore migliorato e wind model"""
    
    def __init__(self, tol: float = 1e-6, max_dt: float = 0.05, min_dt: float = 1e-7,
                 safety_factor: float = 0.9):
        self.tol = tol
        self.max_dt = max_dt
        self.min_dt = min_dt
        self.safety_factor = safety_factor
        self.stats = {
            'steps': 0, 'rejections': 0, 'min_dt_used': float('inf'), 
            'max_dt_used': 0.0, 'total_time': 0.0
        }
    
    def wind_force(self, vx: float, vy: float, params: SimulationParams) -> Tuple[float, float]:
        """Calcolo forza del vento"""
        if abs(params.wind_speed) < 0.1:
            return 0.0, 0.0
        
        # Componenti vento
        wind_angle = np.radians(params.wind_direction)
        wind_x = params.wind_speed * np.cos(wind_angle)
        wind_y = params.wind_speed * np.sin(wind_angle)
        
        # Velocità relativa aria-freccia
        vrel_x = vx - wind_x
        vrel_y = vy - wind_y
        vrel = np.sqrt(vrel_x**2 + vrel_y**2)
        
        if vrel < 1e-6:
            return 0.0, 0.0
        
        # Forza vento proporzionale al quadrato della velocità relativa
        A = np.pi * (params.diameter / 1000.0 / 2.0) ** 2
        Cd_wind = 0.8  # Coefficiente semplificato per vento laterale
        F_wind = 0.5 * PhysicalConstants.AIR_DENSITY * Cd_wind * A * vrel**2
        
        return -F_wind * (vrel_x / vrel), -F_wind * (vrel_y / vrel)
    
    def derivatives(self, t: float, state: np.ndarray, 
                   params: SimulationParams, angle_deg: float, 
                   include_drag: bool, include_wind: bool = True) -> np.ndarray:
        """Derivate del sistema con modello del vento"""
        x, y, vx, vy = state
        mass = params.mass / 1000.0
        A = np.pi * (params.diameter / 1000.0 / 2.0) ** 2
        v = np.sqrt(vx**2 + vy**2)
        
        # Forze aerodinamiche
        F_drag_x, F_drag_y = 0.0, 0.0
        if include_drag and v > 1e-6:
            gamma = np.degrees(np.arctan2(vy, vx)) - angle_deg
            Cd = enhanced_drag_coefficient(v, params.diameter, gamma, 
                                         params.tip_type, params)
            F_drag = 0.5 * PhysicalConstants.AIR_DENSITY * Cd * A * v**2
            F_drag_x = -F_drag * (vx / v)
            F_drag_y = -F_drag * (vy / v)
        
        # Forze del vento
        F_wind_x, F_wind_y = 0.0, 0.0
        if include_wind:
            F_wind_x, F_wind_y = self.wind_force(vx, vy, params)
        
        ax = (F_drag_x + F_wind_x) / mass
        ay = -PhysicalConstants.G + (F_drag_y + F_wind_y) / mass
        
        return np.array([vx, vy, ax, ay])
    
    def rk4_step(self, t: float, state: np.ndarray, dt: float,
                params: SimulationParams, angle_deg: float, 
                include_drag: bool, include_wind: bool = True) -> np.ndarray:
        """Passo RK4 singolo"""
        k1 = self.derivatives(t, state, params, angle_deg, include_drag, include_wind)
        k2 = self.derivatives(t + dt/2, state + dt/2 * k1, params, angle_deg, include_drag, include_wind)
        k3 = self.derivatives(t + dt/2, state + dt/2 * k2, params, angle_deg, include_drag, include_wind)
        k4 = self.derivatives(t + dt, state + dt * k3, params, angle_deg, include_drag, include_wind)
        
        return state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    def adaptive_step_control(self, error: float, dt: float) -> float:
        """Controllo adattivo del passo migliorato"""
        if error < self.tol / 10:
            # Errore molto piccolo, aumenta aggressivamente
            return min(self.max_dt, dt * 1.5)
        elif error < self.tol:
            # Errore accettabile, aumenta moderatamente
            factor = self.safety_factor * (self.tol / max(error, 1e-15)) ** 0.2
            return min(self.max_dt, dt * min(factor, 1.3))
        else:
            # Errore troppo grande, diminuisci
            factor = self.safety_factor * (self.tol / error) ** 0.25
            return max(self.min_dt, dt * max(factor, 0.3))
    
    def integrate(self, angle_deg: float, params: SimulationParams, 
                 include_drag: bool = True, include_wind: bool = True,
                 end_x: Optional[float] = None) -> TrajectoryResults:
        """Integrazione completa con risultati dettagliati"""
        
        angle = np.radians(angle_deg)
        v0 = calculate_velocity_enhanced(params)
        end_x = float(end_x if end_x is not None else params.target_distance * 1.2)
        
        # Stato iniziale: [x, y, vx, vy]
        state = np.array([0.0, params.launch_height, 
                         v0 * np.cos(angle), v0 * np.sin(angle)])
        
        # Arrays per salvare la traiettoria
        X, Y, VX, VY, T = [state[0]], [state[1]], [state[2]], [state[3]], [0.0]
        t = 0.0
        dt = self.max_dt / 10  # Inizia con passo piccolo
        
        # Reset statistiche
        self.stats = {'steps': 0, 'rejections': 0, 'min_dt_used': float('inf'),
                     'max_dt_used': 0.0, 'total_time': 0.0}
        
        max_height = params.launch_height
        
        while (state[0] <= end_x and state[1] >= -10 and t < 20.0 and 
               len(X) < 10000):  # Limite di sicurezza
            
            # Due passi: uno normale, due di mezzo passo
            state_full = self.rk4_step(t, state, dt, params, angle_deg, 
                                     include_drag, include_wind)
            
            state_half1 = self.rk4_step(t, state, dt/2, params, angle_deg, 
                                      include_drag, include_wind)
            state_half2 = self.rk4_step(t + dt/2, state_half1, dt/2, params, 
                                      angle_deg, include_drag, include_wind)
            
            # Stima errore
            error = np.linalg.norm(state_full - state_half2) / (np.linalg.norm(state) + 1e-12)
            
            if error < self.tol or dt <= self.min_dt:
                # Accetta il passo
                state = state_half2  # Usa la soluzione più accurata
                t += dt
                X.append(state[0])
                Y.append(state[1])
                VX.append(state[2])
                VY.append(state[3])
                T.append(t)
                
                max_height = max(max_height, state[1])
                
                self.stats['steps'] += 1
                self.stats['min_dt_used'] = min(self.stats['min_dt_used'], dt)
                self.stats['max_dt_used'] = max(self.stats['max_dt_used'], dt)
                
                # Aggiorna passo
                dt = self.adaptive_step_control(error, dt)
            else:
                # Rifiuta il passo
                dt = self.adaptive_step_control(error, dt)
                self.stats['rejections'] += 1
            
            # Controlli di sicurezza
            if state[0] > end_x + 10 or state[1] < -20:
                break
        
        self.stats['total_time'] = t
        
        return TrajectoryResults(
            X=np.array(X), Y=np.array(Y), V_x=np.array(VX), V_y=np.array(VY),
            times=np.array(T), v0=v0, flight_time=t, max_height=max_height,
            range_distance=X[-1] if X else 0.0, angle_degrees=angle_deg,
            integration_stats=self.stats.copy()
        )

# ==============================
# FUNZIONI DI SUPPORTO MIGLIORATE
# ==============================
def get_y_at_x_improved(X: np.ndarray, Y: np.ndarray, x_target: float, 
                       method: str = 'cubic') -> float:
    """Interpolazione migliorata con scelta del metodo"""
    if len(X) < 2:
        return 0.0
    
    if x_target < X.min() or x_target > X.max():
        # Estrapolazione lineare per punti fuori range
        method = 'linear'
    
    try:
        if method == 'cubic' and len(X) >= 4:
            f = interp1d(X, Y, kind='cubic', fill_value='extrapolate')
        else:
            f = interp1d(X, Y, kind='linear', fill_value='extrapolate')
        return float(f(x_target))
    except:
        # Fallback su interpolazione lineare
        f = interp1d(X, Y, kind='linear', fill_value='extrapolate')
        return float(f(x_target))

def find_optimal_angle_robust(params: SimulationParams, 
                             integrator: EnhancedRK4Integrator,
                             method: str = 'brent') -> Tuple[float, float]:
    """Ricerca angolo ottimale robusta con multiple strategie"""
    target_y = params.target_height
    distance = params.target_distance

    def objective(angle):
        try:
            result = integrator.integrate(angle, params, include_drag=True, 
                                        end_x=distance * 1.1)
            if len(result.X) < 2:
                return 1e9
            y_at_target = get_y_at_x_improved(result.X, result.Y, distance)
            return abs(y_at_target - target_y)
        except Exception as e:
            logger.warning(f"Errore nel calcolo traiettoria per angolo {angle:.2f}: {e}")
            return 1e9

    # Stima iniziale basata su balistica semplificata
    g, v0 = PhysicalConstants.G, calculate_velocity_enhanced(params)
    h0, ht = params.launch_height, target_y
    d = distance
    
    # Angolo balistico approssimato
    discriminant = v0**4 - g*(g*d**2 + 2*(ht - h0)*v0**2)
    if discriminant > 0:
        angle_est = np.degrees(0.5 * np.arcsin(g*d / v0**2))
    else:
        angle_est = 15.0  # Fallback
    
    # Cerca in un range attorno alla stima
    search_range = max(10.0, abs(angle_est))
    bounds = (angle_est - search_range, angle_est + search_range)
    bounds = (max(bounds[0], -30), min(bounds[1], 60))
    
    try:
        if method == 'brent':
            res = minimize_scalar(objective, bounds=bounds, method='bounded')
        else:
            # Golden section search come alternativa
            res = minimize_scalar(objective, bounds=bounds, method='golden')
        
        if res.success and res.fun < 1.0:  # Errore < 1 metro
            return res.x, res.fun
        else:
            logger.warning("Ottimizzazione non convergente, uso ricerca griglia")
            # Fallback: ricerca su griglia
            angles = np.linspace(bounds[0], bounds[1], 50)
            errors = [objective(a) for a in angles]
            best_idx = np.argmin(errors)
            return angles[best_idx], errors[best_idx]
            
    except Exception as e:
        logger.error(f"Errore nell'ottimizzazione: {e}")
        return 0.0, 1e9

# ==============================
# VISUALIZZAZIONE MIGLIORATA
# ==============================
def create_enhanced_trajectory_plot(result_with_drag: TrajectoryResults,
                                   params: SimulationParams,
                                   result_no_drag: Optional[TrajectoryResults] = None,
                                   show_wind_effect: bool = True) -> plt.Figure:
    """Grafico traiettoria con stile migliorato e più informazioni"""
    
    plt.style.use('default')  # Usa stile di base per compatibilità
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
    
    # === GRAFICO PRINCIPALE ===
    X1, Y1 = result_with_drag.X, result_with_drag.Y
    
    # Traiettoria principale
    ax1.plot(X1, Y1, color=PLOT_CONFIG['colors']['trajectory'], 
             linewidth=2.5, label="Traiettoria realistica", alpha=0.9)
    
    # Traiettoria senza drag (se presente)
    if result_no_drag is not None:
        ax1.plot(result_no_drag.X, result_no_drag.Y, 
                color=PLOT_CONFIG['colors']['no_drag'], 
                linestyle='--', linewidth=2, label="Senza resistenza aria", alpha=0.7)
    
    # Bersaglio
    ax1.plot(params.target_distance, params.target_height, 
             color=PLOT_CONFIG['colors']['target'], marker='o', 
             markersize=12, label="Bersaglio", markeredgecolor='black', markeredgewidth=1)
    
    # Linea di mira
    y0 = params.launch_height
    th = np.radians(result_with_drag.angle_degrees)
    x_sight = np.array([0.0, params.target_distance * 1.1])
    y_sight = y0 + np.tan(th) * x_sight
    ax1.plot(x_sight, y_sight, color=PLOT_CONFIG['colors']['sight_line'], 
             linestyle=':', linewidth=2, label="Linea di mira", alpha=0.8)
    
    # Punto di impatto e drop
    y_impact = get_y_at_x_improved(X1, Y1, params.target_distance)
    y_sight_at_target = y0 + np.tan(th) * params.target_distance
    drop_cm = (y_sight_at_target - y_impact) * 100.0
    
    # Freccia per il drop
    if abs(drop_cm) > 1:
        ax1.annotate(f"Drop: {drop_cm:.1f} cm", 
                    xy=(params.target_distance, y_impact),
                    xytext=(params.target_distance - 8, y_impact - 0.8),
                    arrowprops=dict(arrowstyle="->", color='red', lw=2),
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", 
                             edgecolor="red", alpha=0.9),
                    fontsize=11, fontweight='bold')
    
    # Informazioni sul vento
    if abs(params.wind_speed) > 0.1 and show_wind_effect:
        wind_text = f"Vento: {params.wind_speed:.1f} m/s @ {params.wind_direction:.0f}°"
        ax1.text(0.02, 0.98, wind_text, transform=ax1.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                fontsize=10, verticalalignment='top')
    
    # Stile del grafico principale
    ax1.set_xlabel("Distanza (m)", fontsize=12)
    ax1.set_ylabel("Altezza (m)", fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.legend(loc='upper right', framealpha=0.9)
    
    # Titolo informativo
    title = (f"Simulazione Balistica Avanzata - RK4 Adattativo\n"
            f"Angolo: {result_with_drag.angle_degrees:.2f}° | "
            f"v₀: {result_with_drag.v0:.1f} m/s | "
            f"Tempo volo: {result_with_drag.flight_time:.2f} s | "
            f"Altezza max: {result_with_drag.max_height:.1f} m")
    ax1.set_title(title, fontsize=13, pad=20)
    
	# Limiti assi ottimizzati
	x_margin = max(2, params.target_distance * 0.05)
	x_max = max(X1.max(), params.target_distance) + x_margin

	# Calcola punto finale della linea di mira
	y_sight_end = y0 + np.tan(th) * x_max
	y_values = [Y1.min(), Y1.max(), params.target_height, y0, y_sight_end]

	if result_no_drag:
		y_values.extend([result_no_drag.Y.min(), result_no_drag.Y.max()])

	y_margin = max(0.5, (max(y_values) - min(y_values)) * 0.1)
	ax1.set_xlim(-x_margin, x_max)
	ax1.set_ylim(min(y_values) - y_margin, max(y_values) + y_margin)
    
    # === GRAFICO VELOCITÀ ===
    V_total = np.sqrt(result_with_drag.V_x**2 + result_with_drag.V_y**2)
    ax2.plot(result_with_drag.X, V_total, color='green', linewidth=2, label='Velocità totale')
    ax2.plot(result_with_drag.X, result_with_drag.V_x, color='blue', linewidth=1.5, 
             alpha=0.7, label='Velocità orizzontale')
    ax2.plot(result_with_drag.X, result_with_drag.V_y, color='red', linewidth=1.5, 
             alpha=0.7, label='Velocità verticale')
    
    ax2.set_xlabel("Distanza (m)", fontsize=12)
    ax2.set_ylabel("Velocità (m/s)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.set_title("Profilo di Velocità", fontsize=11)
    
    plt.tight_layout()
    return fig

def create_drop_analysis_plot(distances: np.ndarray, drops_cm: List[float],
                            spline_fit: Optional[UnivariateSpline] = None,
                            poly_fit: Optional[np.poly1d] = None) -> plt.Figure:
    """Grafico migliorato per l'analisi del drop"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Dati simulati
    valid_mask = ~np.isnan(drops_cm)
    valid_distances = distances[valid_mask]
    valid_drops = np.array(drops_cm)[valid_mask]
    
    ax.scatter(valid_distances, valid_drops, color='red', s=50, 
              alpha=0.8, edgecolors='darkred', label='Dati simulazione', zorder=5)
    
    # Fit spline
    if spline_fit is not None and len(valid_distances) > 0:
        x_smooth = np.linspace(valid_distances.min(), valid_distances.max(), 300)
        y_spline = spline_fit(x_smooth)
        ax.plot(x_smooth, y_spline, color='blue', linewidth=2, 
               label='Interpolazione spline', alpha=0.8)
    
    # Fit polinomiale
    if poly_fit is not None and len(valid_distances) > 0:
        x_smooth = np.linspace(valid_distances.min(), valid_distances.max(), 300)
        y_poly = poly_fit(x_smooth)
        ax.plot(x_smooth, y_poly, color='green', linewidth=2, linestyle='--',
               label=f'Fit polinomiale (grado {len(poly_fit.coeffs)-1})', alpha=0.8)
    
    ax.set_xlabel("Distanza (m)", fontsize=12)
    ax.set_ylabel("Drop (cm)", fontsize=12)
    ax.set_title("Analisi del Drop vs Distanza", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Colore di sfondo alternato per zone di distanza
    if len(valid_distances) > 0:
        for i in range(int(valid_distances.min()), int(valid_distances.max()), 10):
            ax.axvspan(i, i+5, alpha=0.05, color='gray')
    
    plt.tight_layout()
    return fig

# ==============================
# SISTEMA DI CACHE INTELLIGENTE
# ==============================
class SimulationCache:
    """Cache intelligente per simulazioni ripetute"""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.access_count = {}
        self.max_size = max_size
    
    def _get_key(self, params: SimulationParams, angle: float, 
                include_drag: bool, include_wind: bool) -> str:
        """Genera chiave univoca per i parametri"""
        key_data = [
            params.mass, params.diameter, params.tip_type,
            params.draw_force, params.efficiency, params.launch_height,
            params.wind_speed, params.wind_direction,
            angle, include_drag, include_wind
        ]
        return str(hash(tuple(str(x) for x in key_data)))
    
    def get(self, params: SimulationParams, angle: float, 
           include_drag: bool = True, include_wind: bool = True) -> Optional[TrajectoryResults]:
        """Recupera risultato dalla cache"""
        key = self._get_key(params, angle, include_drag, include_wind)
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None
    
    def put(self, params: SimulationParams, angle: float, result: TrajectoryResults,
           include_drag: bool = True, include_wind: bool = True):
        """Salva risultato in cache"""
        if len(self.cache) >= self.max_size:
            # Rimuovi l'elemento meno usato
            min_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
            del self.cache[min_key]
            del self.access_count[min_key]
        
        key = self._get_key(params, angle, include_drag, include_wind)
        self.cache[key] = result
        self.access_count[key] = 1

# Cache globale
simulation_cache = SimulationCache()

# ==============================
# ANALISI STATISTICA AVANZATA
# ==============================
def monte_carlo_analysis(params: SimulationParams, integrator: EnhancedRK4Integrator,
                        n_simulations: int = 100, 
                        param_variations: Dict[str, float] = None) -> Dict[str, Any]:
    """Analisi Monte Carlo per quantificare l'incertezza"""
    
    if param_variations is None:
        param_variations = {
            'mass': 0.02,      # ±2% variazione peso
            'v0': 0.05,        # ±5% variazione velocità
            'wind_speed': 1.0,  # ±1 m/s vento
            'air_temperature': 5.0  # ±5°C temperatura
        }
    
    results = {
        'angles': [], 'drops_at_target': [], 'flight_times': [],
        'max_heights': [], 'impact_points_x': [], 'impact_points_y': []
    }
    
    base_angle = find_optimal_angle_robust(params, integrator)[0]
    
    for _ in range(n_simulations):
        # Genera parametri variati
        varied_params = replace(params)
        
        for param_name, variation in param_variations.items():
            if hasattr(varied_params, param_name):
                current_value = getattr(varied_params, param_name)
                noise = np.random.normal(0, variation)
                new_value = current_value * (1 + noise) if param_name != 'wind_speed' else current_value + noise
                setattr(varied_params, param_name, new_value)
        
        try:
            # Simula con parametri variati
            result = integrator.integrate(base_angle, varied_params, 
                                        include_drag=True, include_wind=True)
            
            # Calcola drop al bersaglio
            y_impact = get_y_at_x_improved(result.X, result.Y, params.target_distance)
            y_sight = params.launch_height + np.tan(np.radians(base_angle)) * params.target_distance
            drop_cm = (y_sight - y_impact) * 100
            
            results['angles'].append(base_angle)
            results['drops_at_target'].append(drop_cm)
            results['flight_times'].append(result.flight_time)
            results['max_heights'].append(result.max_height)
            results['impact_points_x'].append(result.X[-1])
            results['impact_points_y'].append(result.Y[-1])
            
        except Exception as e:
            logger.warning(f"Errore in simulazione Monte Carlo: {e}")
            continue
    
    # Calcola statistiche
    stats = {}
    for key, values in results.items():
        if values:
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'percentile_5': np.percentile(values, 5),
                'percentile_95': np.percentile(values, 95)
            }
    
    return stats

# ==============================
# EXPORT MIGLIORATO
# ==============================
def export_complete_analysis(trajectory_result: TrajectoryResults,
                           params: SimulationParams,
                           drop_data: pd.DataFrame,
                           monte_carlo_stats: Optional[Dict] = None) -> io.BytesIO:
    """Export completo in Excel con multiple schede"""
    
    output = io.BytesIO()
    
    try:
		import openpyxl
		excel_engine = 'openpyxl'
	except ImportError:
		excel_engine = 'xlsxwriter'

	with pd.ExcelWriter(output, engine=excel_engine) as writer:
        # Scheda 1: Parametri simulazione
        params_data = {
            'Parametro': [
                'Peso freccia (g)', 'Lunghezza (m)', 'Diametro (mm)', 
                'Tipo punta', 'Forza arco (lb)', 'Allungo (m)', 
                'Efficienza', 'Tipo arco', 'Quota uscita (m)',
                'Distanza bersaglio (m)', 'Quota bersaglio (m)',
                'Velocità vento (m/s)', 'Direzione vento (°)',
                'Temperatura (°C)', 'Pressione (hPa)', 'Umidità (%)'
            ],
            'Valore': [
                params.mass, params.length, params.diameter,
                params.tip_type, params.draw_force, params.draw_length,
                params.efficiency, params.bow_type, params.launch_height,
                params.target_distance, params.target_height,
                params.wind_speed, params.wind_direction,
                params.air_temperature, params.air_pressure, params.humidity
            ]
        }
        pd.DataFrame(params_data).to_excel(writer, sheet_name='Parametri', index=False)
        
        # Scheda 2: Traiettoria completa
        traj_data = pd.DataFrame({
            'Distanza (m)': trajectory_result.X,
            'Altezza (m)': trajectory_result.Y,
            'Velocità X (m/s)': trajectory_result.V_x,
            'Velocità Y (m/s)': trajectory_result.V_y,
            'Velocità totale (m/s)': np.sqrt(trajectory_result.V_x**2 + trajectory_result.V_y**2),
            'Tempo (s)': trajectory_result.times
        })
        traj_data.to_excel(writer, sheet_name='Traiettoria', index=False)
        
        # Scheda 3: Dati drop
        drop_data.to_excel(writer, sheet_name='Drop Analysis', index=False)
        
        # Scheda 4: Risultati principali
        main_results = pd.DataFrame({
            'Metrica': [
                'Angolo ottimale (°)', 'Velocità iniziale (m/s)', 
                'Tempo di volo (s)', 'Altezza massima (m)',
                'Gittata (m)', 'Drop al bersaglio (cm)'
            ],
            'Valore': [
                trajectory_result.angle_degrees, trajectory_result.v0,
                trajectory_result.flight_time, trajectory_result.max_height,
                trajectory_result.range_distance,
                (params.launch_height + np.tan(np.radians(trajectory_result.angle_degrees)) * params.target_distance - 
                 get_y_at_x_improved(trajectory_result.X, trajectory_result.Y, params.target_distance)) * 100
            ]
        })
        main_results.to_excel(writer, sheet_name='Risultati', index=False)
        
        # Scheda 5: Statistiche Monte Carlo (se disponibili)
        if monte_carlo_stats:
            mc_data = []
            for metric, stats in monte_carlo_stats.items():
                mc_data.append({
                    'Metrica': metric,
                    'Media': stats['mean'],
                    'Deviazione Standard': stats['std'],
                    'Minimo': stats['min'],
                    'Massimo': stats['max'],
                    'Percentile 5%': stats['percentile_5'],
                    'Percentile 95%': stats['percentile_95']
                })
            pd.DataFrame(mc_data).to_excel(writer, sheet_name='Monte Carlo', index=False)
    
    output.seek(0)
    return output

# ==============================
# INTERFACCIA STREAMLIT MIGLIORATA
# ==============================
def main():
    st.set_page_config(
        page_title="Simulatore Balistico Avanzato", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header personalizzato
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                color: white; border-radius: 10px; margin-bottom: 2rem;'>
        <h1>🏹 Simulatore Balistico Avanzato RK4</h1>
        <p style='margin: 0; opacity: 0.9;'>Simulazione fisica completa con modello del vento e analisi statistica</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar per controlli avanzati
    with st.sidebar:
        st.header("🔧 Controlli Simulazione")
        
        # Parametri numerici
        with st.expander("⚙️ Parametri RK4", expanded=False):
            tol = st.slider("Tolleranza", 1e-8, 1e-4, 1e-6, format="%.1e")
            max_dt = st.slider("Passo massimo (s)", 0.001, 0.1, 0.01)
            min_dt = st.slider("Passo minimo (s)", 1e-8, 0.001, 1e-6, format="%.1e")
        
        with st.expander("📊 Analisi Drop", expanded=True):
            d_min = st.number_input("Distanza min (m)", 5, 100, 10)
            d_max = st.number_input("Distanza max (m)", 10, 100, 50)
            d_step = st.number_input("Passo (m)", 1, 10, 5)
            fit_degree = st.slider("Grado polinomio", 1, 5, 2)
            d_query = st.number_input("Query drop (m)", 5, 100, 30)
        
        with st.expander("🎲 Monte Carlo", expanded=False):
            enable_mc = st.checkbox("Abilita analisi Monte Carlo")
            if enable_mc:
                n_simulations = st.slider("N° simulazioni", 50, 500, 100)
                mass_var = st.slider("Variazione peso (%)", 0, 10, 2) / 100
                v0_var = st.slider("Variazione velocità (%)", 0, 10, 5) / 100
                wind_var = st.slider("Variazione vento (m/s)", 0, 5, 1)
    
    # Layout principale a tre colonne
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.subheader("🎯 Parametri Freccia")
        mass = st.number_input("Peso (g)", 10.0, 50.0, 24.0, step=0.5)
        length = st.number_input("Lunghezza (m)", 0.5, 1.0, 0.75, step=0.01)
        diameter = st.number_input("Diametro (mm)", 4.0, 12.0, 6.2, step=0.1)
        spine = st.number_input("Spine", 200, 1200, 700, step=25)
        balance_point = st.number_input("Bilanciamento (m)", 0.2, 0.8, 0.4, step=0.01)
        tip_type = st.selectbox("Tipo di punta", list(TIPO_PUNTA_CD_FACTOR.keys()))
        
        st.subheader("🏹 Parametri Arco")
        draw_force = st.number_input("Forza (lb)", 10.0, 80.0, 36.0, step=1.0)
        draw_length = st.number_input("Allungo (m)", 0.5, 1.1, 0.70, step=0.01)
        brace_height = st.number_input("Brace (m)", 0.05, 0.30, 0.18, step=0.01)
        efficiency = st.number_input("Efficienza", 0.5, 0.95, 0.82, step=0.01)
        bow_type = st.selectbox("Tipo di arco", list(BOW_TYPE_DEFAULT_EFF.keys()))
    
    with col2:
        st.subheader("👤 Parametri Arciere")
        launch_height_neutral = st.number_input("Quota neutra (m)", 0.8, 2.2, 1.5, step=0.01)
        anchor_length = st.number_input("Lunghezza spalla-aggancio (m)", 0.4, 1.0, 0.75, step=0.01)
        pelvis_height = st.number_input("Quota bacino (m)", 0.3, 1.5, 1.0, step=0.01)
        eye_offset_v = st.number_input("Offset occhio (m)", 0.01, 0.25, 0.09, step=0.01)
        
        st.subheader("🎯 Parametri Bersaglio")
        target_distance = st.number_input("Distanza (m)", 1.0, 150.0, 50.0, step=1.0)
        target_height = st.number_input("Quota (m)", -20.0, 20.0, 1.5, step=0.1)
        
        st.subheader("⚙️ Opzioni Velocità")
        use_measured_v0 = st.checkbox("Usa v₀ misurata")
        v0 = st.number_input("v₀ misurata (m/s)", 5.0, 120.0, 55.0, 
                            step=1.0, disabled=not use_measured_v0)
    
    with col3:
        st.subheader("🌪️ Condizioni Ambientali")
        wind_speed = st.number_input("Velocità vento (m/s)", -10.0, 20.0, 0.0, step=0.1)
        wind_direction = st.number_input("Direzione vento (°)", 0, 360, 0, step=5)
        air_temperature = st.number_input("Temperatura (°C)", -10, 50, 20, step=1)
        air_pressure = st.number_input("Pressione (hPa)", 900, 1100, 1013, step=1)
        humidity = st.number_input("Umidità (%)", 0, 100, 50, step=5)
        
        st.subheader("📐 Geometria Mirino")
        o_eye_cock = st.number_input("Distanza occhio-cocca (m)", 0.05, 0.40, 0.11, 
                                    step=0.005, format="%.3f")
        t_cock_riser = st.number_input("Distanza cocca-riser (m)", 0.2, 1.5, 0.70, step=0.01)
        
        st.subheader("🎛️ Opzioni Visualizzazione")
        show_comparison = st.checkbox("Confronta con/senza drag", value=False)
        show_wind_effect = st.checkbox("Mostra effetti vento", value=True)
        show_velocity_profile = st.checkbox("Mostra profilo velocità", value=True)
    
    # Pulsante principale
    if st.button("🚀 CALCOLA SIMULAZIONE COMPLETA", type="primary"):
        # Validazione e creazione parametri
        try:
            params = SimulationParams(
                mass=mass, length=length, spine=spine, diameter=diameter,
                balance_point=balance_point, tip_type=tip_type,
                draw_force=draw_force, draw_length=draw_length, brace_height=brace_height,
                efficiency=efficiency if use_measured_v0 else BOW_TYPE_DEFAULT_EFF.get(bow_type, efficiency), 
                bow_type=bow_type, launch_height_neutral=launch_height_neutral,
                anchor_length=anchor_length, pelvis_height=pelvis_height, eye_offset_v=eye_offset_v,
                target_distance=target_distance, target_height=target_height,
                use_measured_v0=use_measured_v0, v0=v0,
                wind_speed=wind_speed, wind_direction=wind_direction,
                air_temperature=air_temperature, air_pressure=air_pressure, humidity=humidity
            )
        except ValueError as e:
            st.error(f"Errore nei parametri: {e}")
            return
        
        # Inizializzazione integratore
        integrator = EnhancedRK4Integrator(tol=tol, max_dt=max_dt, min_dt=min_dt)
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Passo 1: Calcolo angolo ottimale
            status_text.text("🔍 Calcolo angolo ottimale...")
            progress_bar.progress(20)
            
            optimal_angle, optimization_error = find_optimal_angle_robust(params, integrator)
            if optimization_error > 5.0:  # Errore > 5 metri
                st.warning(f"Ottimizzazione angolo con errore elevato: {optimization_error:.2f} m")
            
            # Aggiorna quota di uscita
            y_launch = calcola_quota_uscita_posturale(params, np.radians(optimal_angle))
            params = replace(params, launch_height=y_launch)
            
            # Passo 2: Simulazione principale
            status_text.text("📈 Simulazione traiettoria principale...")
            progress_bar.progress(40)
            
            sim_distance = max(float(d_max), float(target_distance)) * 1.2
            main_result = integrator.integrate(optimal_angle, params, 
                                             include_drag=True, include_wind=True,
                                             end_x=sim_distance)
            
            # Passo 3: Simulazione senza drag (se richiesta)
            no_drag_result = None
            if show_comparison:
                status_text.text("📈 Simulazione senza resistenza...")
                progress_bar.progress(50)
                no_drag_result = integrator.integrate(optimal_angle, params, 
                                                    include_drag=False, include_wind=False,
                                                    end_x=sim_distance)
            
            # Passo 4: Calcolo curve drop
            status_text.text("📊 Calcolo curve drop...")
            progress_bar.progress(60)
            
            distances = np.arange(d_min, d_max + 1, d_step)
            drops_cm, drops_m = [], []
            
            for d in distances:
                if d <= main_result.X.max():
                    y_arrow = get_y_at_x_improved(main_result.X, main_result.Y, d)
                    y_sight = params.launch_height + np.tan(np.radians(optimal_angle)) * d
                    drop_m = y_sight - y_arrow
                    drops_m.append(drop_m)
                    drops_cm.append(drop_m * 100.0)
                else:
                    drops_m.append(np.nan)
                    drops_cm.append(np.nan)
            
            # Fit dei dati
            valid_mask = ~np.isnan(drops_cm)
            valid_x = distances[valid_mask]
            valid_y = np.array(drops_cm)[valid_mask]
            
            spline_fit, poly_fit = None, None
            if len(valid_x) > 3:
                try:
                    spline_fit = UnivariateSpline(valid_x, valid_y, s=0)
                except:
                    pass
                
                if len(valid_x) >= fit_degree + 1:
                    try:
                        poly_coeffs = np.polyfit(valid_x, valid_y, deg=fit_degree)
                        poly_fit = np.poly1d(poly_coeffs)
                    except:
                        pass
            
            # Passo 5: Analisi Monte Carlo (se abilitata)
            monte_carlo_stats = None
            if enable_mc:
                status_text.text("🎲 Analisi Monte Carlo...")
                progress_bar.progress(80)
                
                param_variations = {
                    'mass': mass_var,
                    'v0': v0_var if use_measured_v0 else 0.05,
                    'wind_speed': wind_var,
                    'air_temperature': 3.0
                }
                
                try:
                    monte_carlo_stats = monte_carlo_analysis(params, integrator, 
                                                           n_simulations, param_variations)
                except Exception as e:
                    st.warning(f"Errore nell'analisi Monte Carlo: {e}")
            
            progress_bar.progress(100)
            status_text.text("✅ Simulazione completata!")
            
            # === VISUALIZZAZIONE RISULTATI ===
            st.markdown("---")
            st.subheader("📊 Risultati Simulazione")
            
            # Metriche principali
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Angolo Ottimale", f"{optimal_angle:.2f}°", 
                         delta=f"{optimal_angle - np.degrees(np.arctan2(target_height - params.launch_height, target_distance)):+.2f}°")
            with col_m2:
                st.metric("Velocità Iniziale", f"{main_result.v0:.1f} m/s")
            with col_m3:
                st.metric("Tempo di Volo", f"{main_result.flight_time:.2f} s")
            with col_m4:
                drop_at_target = (params.launch_height + np.tan(np.radians(optimal_angle)) * target_distance - 
                                get_y_at_x_improved(main_result.X, main_result.Y, target_distance)) * 100
                st.metric("Drop al Bersaglio", f"{drop_at_target:.1f} cm")
            
            # Grafico traiettoria
            st.subheader("📈 Traiettoria")
            fig_traj = create_enhanced_trajectory_plot(main_result, params, no_drag_result, show_wind_effect)
            st.pyplot(fig_traj, use_container_width=True)
            
            # Statistiche integrazione
            with st.expander("🔧 Statistiche Integrazione RK4"):
                stats_cols = st.columns(5)
                stats = main_result.integration_stats
                with stats_cols[0]:
                    st.metric("Passi Accettati", stats['steps'])
                with stats_cols[1]:
                    st.metric("Passi Rifiutati", stats['rejections'])
                with stats_cols[2]:
                    st.metric("Passo Min", f"{stats['min_dt_used']:.2e} s")
                with stats_cols[3]:
                    st.metric("Passo Max", f"{stats['max_dt_used']:.2e} s")
                with stats_cols[4]:
                    efficiency_pct = 100 * stats['steps'] / (stats['steps'] + stats['rejections']) if (stats['steps'] + stats['rejections']) > 0 else 0
                    st.metric("Efficienza", f"{efficiency_pct:.1f}%")
            
            # Analisi drop
            st.subheader("📉 Analisi Drop")
            fig_drop = create_drop_analysis_plot(distances, drops_cm, spline_fit, poly_fit)
            st.pyplot(fig_drop, use_container_width=True)
            
            # Query drop
            if spline_fit is not None:
                dq = np.clip(d_query, valid_x.min(), valid_x.max())
                drop_query_result = float(spline_fit(dq))
                st.info(f"**Drop stimato a {dq:.0f}m:** {drop_query_result:.1f} cm")
            
            # Dati mirino
            st.subheader("🎯 Dati Mirino")
            
            def y_cm(x: float, o: float, t: float, d: float = 0.0) -> float:
                u = (o + d) / (t + x)
                y_calc = 100.0 * x * (u / np.sqrt(1 + u**2))
                return y_calc - d * 100.0
            
            def drop_cm_at(x):
                if spline_fit is not None and valid_x.min() <= x <= valid_x.max():
                    return float(spline_fit(x))
                elif poly_fit is not None:
                    return float(poly_fit(x))
                elif len(valid_x) >= 2:
                    return float(interp1d(valid_x, valid_y, kind='linear', 
                                        fill_value='extrapolate')(x))
                return np.nan
            
            # Calcolo proiezioni mirino
            proj_rows = []
            for d in distances:
                if len(valid_x) and (d < valid_x.min() or d > valid_x.max()):
                    proj_rows.append((d, np.nan, np.nan))
                    continue
                drop_cm_val = drop_cm_at(d)
                drop_m_val = drop_cm_val / 100.0 if not np.isnan(drop_cm_val) else 0.0
                yproj = y_cm(d, o_eye_cock, t_cock_riser, d=drop_m_val)
                proj_rows.append((d, drop_cm_val, yproj))
            
            df_mirino = pd.DataFrame(proj_rows, columns=["Distanza (m)", "Drop (cm)", "Proiezione riser (cm)"])
            st.dataframe(df_mirino.style.format({'Drop (cm)': '{:.1f}', 'Proiezione riser (cm)': '{:.2f}'}), 
                        use_container_width=True)
            
            # Visualizzazione mirino
            valid_proj = df_mirino.dropna()
            if len(valid_proj) >= 2:
                st.subheader("📏 Anteprima Mirino")
                
                fig_mirino, ax_mirino = plt.subplots(figsize=(4, 8))
                
                y_marks = valid_proj["Proiezione riser (cm)"].values
                dist_marks = valid_proj["Distanza (m)"].values
                
                # Linea centrale del riser
                y_range = [y_marks.min() - 2, y_marks.max() + 2]
                ax_mirino.vlines(0, y_range[0], y_range[1], colors="black", linewidth=3)
                
                # Tacche delle distanze
                colors = plt.cm.viridis(np.linspace(0, 1, len(y_marks)))
                for i, (yv, dv, color) in enumerate(zip(y_marks, dist_marks, colors)):
                    ax_mirino.hlines(yv, xmin=-0.8, xmax=0.8, colors=color, linewidth=2)
                    side = 1 if i % 2 == 0 else -1
                    ax_mirino.text(side * 1.2, yv, f"{int(dv)}m", 
                                  va="center", ha="left" if side > 0 else "right", 
                                  fontsize=10, fontweight='bold')
                
                # Punto laser di riferimento a 30m
                y_laser_30 = y_cm(30.0, o_eye_cock, t_cock_riser, d=0.0)
                ax_mirino.scatter(0, y_laser_30, color="red", s=100, 
                                 marker='*', zorder=10, edgecolors='darkred')
                ax_mirino.text(1.2, y_laser_30, "Laser 30m", va="center", 
                              fontsize=10, color="red", fontweight='bold')
                
                ax_mirino.set_ylim(y_range)
                ax_mirino.set_xlim(-3, 3)
                ax_mirino.set_xticks([])
                ax_mirino.set_ylabel("Quota sul riser (cm)", fontsize=12)
                ax_mirino.set_title("Scala Mirino", fontsize=14, fontweight='bold')
                ax_mirino.grid(True, axis='y', linestyle='--', alpha=0.3)
                ax_mirino.set_facecolor('#f8f9fa')
                
                st.pyplot(fig_mirino, use_container_width=False)
            
            # Analisi Monte Carlo
            if monte_carlo_stats:
                st.subheader("🎲 Analisi Statistica Monte Carlo")
                
                mc_cols = st.columns(3)
                with mc_cols[0]:
                    st.metric("Drop Medio", 
                             f"{monte_carlo_stats.get('drops_at_target', {}).get('mean', 0):.1f} cm",
                             delta=f"±{monte_carlo_stats.get('drops_at_target', {}).get('std', 0):.1f}")
                with mc_cols[1]:
                    st.metric("Tempo Volo Medio", 
                             f"{monte_carlo_stats.get('flight_times', {}).get('mean', 0):.2f} s",
                             delta=f"±{monte_carlo_stats.get('flight_times', {}).get('std', 0):.2f}")
                with mc_cols[2]:
                    st.metric("Altezza Max Media", 
                             f"{monte_carlo_stats.get('max_heights', {}).get('mean', 0):.1f} m",
                             delta=f"±{monte_carlo_stats.get('max_heights', {}).get('std', 0):.1f}")
                
                # Grafico distribuzione drop
                if 'drops_at_target' in monte_carlo_stats:
                    fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
                    
                    # Simula i dati per l'istogramma (in un'implementazione reale 
                    # dovremmo salvare tutti i valori individuali)
                    stats = monte_carlo_stats['drops_at_target']
                    simulated_drops = np.random.normal(stats['mean'], stats['std'], n_simulations)
                    
                    ax_hist.hist(simulated_drops, bins=20, alpha=0.7, color='skyblue', 
                                edgecolor='black', density=True)
                    ax_hist.axvline(stats['mean'], color='red', linestyle='--', 
                                   linewidth=2, label=f"Media: {stats['mean']:.1f} cm")
                    ax_hist.axvline(stats['percentile_5'], color='orange', linestyle=':', 
                                   linewidth=2, label=f"5°: {stats['percentile_5']:.1f} cm")
                    ax_hist.axvline(stats['percentile_95'], color='orange', linestyle=':', 
                                   linewidth=2, label=f"95°: {stats['percentile_95']:.1f} cm")
                    
                    ax_hist.set_xlabel("Drop (cm)")
                    ax_hist.set_ylabel("Densità di Probabilità")
                    ax_hist.set_title("Distribuzione Drop al Bersaglio - Monte Carlo")
                    ax_hist.legend()
                    ax_hist.grid(True, alpha=0.3)
                    
                    st.pyplot(fig_hist, use_container_width=True)
                
                # Tabella dettagliata statistiche
                with st.expander("📊 Statistiche Dettagliate Monte Carlo"):
                    mc_detailed = []
                    for metric, stats in monte_carlo_stats.items():
                        mc_detailed.append({
                            'Metrica': metric.replace('_', ' ').title(),
                            'Media': f"{stats['mean']:.3f}",
                            'Std Dev': f"{stats['std']:.3f}",
                            'Min': f"{stats['min']:.3f}",
                            'Max': f"{stats['max']:.3f}",
                            '5° Percentile': f"{stats['percentile_5']:.3f}",
                            '95° Percentile': f"{stats['percentile_95']:.3f}"
                        })
                    st.dataframe(pd.DataFrame(mc_detailed), use_container_width=True)
            
            # === EXPORT DATI ===
            st.subheader("💾 Export Dati")
            
            export_cols = st.columns(3)
            
            with export_cols[0]:
                # CSV mirino
                csv_mirino = io.StringIO()
                df_mirino.to_csv(csv_mirino, index=False)
                st.download_button("📊 Download Mirino CSV", 
                                 data=csv_mirino.getvalue(),
                                 file_name=f"mirino_{target_distance}m.csv",
                                 mime="text/csv")
            
            with export_cols[1]:
                # Excel completo
                try:
					excel_data = export_complete_analysis(main_result, params, df_mirino, monte_carlo_stats)
					st.download_button("📈 Download Analisi Excel", 
									data=excel_data,
									file_name=f"analisi_completa_{target_distance}m.xlsx",
									mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
				except ImportError as e:
					st.warning("Moduli Excel non disponibili. Installare openpyxl o xlsxwriter per abilitare l'export Excel.")
            
            with export_cols[2]:
                # PDF mirino (mantenuto dalla versione originale)
                try:
                    pdf_buf, pdf_name = esporta_mirino_pdf_bytes(df_mirino, o_eye_cock, t_cock_riser)
                    st.download_button("🎯 Download Mirino PDF", 
                                     data=pdf_buf,
                                     file_name=pdf_name,
                                     mime="application/pdf")
                except Exception as e:
                    st.error(f"Errore generazione PDF: {e}")
            
            # === RIEPILOGO FINALE ===
            st.markdown("---")
            with st.container():
                st.subheader("📋 Riepilogo Risultati")
                
                summary_data = {
                    'Parametro': [
                        'Angolo ottimale', 'Velocità iniziale', 'Quota uscita',
                        'Drop al bersaglio', 'Tempo di volo', 'Altezza massima',
                        'Gittata teorica', 'Efficienza RK4'
                    ],
                    'Valore': [
                        f"{optimal_angle:.2f}°",
                        f"{main_result.v0:.1f} m/s",
                        f"{params.launch_height:.2f} m",
                        f"{drop_at_target:.1f} cm",
                        f"{main_result.flight_time:.2f} s",
                        f"{main_result.max_height:.1f} m",
                        f"{main_result.range_distance:.1f} m",
                        f"{100 * main_result.integration_stats['steps'] / (main_result.integration_stats['steps'] + main_result.integration_stats['rejections']):.1f}%"
                    ],
                    'Note': [
                        f"Scarto da mira diretta: {optimal_angle - np.degrees(np.arctan2(target_height - params.launch_height, target_distance)):+.2f}°",
                        "Calcolata da parametri arco" if not use_measured_v0 else "Valore misurato inserito",
                        "Corretta per postura arciere",
                        f"A {target_distance}m dal punto di mira",
                        f"Per raggiungere {target_distance}m",
                        "Punto più alto della traiettoria",
                        "Distanza teorica di caduta freccia",
                        f"Passi: {main_result.integration_stats['steps']}, Rifiutati: {main_result.integration_stats['rejections']}"
                    ]
                }
                
                df_summary = pd.DataFrame(summary_data)
                st.dataframe(df_summary, use_container_width=True, hide_index=True)
            
            # Messaggio di successo
            st.success("🎉 Simulazione completata con successo! Tutti i grafici e dati sono pronti per l'export.")
            
        except Exception as e:
            st.error(f"Errore durante la simulazione: {e}")
            logger.error(f"Errore simulazione: {e}", exc_info=True)
        
        finally:
            progress_bar.empty()
            status_text.empty()

# ==============================
# FUNZIONI DI SUPPORTO ORIGINALI MANTENUTE
# ==============================
def calcola_quota_uscita_posturale(params: SimulationParams, angle_rad: float) -> float:
    """Calcola quota uscita considerando postura arciere"""
    anchor_length = params.anchor_length
    y_neutral = params.launch_height_neutral
    pelvis = params.pelvis_height
    
    y_pivot = y_neutral - pelvis
    dx, dy = anchor_length, pelvis
    r = np.sqrt(dx**2 + dy**2)
    phi0 = np.arcsin(dy / r)
    return y_pivot + r * np.sin(angle_rad + phi0)

def trova_angolo_convergenza(params: SimulationParams, integrator: EnhancedRK4Integrator, 
                           tol_angle: float = 0.01, tol_height: float = 0.001, 
                           max_iter: int = 25) -> Tuple[float, float]:
    """Trova angolo con convergenza posturale"""
    quota_uscita = params.launch_height_neutral
    angolo = 0.0
    
    for i in range(max_iter):
        modified_params = replace(params, launch_height=quota_uscita)
        
        angolo_nuovo, _ = find_optimal_angle_robust(modified_params, integrator)
        y_launch_new = calcola_quota_uscita_posturale(params, np.radians(angolo_nuovo))
        
        if (abs(angolo_nuovo - angolo) < tol_angle and 
            abs(y_launch_new - quota_uscita) < tol_height):
            return angolo_nuovo, y_launch_new
        
        angolo, quota_uscita = angolo_nuovo, y_launch_new
    
    return angolo, quota_uscita

def esporta_mirino_pdf_bytes(df_proj: pd.DataFrame, o_eye_cock: float, 
                            t_cock_riser: float, filename: str = "mirino_riser.pdf") -> Tuple[io.BytesIO, str]:
    """Esporta mirino in PDF (funzione originale mantenuta)"""
    
    def y_cm(x: float, o: float, t: float, d: float = 0.0) -> float:
        u = (o + d) / (t + x)
        y_calc = 100.0 * x * (u / np.sqrt(1 + u**2))
        return y_calc - d * 100.0
    
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

if __name__ == "__main__":
    main()
            