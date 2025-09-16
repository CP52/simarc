# simarc_mira_rk4_enhanced.py
# Simulatore Balistico Avanzato con RK4 Adattativo e Generatore Mirino Completo
# Versione migliorata con analisi scientifica aggiornata e interfaccia ottimizzata

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
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
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
# CONFIGURAZIONE E COSTANTI FISICHE
# ==============================
class PhysicalConstants:
    """Costanti fisiche aggiornate secondo letteratura scientifica"""
    G = 9.80665  # Accelerazione gravitazionale standard [m/s²]
    AIR_DENSITY_STP = 1.225  # Densità dell'aria a STP [kg/m³]
    AIR_VISCOSITY = 1.81e-5  # Viscosità dinamica dell'aria [Pa·s]
    R_SPECIFIC_AIR = 287.058  # Costante specifica dell'aria secca [J/(kg·K)]

# Configurazione grafica moderna
PLOT_CONFIG = {
    'colors': {
        'primary': '#2E86AB',      # Blu professionale
        'secondary': '#A23B72',     # Viola scuro
        'accent': '#F18F01',        # Arancione brillante
        'danger': '#C73E1D',        # Rosso
        'success': '#588157',       # Verde
        'info': '#4EA5D9',         # Azzurro
        'warning': '#F4A261'        # Giallo-arancio
    },
    'figsize': (14, 8),
    'dpi': 100,
    'font_size': 12
}

# Coefficienti aerodinamici basati su ricerca moderna (Litz, 2017; McCoy, 2012)
TIPO_PUNTA_CD_FACTOR = {
    "Slanciata (field/bullet)": 0.92,    # Ottimizzata per penetrazione
    "Standard": 1.00,                     # Riferimento
    "Broadhead (larga)": 1.18,           # Aumentata resistenza per lame
    "Judo Point": 1.32,                  # Punti di pratica con resistenza elevata
    "Bodkin": 0.88,                      # Punta medievale ottimizzata
    "Blunt": 1.45                        # Punta piatta per small game
}

BOW_TYPE_DEFAULT_EFF = {
    "longbow": 0.73,           # Efficienza tradizionale
    "ricurvo": 0.84,          # Migliorata geometria
    "compound": 0.89,         # Sistemi cam moderni
    "takedown": 0.81,         # Ricurvo smontabile
    "barebow": 0.79,          # Senza accessori
    "hunting_recurve": 0.86   # Ricurvo da caccia ottimizzato
}

# ==============================
# CLASSI DATI CON VALIDAZIONE AVANZATA
# ==============================
@dataclass
class SimulationParams:
    """Parametri simulazione con validazione scientifica rigorosa"""
    # Parametri freccia
    mass: float                    # [g]
    length: float                  # [m]  
    spine: int                     # [lb/in²]
    diameter: float               # [mm]
    balance_point: float          # [m] dal nock
    tip_type: str
    
    # Parametri arco
    draw_force: float             # [lb]
    draw_length: float            # [m]
    brace_height: float           # [m]
    efficiency: float             # [0-1]
    bow_type: str
    
    # Parametri arciere
    launch_height_neutral: float  # [m]
    anchor_length: float          # [m]
    pelvis_height: float          # [m]
    eye_offset_v: float           # [m]
    
    # Parametri bersaglio
    target_distance: float        # [m]
    target_height: float          # [m]
    
    # Velocità
    use_measured_v0: bool
    v0: float                     # [m/s]
    
    # Parametri ambientali avanzati
    wind_speed: float = 0.0       # [m/s] positivo=favorevole
    air_temperature: float = 15.0  # [°C] Standard ISA
    air_pressure: float = 1013.25  # [hPa] Pressione standard
    humidity: float = 50.0         # [%] Umidità relativa
    altitude: float = 0.0          # [m] Quota sul livello del mare
    
    # Parametri calcolati
    launch_height: Optional[float] = None

    def __post_init__(self):
        self.validate()
        if self.launch_height is None:
            self.launch_height = self.launch_height_neutral
        
        # Correzioni per altitudine
        self._apply_altitude_corrections()
    
    def _apply_altitude_corrections(self):
        """Applica correzioni per altitudine secondo modello atmosferico standard"""
        if self.altitude > 0:
            # Correzione pressione barometrica
            self.air_pressure *= (1 - 0.0065 * self.altitude / 288.15) ** 5.255
            # Correzione temperatura (gradiente troposferico)
            self.air_temperature -= 0.0065 * self.altitude
    
    def validate(self):
        """Validazione parametri con limiti fisici realistici"""
        validations = [
            (5.0 <= self.mass <= 100.0, "Peso freccia deve essere 5-100g"),
            (0.50 <= self.length <= 1.50, "Lunghezza freccia 0.5-1.5m"),
            (3.0 <= self.diameter <= 15.0, "Diametro 3-15mm"),
            (0.0 <= self.balance_point <= self.length, "Punto bilanciamento non valido"),
            (15.0 <= self.draw_force <= 150.0, "Forza arco 15-150lb"),
            (self.draw_length > self.brace_height + 0.05, "Allungo insufficiente"),
            (0.50 <= self.efficiency <= 0.95, "Efficienza 50-95%"),
            (self.target_distance > 1.0, "Distanza minima 1m"),
            (-25 <= self.wind_speed <= 25, "Vento -25/+25 m/s"),
            (-30 <= self.air_temperature <= 60, "Temperatura -30/+60°C"),
            (500 <= self.air_pressure <= 1100, "Pressione 500-1100 hPa"),
            (0 <= self.humidity <= 100, "Umidità 0-100%"),
            (0 <= self.altitude <= 5000, "Altitudine max 5000m")
        ]
        
        for condition, message in validations:
            if not condition:
                raise ValueError(f"Parametro non valido: {message}")

@dataclass
class TrajectoryResults:
    """Risultati simulazione con metadati completi"""
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
    energy_loss: float = 0.0  # Perdita energia [J]
    stability_factor: float = 1.0  # Fattore stabilità

# ==============================
# MODELLO AERODINAMICO AVANZATO
# ==============================
#@lru_cache(maxsize=2000)
def calculate_air_density(temp_c: float, pressure_hpa: float, humidity_pct: float) -> float:
    """Calcola densità aria con correzione umidità (ASHRAE, 2017)"""
    temp_k = temp_c + 273.15
    
    # Pressione vapore saturo (Magnus formula)
    p_sat = 610.78 * np.exp(17.2694 * temp_c / (temp_c + 238.3))
    p_vapor = humidity_pct / 100.0 * p_sat / 100.0  # Converti in hPa
    p_dry = pressure_hpa - p_vapor
    
    # Densità aria secca + vapor d'acqua
    rho_dry = p_dry * 100 / (PhysicalConstants.R_SPECIFIC_AIR * temp_k)
    rho_vapor = p_vapor * 100 / (461.5 * temp_k)  # R_vapor = 461.5 J/(kg·K)
    
    return rho_dry + rho_vapor

#@lru_cache(maxsize=1500)
def reynolds_number_enhanced(v: float, diameter_mm: float, params: SimulationParams) -> float:
    """Numero Reynolds con proprietà aria variabili"""
    rho = calculate_air_density(params.air_temperature, params.air_pressure, params.humidity)
    mu = PhysicalConstants.AIR_VISCOSITY * (params.air_temperature + 273.15) / 291.15  # Sutherland law
    d_m = diameter_mm / 1000.0
    
    return rho * v * d_m / mu

def enhanced_drag_coefficient(v: float, diameter_mm: float, angle_of_attack_deg: float, 
                            tip_type: str, params: SimulationParams) -> float:
    """Modello Cd avanzato basato su CFD e dati sperimentali (Litz, 2017)"""
    Re = reynolds_number_enhanced(v, diameter_mm, params)
    
    # Transizioni Reynolds più accurate per cilindri
    if Re < 1e3:
        Cd_base = 2.2
    elif Re < 2e3:
        Cd_base = 2.2 - 0.8 * (Re - 1e3) / 1e3
    elif Re < 1e4:
        Cd_base = 1.4 - 0.15 * (Re - 2e3) / 8e3
    elif Re < 3e4:
        Cd_base = 1.25 + 0.95 * (Re - 1e4) / 2e4
    elif Re < 1e5:
        Cd_base = 2.2 - 0.8 * (Re - 3e4) / 7e4
    elif Re < 3e5:
        Cd_base = 1.4 - 0.2 * (Re - 1e5) / 2e5
    else:
        Cd_base = 1.2
    
    # Effetto angolo attacco (modello quadratico migliorato)
    gamma_rad = np.radians(abs(angle_of_attack_deg))
    angle_factor = 1 + 2.5 * gamma_rad**1.5 + 0.8 * gamma_rad**3
    
    # Fattore forma punta (basato su dati sperimentali)
    tip_factor = TIPO_PUNTA_CD_FACTOR.get(tip_type, 1.0)
    
    # Correzione densità (effetto comprimibilità minore)
    density_factor = 1 + 0.003 * (calculate_air_density(params.air_temperature, 
                                                        params.air_pressure, 
                                                        params.humidity) / 
                                   PhysicalConstants.AIR_DENSITY_STP - 1)
    
    return Cd_base * angle_factor * tip_factor * density_factor

def calculate_velocity_enhanced(params: SimulationParams) -> float:
    """Calcolo velocità con modello energetico avanzato"""
    if params.use_measured_v0:
        return params.v0
    
    mass_kg = params.mass / 1000.0
    F_N = params.draw_force * 4.44822  # lb -> N
    draw_m = max(0.001, params.draw_length - params.brace_height)
    
    # Energia potenziale arco (modello parabolico più realistico)
    E_stored = params.efficiency * 0.5 * F_N * draw_m
    
    # Correzioni ambientali
    air_density_ratio = (calculate_air_density(params.air_temperature, 
                                              params.air_pressure, 
                                              params.humidity) / 
                        PhysicalConstants.AIR_DENSITY_STP)
    
    # Correzione resistenza iniziale
    initial_drag_correction = 1 - 0.015 * (air_density_ratio - 1)
    
    v0 = np.sqrt(max(1.0, 2 * E_stored / mass_kg)) * initial_drag_correction
    return min(v0, 150.0)  # Limite fisico realistico

# ==============================
# INTEGRATORE RK4 CON MODELLO VENTO 3D SEMPLIFICATO
# ==============================
class AdvancedRK4Integrator:
    """Integratore RK4 con controllo adattivo e modello fisico completo"""
    
    def __init__(self, tol: float = 1e-6, max_dt: float = 0.02, min_dt: float = 1e-8,
                 safety_factor: float = 0.9, max_steps: int = 50000):
        self.tol = tol
        self.max_dt = max_dt
        self.min_dt = min_dt
        self.safety_factor = safety_factor
        self.max_steps = max_steps
        self.stats = self._init_stats()
    
    def _init_stats(self) -> Dict[str, Any]:
        return {
            'steps': 0, 'rejections': 0, 'min_dt_used': float('inf'),
            'max_dt_used': 0.0, 'total_time': 0.0, 'energy_loss': 0.0
        }
    
    def wind_force_model(self, vx: float, vy: float, params: SimulationParams) -> Tuple[float, float]:
        """Modello vento semplificato 2D con profilo realistico"""
        if abs(params.wind_speed) < 0.05:
            return 0.0, 0.0
        
        # Vento solo orizzontale (semplificazione 2D)
        wind_x = params.wind_speed
        wind_y = 0.0
        
        # Velocità relativa
        vrel_x = vx - wind_x
        vrel_y = vy - wind_y
        vrel = np.sqrt(vrel_x**2 + vrel_y**2)
        
        if vrel < 1e-6:
            return 0.0, 0.0
        
        # Area frontale freccia
        A = np.pi * (params.diameter / 1000.0 / 2.0) ** 2
        rho = calculate_air_density(params.air_temperature, params.air_pressure, params.humidity)
        
        # Coefficiente resistenza vento (diverso da quello balistico)
        Cd_wind = 1.1  # Leggermente maggiore per effetto crossflow
        
        # Forza vento
        F_wind = 0.5 * rho * Cd_wind * A * vrel**2
        
        # Solo componente orizzontale
        F_wind_x = -F_wind * (vrel_x / vrel)
        F_wind_y = 0.0
        
        return F_wind_x, F_wind_y
    
    def system_derivatives(self, t: float, state: np.ndarray, params: SimulationParams, 
                          angle_deg: float, include_drag: bool = True, 
                          include_wind: bool = True) -> np.ndarray:
        """Sistema equazioni differenziali completo"""
        x, y, vx, vy = state
        mass_kg = params.mass / 1000.0
        
        # Forze aerodinamiche
        F_drag_x = F_drag_y = 0.0
        if include_drag:
            v_total = np.sqrt(vx**2 + vy**2)
            if v_total > 0.1:
                # Angolo attacco rispetto traiettoria di volo
                flight_angle = np.degrees(np.arctan2(vy, vx))
                angle_of_attack = flight_angle - angle_deg
                
                Cd = enhanced_drag_coefficient(v_total, params.diameter, 
                                             angle_of_attack, params.tip_type, params)
                
                A = np.pi * (params.diameter / 1000.0 / 2.0) ** 2
                rho = calculate_air_density(params.air_temperature, 
                                          params.air_pressure, params.humidity)
                
                F_drag_total = 0.5 * rho * Cd * A * v_total**2
                F_drag_x = -F_drag_total * (vx / v_total)
                F_drag_y = -F_drag_total * (vy / v_total)
        
        # Forze vento
        F_wind_x = F_wind_y = 0.0
        if include_wind:
            F_wind_x, F_wind_y = self.wind_force_model(vx, vy, params)
        
        # Accelerazioni
        ax = (F_drag_x + F_wind_x) / mass_kg
        ay = -PhysicalConstants.G + (F_drag_y + F_wind_y) / mass_kg
        
        return np.array([vx, vy, ax, ay])
    
    def rk4_step(self, t: float, state: np.ndarray, dt: float, params: SimulationParams,
                angle_deg: float, include_drag: bool = True, include_wind: bool = True) -> np.ndarray:
        """Singolo passo RK4"""
        k1 = self.system_derivatives(t, state, params, angle_deg, include_drag, include_wind)
        k2 = self.system_derivatives(t + dt/2, state + dt/2 * k1, params, angle_deg, 
                                    include_drag, include_wind)
        k3 = self.system_derivatives(t + dt/2, state + dt/2 * k2, params, angle_deg,
                                    include_drag, include_wind)
        k4 = self.system_derivatives(t + dt, state + dt * k3, params, angle_deg,
                                    include_drag, include_wind)
        
        return state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    def adaptive_step_control(self, error: float, dt: float) -> float:
        """Controllo adattivo step size ottimizzato"""
        if error < self.tol / 100:
            return min(self.max_dt, dt * 2.0)
        elif error < self.tol:
            factor = self.safety_factor * (self.tol / max(error, 1e-16)) ** 0.2
            return min(self.max_dt, dt * min(factor, 1.5))
        else:
            factor = self.safety_factor * (self.tol / error) ** 0.25
            return max(self.min_dt, dt * max(factor, 0.2))
    
    def integrate_trajectory(self, angle_deg: float, params: SimulationParams,
                           include_drag: bool = True, include_wind: bool = True,
                           max_range: Optional[float] = None) -> TrajectoryResults:
        """Integrazione completa traiettoria"""
        
        angle_rad = np.radians(angle_deg)
        v0 = calculate_velocity_enhanced(params)
        max_range = max_range or params.target_distance * 1.5
        
        # Stato iniziale [x, y, vx, vy]
        state = np.array([0.0, params.launch_height,
                         v0 * np.cos(angle_rad), v0 * np.sin(angle_rad)])
        
        # Storage arrays
        trajectory_data = {
            'X': [state[0]], 'Y': [state[1]], 
            'VX': [state[2]], 'VY': [state[3]], 'T': [0.0]
        }
        
        t = 0.0
        dt = self.max_dt / 5
        max_height = params.launch_height
        initial_energy = 0.5 * params.mass/1000.0 * v0**2
        
        # Reset stats
        self.stats = self._init_stats()
        
        # Ciclo integrazione principale
        while (len(trajectory_data['X']) < self.max_steps and
               state[0] <= max_range and state[1] >= -50.0 and
               t < 30.0):
            
            # Calcolo con step doppio
            state_full = self.rk4_step(t, state, dt, params, angle_deg, 
                                      include_drag, include_wind)
            
            # Calcolo con due half-step
            state_half1 = self.rk4_step(t, state, dt/2, params, angle_deg,
                                       include_drag, include_wind)
            state_half2 = self.rk4_step(t + dt/2, state_half1, dt/2, params,
                                       angle_deg, include_drag, include_wind)
            
            # Stima errore locale
            error_est = np.linalg.norm(state_full - state_half2) / (np.linalg.norm(state) + 1e-10)
            
            if error_est < self.tol or dt <= self.min_dt:
                # Accetta step
                state = state_half2  # Usa soluzione più accurata
                t += dt
                
                trajectory_data['X'].append(state[0])
                trajectory_data['Y'].append(state[1])
                trajectory_data['VX'].append(state[2])
                trajectory_data['VY'].append(state[3])
                trajectory_data['T'].append(t)
                
                max_height = max(max_height, state[1])
                
                self.stats['steps'] += 1
                self.stats['min_dt_used'] = min(self.stats['min_dt_used'], dt)
                self.stats['max_dt_used'] = max(self.stats['max_dt_used'], dt)
                
                dt = self.adaptive_step_control(error_est, dt)
            else:
                # Rifiuta step
                dt = self.adaptive_step_control(error_est, dt)
                self.stats['rejections'] += 1
            
            # Controlli sicurezza
            if state[0] > max_range * 1.2 or state[1] < -100:
                break
        
        self.stats['total_time'] = t
        
        # Calcola perdita energia
        final_energy = 0.5 * params.mass/1000.0 * (state[2]**2 + state[3]**2)
        energy_loss = max(0, initial_energy - final_energy)
        self.stats['energy_loss'] = energy_loss
        
        return TrajectoryResults(
            X=np.array(trajectory_data['X']),
            Y=np.array(trajectory_data['Y']),
            V_x=np.array(trajectory_data['VX']),
            V_y=np.array(trajectory_data['VY']),
            times=np.array(trajectory_data['T']),
            v0=v0,
            flight_time=t,
            max_height=max_height,
            range_distance=trajectory_data['X'][-1] if trajectory_data['X'] else 0.0,
            angle_degrees=angle_deg,
            integration_stats=self.stats.copy(),
            energy_loss=energy_loss
        )

# ==============================
# FUNZIONI MATEMATICHE DI SUPPORTO
# ==============================
def interpolate_trajectory_point(X: np.ndarray, Y: np.ndarray, x_target: float,
                                method: str = 'cubic', extrapolate: bool = False) -> float:
    """Interpolazione robusta punti traiettoria"""
    if len(X) < 2:
        return 0.0
    
    # Ordina dati per X crescente
    sorted_indices = np.argsort(X)
    X_sorted = X[sorted_indices]
    Y_sorted = Y[sorted_indices]
    
    # Gestione extrapolazione
    if not extrapolate:
        x_target = np.clip(x_target, X_sorted.min(), X_sorted.max())
    
    try:
        if method == 'cubic' and len(X_sorted) >= 4:
            interp_func = interp1d(X_sorted, Y_sorted, kind='cubic',
                                  bounds_error=not extrapolate, fill_value='extrapolate')
        else:
            interp_func = interp1d(X_sorted, Y_sorted, kind='linear',
                                  bounds_error=not extrapolate, fill_value='extrapolate')
        return float(interp_func(x_target))
    
    except Exception as e:
        logger.warning(f"Errore interpolazione: {e}")
        # Fallback interpolazione lineare semplice
        return float(np.interp(x_target, X_sorted, Y_sorted))

def find_optimal_firing_angle(params: SimulationParams, integrator: AdvancedRK4Integrator,
                             method: str = 'brent') -> Tuple[float, float]:
    """Ricerca angolo ottimale con algoritmi robusti"""
    target_height= params.target_height
    target_distance = params.target_distance

    def objective_function(angle_deg):
        """Funzione obiettivo per ottimizzazione"""
        try:
            result = integrator.integrate_trajectory(angle_deg, params,
                                                   include_drag=True, include_wind=True,
                                                   max_range=target_distance * 1.2)
            if len(result.X) < 3:
                return 1e6
            
            y_at_target = interpolate_trajectory_point(result.X, result.Y, target_distance)
            error = abs(y_at_target - params.target_height)
            
            return error
        except Exception as e:
            logger.warning(f"Errore calcolo traiettoria angolo {angle_deg:.2f}°: {e}")
            return 1e6

    # Stima iniziale fisica
    v0_est = calculate_velocity_enhanced(params)
    h_diff = params.target_height - params.launch_height
    
    # Formula balistica approssimata con correzione drag
    try:
        discriminant = v0_est**4 - PhysicalConstants.G * (PhysicalConstants.G * target_distance**2 + 
                                                         2 * h_diff * v0_est**2)
        if discriminant > 0:
            angle_est = np.degrees(0.5 * np.arcsin(PhysicalConstants.G * target_distance / v0_est**2))
        else:
            angle_est = 20.0
    except:
        angle_est = 15.0
    
    # Range ricerca adattivo
    search_range = max(15.0, abs(angle_est) * 1.5)
    bounds = (max(-45, angle_est - search_range), min(75, angle_est + search_range))
    
    try:
        # Ottimizzazione principale
        if method == 'brent':
            from scipy.optimize import minimize_scalar
            result = minimize_scalar(objective_function, bounds=bounds, method='bounded',
                                   options={'xatol': 0.01})
        else:
            # Golden section search alternativo
            result = minimize_scalar(objective_function, bounds=bounds, method='golden')
        
        if result.success and result.fun < 2.0:  # Errore < 2m accettabile
            return result.x, result.fun
        else:
            raise ValueError("Ottimizzazione non convergente")
            
    except Exception as e:
        logger.warning(f"Ottimizzazione fallita: {e}. Uso ricerca griglia.")
        
        # Fallback: ricerca griglia fine
        angles_test = np.linspace(bounds[0], bounds[1], 100)
        errors = [objective_function(a) for a in angles_test]
        best_idx = np.argmin(errors)
        
        return angles_test[best_idx], errors[best_idx]

# ==============================
# SISTEMA GENERAZIONE MIRINO AVANZATO
# ==============================

class SightScalePDFGenerator:
    """Generatore PDF scala mirino con layout professionale"""
    
    def __init__(self, page_size=A4):
        self.page_size = page_size
        self.width, self.height = page_size
        
    def create_sight_scale_pdf(self, sight_data: pd.DataFrame, 
                              sight_calculator: SightingSystemCalculator,
                              params: SimulationParams) -> io.BytesIO:
        """Crea PDF scala mirino completo"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=self.page_size,
                              rightMargin=2*cm, leftMargin=2*cm,
                              topMargin=2*cm, bottomMargin=2*cm)
        
        # Stili documento
        styles = getSampleStyleSheet()
        title_style = styles['Title']
        heading_style = styles['Heading2']
        normal_style = styles['Normal']
        
        # Contenuto documento
        story = []
        
        # Titolo
        title = Paragraph("SCALA MIRINO BALISTICA", title_style)
        story.append(title)
        story.append(Spacer(1, 0.5*cm))
        
        # Informazioni configurazione
        config_data = [
            ['Parametro', 'Valore'],
            ['Peso freccia', f"{params.mass}g"],
            ['Velocità iniziale', f"{calculate_velocity_enhanced(params):.1f} m/s"],
            ['Tipo arco', params.bow_type.title()],
            ['Forza arco', f"{params.draw_force} lb"],
            ['Distanza occhio-cocca', f"{sight_calculator.eye_to_nock*100:.1f} cm"],
            ['Distanza cocca-riser', f"{sight_calculator.nock_to_riser*100:.1f} cm"],
            ['Condizioni vento', f"{params.wind_speed:+.1f} m/s" if abs(params.wind_speed) > 0.1 else "Assente"],
            ['Temperatura', f"{params.air_temperature:.0f}°C"],
            ['Umidità', f"{params.humidity:.0f}%"]
        ]
        
        config_table = Table(config_data, colWidths=[6*cm, 4*cm])
        config_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(config_table)
        story.append(Spacer(1, 1*cm))
        
        # Tabella scala mirino
        heading = Paragraph("Scala Distanze", heading_style)
        story.append(heading)
        story.append(Spacer(1, 0.3*cm))
        
        # Dati tabella mirino
        sight_table_data = [['Distanza (m)', 'Drop (cm)', 'Posizione Riser (cm)']]
        
        for _, row in sight_data.iterrows():
            sight_table_data.append([
                str(int(row['Distanza (m)'])),
                f"{row['Drop (cm)']:.1f}",
                f"{row['Proiezione riser (cm)']:.2f}"
            ])
        
        sight_table = Table(sight_table_data, colWidths=[3*cm, 3*cm, 4*cm])
        sight_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.lightgrey, colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
        ]))
        
        story.append(sight_table)
        story.append(Spacer(1, 1*cm))
        
        
        notes_text = """
        <b>Note Tecniche:</b><br/>
        • I valori sono calcolati per le condizioni ambientali specificate<br/>
        • La scala è ottimizzata per il setup geometrico dell'arciere<br/>
        • Verificare periodicamente la taratura con tiri di prova<br/>
        • Valori positivi = sopra il punto zero, negativi = sotto<br/>
        """
        
        notes = Paragraph(notes_text, normal_style)
        story.append(notes)
        
        # Costruisci PDF
        doc.build(story)
        buffer.seek(0)
        
        return buffer


def esporta_mirino_pdf_bytes(df_proj: pd.DataFrame, o_eye_cock: float, 
                            t_cock_riser: float, filename: str = "mirino_riser.pdf") -> Tuple[io.BytesIO, str]:
    """Esporta mirino in PDF grafico con tacche e laser a 30m"""
    
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

    # Colonna riser
    c.setLineWidth(2)
    c.line(x_center, y0_pt + cm2pt(y_min), x_center, y0_pt + cm2pt(y_max))

    # Tacche
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

    # Linea zero
    y_zero_pt = y0_pt + cm2pt(0.0)
    c.setStrokeColorRGB(1, 0, 0)
    c.line(x_center - 25, y_zero_pt, x_center + 25, y_zero_pt)
    c.setFont("Helvetica-Bold", 8)
    c.drawString(x_center + 30, y_zero_pt - 3, "0 cm (base)")
    c.setStrokeColorRGB(0, 0, 0)

    # Punto laser a 30m
    y_laser_pt = y0_pt + cm2pt(y_cm(30.0, o_eye_cock, t_cock_riser, 0.0))
    c.setFillColorRGB(0, 1, 0)
    c.circle(x_center, y_laser_pt, 2.5, fill=1, stroke=0)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(x_center + 30, y_laser_pt - 4, "Laser 30 m")

    # Barre di controllo 5 cm
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
# VISUALIZZAZIONE GRAFICA AVANZATA
# ==============================
def create_comprehensive_trajectory_plot(main_result: TrajectoryResults,
                                        params: SimulationParams,
                                        no_drag_result: Optional[TrajectoryResults] = None,
                                        show_wind: bool = True, target_distance: Optional[float] = None) -> plt.Figure:
    """Grafico traiettoria completo con analisi dettagliata"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # Layout subplot complesso
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.2)
    ax_traj = fig.add_subplot(gs[0, :])
    ax_vel = fig.add_subplot(gs[1, 0])
    ax_energy = fig.add_subplot(gs[1, 1])
    ax_drop = fig.add_subplot(gs[2, :])
    
    # === GRAFICO PRINCIPALE TRAIETTORIA ===
    X1, Y1 = main_result.X, main_result.Y
    
    # Traiettoria realistica
    ax_traj.plot(X1, Y1, color=PLOT_CONFIG['colors']['primary'], 
                linewidth=3, label="Traiettoria realistica", alpha=0.9)
    
    # Traiettoria senza drag (se presente)
    if no_drag_result is not None:
        ax_traj.plot(no_drag_result.X, no_drag_result.Y,
                    color=PLOT_CONFIG['colors']['secondary'],
                    linestyle='--', linewidth=2, 
                    label="Traiettoria ideale (senza drag)", alpha=0.7)
    
    # Bersaglio
    ax_traj.scatter(params.target_distance, params.target_height,
                   color=PLOT_CONFIG['colors']['accent'], s=200,
                   marker='o', label="Bersaglio", 
                   edgecolors='black', linewidth=2, zorder=10)
    
    # ===== CORREZIONE PRINCIPALE: CALCOLO LIMITI CONSIDERANDO LINEA DI MIRA =====
    
    # Calcola punto di impatto teorico della linea di mira (dove y=0 o al bersaglio)
    y0 = params.launch_height
    angle_rad = np.radians(main_result.angle_degrees)
    
    # Estensione linea di mira: calcola dove interseca y=0 o usa distanza massima ragionevole
    if angle_rad != 0:
        # Punto dove la linea di mira interseca y=0 (suolo)
        x_sight_ground = -y0 / np.tan(angle_rad) if np.tan(angle_rad) != 0 else params.target_distance * 2
        # Assicurati che sia ragionevole
        x_sight_max = max(params.target_distance * 1.3, min(abs(x_sight_ground), params.target_distance * 2))
    else:
        x_sight_max = params.target_distance * 1.3
    
    # Determina limite X considerando tutti gli elementi
    x_elements = [
        X1.max() if len(X1) > 0 else 0,  # Traiettoria freccia
        params.target_distance,           # Bersaglio
        x_sight_max                       # Estensione linea di mira
    ]
    
    if no_drag_result is not None:
        x_elements.append(no_drag_result.X.max() if len(no_drag_result.X) > 0 else 0)
    
    x_max_plot = max(x_elements)
    x_margin = max(2, x_max_plot * 0.02)  # Margine 2%
    x_max_final = x_max_plot + x_margin
    
    # Linea di mira ESTESA fino ai nuovi limiti
    x_sight = np.array([0, x_max_final])
    y_sight = y0 + np.tan(angle_rad) * x_sight
    ax_traj.plot(x_sight, y_sight, color=PLOT_CONFIG['colors']['danger'],
                linestyle=':', linewidth=2, label="Linea di mira", alpha=0.8)
    
    # ===== FINE CORREZIONE =====
    
    # Drop al bersaglio
    y_impact = interpolate_trajectory_point(X1, Y1, params.target_distance)
    y_sight_target = y0 + np.tan(angle_rad) * params.target_distance
    drop_cm = (y_sight_target - y_impact) * 100.0
    
    # Info vento
    if abs(params.wind_speed) > 0.1 and show_wind:
        wind_desc = "favorevole" if params.wind_speed > 0 else "contrario"
        wind_text = f"Vento: {params.wind_speed:+.1f} m/s ({wind_desc})"
        ax_traj.text(0.02, 0.98, wind_text, transform=ax_traj.transAxes,
                    bbox=dict(boxstyle="round,pad=0.4", 
                            facecolor=PLOT_CONFIG['colors']['info'], alpha=0.8),
                    fontsize=11, verticalalignment='top', color='white')
    
    # Stile grafico principale
    ax_traj.set_xlabel("Distanza (m)", fontsize=12, fontweight='bold')
    ax_traj.set_ylabel("Altezza (m)", fontsize=12, fontweight='bold')
    ax_traj.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax_traj.legend(loc='upper right', framealpha=0.9, fontsize=10)
    
    # Titolo informativo
    title = (
        f"Simulazione Balistica Avanzata - Integrazione RK4 Adattiva\n"
        f"Angolo: {main_result.angle_degrees:.2f}° | "
        f"v₀: {main_result.v0:.1f} m/s | "
        f"Tempo volo: {main_result.flight_time:.2f} s | "
        f"Altezza max: {main_result.max_height:.1f} m"
    )
    ax_traj.set_title(title, fontsize=14, pad=20, fontweight='bold')

    ax_traj.set_title(title, fontsize=14, pad=20, fontweight='bold')
    
    # ===== LIMITI ASSI CORRETTI =====
    
    # Limite X: usa il nuovo calcolo
    ax_traj.set_xlim(0, x_max_final)
    
    # Limiti Y: considera tutti i punti inclusa la linea di mira estesa
    y_elements = [
        Y1.min() if len(Y1) > 0 else 0, 
        Y1.max() if len(Y1) > 0 else 0, 
        params.target_height, 
        y0, 
        y_sight_target,
        y_sight[1]  # Punto finale linea di mira estesa
    ]
    
    if no_drag_result:
        y_elements.extend([no_drag_result.Y.min(), no_drag_result.Y.max()])
    
    y_min_all = min(y_elements)
    y_max_all = max(y_elements)
    
    # Logica limiti Y migliorata
    if params.target_height >= y0:
        # Tiro in piano o verso l'alto
        y_min_plot = min(0.0, y_min_all - 0.5)
        y_max_plot = y_max_all + 1.0
    else:
        # Tiro verso il basso
        y_min_plot = y_min_all - 1.0
        y_max_plot = max(y0, y_max_all) + 0.5
    
    ax_traj.set_ylim(y_min_plot, y_max_plot)
    
    # Annotazione Drop (dopo aver impostato i limiti)
    if abs(drop_cm) > 0.5:
        y_center = 0.5 * (y_min_plot + y_max_plot)
        offset = -1.0 if y_impact > y_center else 1.0
        y_text = np.clip(y_impact + offset, y_min_plot + 0.3, y_max_plot - 0.3)
        
        ax_traj.annotate(
            f"Drop: {drop_cm:.1f} cm",
            xy=(params.target_distance, y_impact),
            xytext=(params.target_distance - 0.25 * params.target_distance, y_text),
            arrowprops=dict(arrowstyle="->", color=PLOT_CONFIG['colors']['danger'], lw=2),
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="red", alpha=0.9),
            fontsize=11, fontweight='bold', clip_on=True
        )
    
    # === RESTO DEL CODICE INVARIATO ===
    # (grafici velocità, energia, drop rimangono identici)
    
    # === GRAFICO VELOCITÀ ===
    V_total = np.sqrt(main_result.V_x**2 + main_result.V_y**2)
    ax_vel.plot(main_result.X, V_total, color=PLOT_CONFIG['colors']['success'], 
               linewidth=2.5, label='Velocità totale')
    ax_vel.plot(main_result.X, main_result.V_x, color=PLOT_CONFIG['colors']['primary'], 
               linewidth=1.8, alpha=0.8, label='Componente X')
    ax_vel.plot(main_result.X, main_result.V_y, color=PLOT_CONFIG['colors']['danger'], 
               linewidth=1.8, alpha=0.8, label='Componente Y')
    
    ax_vel.set_xlabel("Distanza (m)", fontsize=11)
    ax_vel.set_ylabel("Velocità (m/s)", fontsize=11)
    ax_vel.grid(True, alpha=0.3)
    ax_vel.legend(fontsize=9)
    ax_vel.set_title("Profilo Velocità", fontsize=12, fontweight='bold')
    
    # === GRAFICO ENERGIA ===
    mass_kg = params.mass / 1000.0
    kinetic_energy = 0.5 * mass_kg * V_total**2
    initial_energy = kinetic_energy[0]
    energy_retention = kinetic_energy / initial_energy * 100
    
    ax_energy.plot(main_result.X, energy_retention, 
                  color=PLOT_CONFIG['colors']['warning'], linewidth=2.5)
    ax_energy.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='90% ritenzione')
    ax_energy.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='80% ritenzione')
    
    ax_energy.set_xlabel("Distanza (m)", fontsize=11)
    ax_energy.set_ylabel("Ritenzione Energia (%)", fontsize=11)
    ax_energy.grid(True, alpha=0.3)
    ax_energy.legend(fontsize=9)
    ax_energy.set_title("Ritenzione Energia Cinetica", fontsize=12, fontweight='bold')
    ax_energy.set_ylim(70, 105)
    
    # === GRAFICO DROP ===
    # Calcola drop per range esteso
    distances_drop = np.linspace(5, params.target_distance * 1.2, 50)
    drops_calculated = []
    
    for d in distances_drop:
        if d <= main_result.X.max():
            y_arrow = interpolate_trajectory_point(main_result.X, main_result.Y, d)
            y_sight = y0 + np.tan(angle_rad) * d
            drop = (y_sight - y_arrow) * 100  # cm
            drops_calculated.append(drop)
        else:
            drops_calculated.append(np.nan)
    
    valid_mask = ~np.isnan(drops_calculated)
    if np.any(valid_mask):
        ax_drop.plot(distances_drop[valid_mask], np.array(drops_calculated)[valid_mask],
                    color=PLOT_CONFIG['colors']['secondary'], linewidth=2.5,
                    marker='o', markersize=4, alpha=0.8)
        
        # Evidenzia bersaglio
        if params.target_distance <= distances_drop.max():
            target_drop_idx = np.argmin(np.abs(distances_drop - params.target_distance))
            if not np.isnan(drops_calculated[target_drop_idx]):
                ax_drop.scatter(params.target_distance, drops_calculated[target_drop_idx],
                               color=PLOT_CONFIG['colors']['accent'], s=150, 
                               marker='*', edgecolors='black', linewidth=2, zorder=10)
    
    ax_drop.set_xlabel("Distanza (m)", fontsize=11)
    ax_drop.set_ylabel("Drop (cm)", fontsize=11)
    ax_drop.grid(True, alpha=0.3)
    ax_drop.set_title("Curva Drop", fontsize=12, fontweight='bold')
    
    # Zona comfort tiro (±5cm)
    ax_drop.axhspan(-5, 5, alpha=0.2, color='green', label='Zona comfort (±5cm)')
    ax_drop.legend(fontsize=9)
    
    plt.tight_layout()
    return fig

def create_sight_scale_visualization(sight_data: pd.DataFrame, eye_to_nock: float, nock_to_riser: float, laser_distance: float = 30.0) -> plt.Figure:
    """Visualizzazione scala mirino interattiva"""
    
    fig, ax = plt.subplots(figsize=(6, 10))
    
    if len(sight_data) == 0:
        ax.text(0.5, 0.5, "Nessun dato disponibile", 
               transform=ax.transAxes, ha='center', va='center',
               fontsize=14, color='red')
        return fig
    
    # Estrai dati
    distances = sight_data['Distanza (m)'].values
    projections = sight_data['Proiezione riser (cm)'].values
    
    # Linea centrale riser
    y_range = [projections.min() - 3, projections.max() + 3]
    ax.axvline(x=0, ymin=0, ymax=1, color='black', linewidth=6, alpha=0.8)
    
    # Tacche distanze con colori graduali
    colors_grad = plt.cm.viridis(np.linspace(0, 1, len(distances)))
    
    for i, (dist, proj, color) in enumerate(zip(distances, projections, colors_grad)):
        # Tacca orizzontale
        ax.hlines(proj, xmin=-1.5, xmax=1.5, colors=color, linewidth=3, alpha=0.9)
        
        # Etichetta distanza
        side = 1 if i % 2 == 0 else -1
        ax.text(side * 2.2, proj, f"{int(dist)}m", 
               va='center', ha='left' if side > 0 else 'right',
               fontsize=11, fontweight='bold', color=color)
        
        # Drop info
        drop_val = sight_data.iloc[i]['Drop (cm)']
        ax.text(side * 3.5, proj, f"({drop_val:.1f}cm)", 
               va='center', ha='left' if side > 0 else 'right',
               fontsize=9, style='italic', alpha=0.7)


    # ---- Laser geometrico: raggio dalla punta della freccia ----
    # Usa la stessa geometria del PDF: y_cm(x, o, t, d=0) dove
    #   o = eye_to_nock, t = nock_to_riser, x = laser_distance (m)
    def _y_cm(x: float, o: float, t: float, d: float = 0.0) -> float:
        u = (o + d) / (t + x)
        y_calc = 100.0 * x * (u / np.sqrt(1 + u**2))  # cm sul riser
        return y_calc - d * 100.0
    
    y_laser = _y_cm(laser_distance, eye_to_nock, nock_to_riser, d=0.0)
    # Disegna il laser come stella rossa sulla colonna del riser (x=0)
    ax.scatter(0, y_laser, marker='*', s=220, edgecolors='darkred', color='red', zorder=10)
    ax.text(2.2, y_laser, f"Laser {int(laser_distance)} m", va='center', fontsize=11, color='red', fontweight='bold')
    
    # Allarga l'asse Y se necessario per includere il laser
    y_min_cur, y_max_cur = ax.get_ylim()
    new_min = min(y_min_cur, y_laser - 2)
    new_max = max(y_max_cur, y_laser + 2)
    ax.set_ylim(new_min, new_max)

    # Stile grafico
    ax.set_xlim(-5, 5)
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_ylabel("Posizione su Riser (cm)", fontsize=14, fontweight='bold')
    ax.set_title("Scala Mirino Verticale", fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.set_xticks([])  # Rimuovi tick X
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    return fig

# ==============================
# ANALISI MONTE CARLO AVANZATA
# ==============================
class MonteCarloAnalyzer:
    """Analizzatore statistico Monte Carlo per incertezze parametriche"""
    
    def __init__(self, n_simulations: int = 200):
        self.n_simulations = n_simulations
        self.results_cache = {}
    
    def run_uncertainty_analysis(self, base_params: SimulationParams, 
                                integrator: AdvancedRK4Integrator,
                                variations: Dict[str, float] = None) -> Dict[str, Any]:
        """Esegue analisi incertezza completa"""
        
        if variations is None:
            variations = {
                'mass': 0.03,           # ±3% peso freccia
                'v0': 0.06,             # ±6% velocità
                'wind_speed': 1.5,      # ±1.5 m/s vento
                'air_temperature': 8.0, # ±8°C temperatura
                'draw_force': 0.02,     # ±2% forza arco
                'diameter': 0.05        # ±5% diametro
            }
        
        # Trova angolo base
        base_angle, _ = find_optimal_firing_angle(base_params, integrator)
        
        # Storage risultati
        mc_results = {
            'angles': [], 'drops_at_target': [], 'flight_times': [],
            'max_heights': [], 'velocities': [], 'impact_points_x': [],
            'impact_points_y': [], 'energy_losses': []
        }
        
        # Simulazioni Monte Carlo
        for i in range(self.n_simulations):
            try:
                # Genera parametri perturbati
                perturbed_params = self._generate_perturbed_params(base_params, variations)
                
                # Simula traiettoria
                result = integrator.integrate_trajectory(base_angle, perturbed_params,
                                                       include_drag=True, include_wind=True,
                                                       max_range=base_params.target_distance * 1.3)
                
                # Calcola metriche
                y_impact = interpolate_trajectory_point(result.X, result.Y, base_params.target_distance)
                y_sight = (base_params.launch_height + 
                          np.tan(np.radians(base_angle)) * base_params.target_distance)
                drop_cm = (y_sight - y_impact) * 100
                
                # Salva risultati
                mc_results['angles'].append(base_angle)
                mc_results['drops_at_target'].append(drop_cm)
                mc_results['flight_times'].append(result.flight_time)
                mc_results['max_heights'].append(result.max_height)
                mc_results['velocities'].append(result.v0)
                mc_results['impact_points_x'].append(result.X[-1] if len(result.X) > 0 else 0)
                mc_results['impact_points_y'].append(result.Y[-1] if len(result.Y) > 0 else 0)
                mc_results['energy_losses'].append(result.energy_loss)
                
            except Exception as e:
                logger.warning(f"Errore simulazione MC {i+1}: {e}")
                continue
        
        # Calcola statistiche
        stats = self._calculate_statistics(mc_results)
        
        return stats
    
    def _generate_perturbed_params(self, base_params: SimulationParams, 
                                  variations: Dict[str, float]) -> SimulationParams:
        """Genera parametri con perturbazioni stocastiche"""
        perturbed = replace(base_params)
        
        for param_name, std_dev in variations.items():
            if hasattr(perturbed, param_name):
                current_val = getattr(perturbed, param_name)
                
                if param_name in ['wind_speed', 'air_temperature']:
                    # Variazione assoluta
                    noise = np.random.normal(0, std_dev)
                    new_val = current_val + noise
                else:
                    # Variazione relativa
                    noise = np.random.normal(0, std_dev)
                    new_val = current_val * (1 + noise)
                
                # Applica limiti fisici
                new_val = self._apply_parameter_limits(param_name, new_val)
                setattr(perturbed, param_name, new_val)
        
        return perturbed
    
    def _apply_parameter_limits(self, param_name: str, value: float) -> float:
        """Applica limiti fisici realistici ai parametri"""
        limits = {
            'mass': (5.0, 100.0),
            'v0': (10.0, 150.0),
            'wind_speed': (-25.0, 25.0),
            'air_temperature': (-30.0, 60.0),
            'draw_force': (15.0, 150.0),
            'diameter': (3.0, 15.0)
        }
        
        if param_name in limits:
            min_val, max_val = limits[param_name]
            return np.clip(value, min_val, max_val)
        
        return value
    
    def _calculate_statistics(self, results: Dict[str, List]) -> Dict[str, Any]:
        """Calcola statistiche descrittive complete"""
        stats = {}
        
        for metric, values in results.items():
            if values:  # Solo se ci sono valori
                values_array = np.array(values)
                stats[metric] = {
                    'mean': np.mean(values_array),
                    'std': np.std(values_array),
                    'min': np.min(values_array),
                    'max': np.max(values_array),
                    'median': np.median(values_array),
                    'percentile_5': np.percentile(values_array, 5),
                    'percentile_95': np.percentile(values_array, 95),
                    'q1': np.percentile(values_array, 25),
                    'q3': np.percentile(values_array, 75),
                    'iqr': np.percentile(values_array, 75) - np.percentile(values_array, 25),
                    'cv': np.std(values_array) / np.abs(np.mean(values_array)) * 100  # Coeff. variazione
                }
        
        return stats

# ==============================
# EXPORT DATI COMPLETO
# ==============================
def export_comprehensive_analysis(trajectory_result: TrajectoryResults,
                                 params: SimulationParams,
                                 sight_data: pd.DataFrame,
                                 monte_carlo_stats: Optional[Dict] = None) -> io.BytesIO:
    """Export completo analisi in Excel multi-sheet"""
    
    output = io.BytesIO()
    
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet 1: Configurazione
            config_data = pd.DataFrame({
                'Parametro': [
                    'Peso freccia (g)', 'Lunghezza (m)', 'Diametro (mm)', 'Spine',
                    'Tipo punta', 'Forza arco (lb)', 'Allungo (m)', 'Brace height (m)',
                    'Efficienza', 'Tipo arco', 'Altezza lancio (m)',
                    'Distanza bersaglio (m)', 'Altezza bersaglio (m)',
                    'Velocità vento (m/s)', 'Temperatura (°C)', 'Pressione (hPa)',
                    'Umidità (%)', 'Altitudine (m)'
                ],
                'Valore': [
                    params.mass, params.length, params.diameter, params.spine,
                    params.tip_type, params.draw_force, params.draw_length, params.brace_height,
                    params.efficiency, params.bow_type, params.launch_height,
                    params.target_distance, params.target_height,
                    params.wind_speed, params.air_temperature, params.air_pressure,
                    params.humidity, params.altitude
                ]
            })
            config_data.to_excel(writer, sheet_name='Configurazione', index=False)
            
            # Sheet 2: Risultati principali
            y_impact = interpolate_trajectory_point(trajectory_result.X, trajectory_result.Y, 
                                                   params.target_distance)
            y_sight = (params.launch_height + 
                      np.tan(np.radians(trajectory_result.angle_degrees)) * params.target_distance)
            drop_target = (y_sight - y_impact) * 100
            
            results_data = pd.DataFrame({
                'Metrica': [
                    'Angolo ottimale (°)', 'Velocità iniziale (m/s)', 'Tempo di volo (s)',
                    'Altezza massima (m)', 'Gittata massima (m)', 'Drop al bersaglio (cm)',
                    'Perdita energia (J)', 'Efficienza integrazione (%)'
                ],
                'Valore': [
                    trajectory_result.angle_degrees, trajectory_result.v0,
                    trajectory_result.flight_time, trajectory_result.max_height,
                    trajectory_result.range_distance, drop_target,
                    trajectory_result.energy_loss,
                    100 * trajectory_result.integration_stats['steps'] / 
                    (trajectory_result.integration_stats['steps'] + 
                     trajectory_result.integration_stats['rejections'])
                ]
            })
            results_data.to_excel(writer, sheet_name='Risultati_Principali', index=False)
            
            # Sheet 3: Traiettoria completa (campionamento per dimensioni file)
            max_points = 500
            step = max(1, len(trajectory_result.X) // max_points)
            
            trajectory_data = pd.DataFrame({
                'Distanza (m)': trajectory_result.X[::step],
                'Altezza (m)': trajectory_result.Y[::step],
                'Velocità X (m/s)': trajectory_result.V_x[::step],
                'Velocità Y (m/s)': trajectory_result.V_y[::step],
                'Velocità totale (m/s)': np.sqrt(trajectory_result.V_x[::step]**2 + 
                                                trajectory_result.V_y[::step]**2),
                'Tempo (s)': trajectory_result.times[::step],
                'Energia cinetica (J)': 0.5 * (params.mass/1000.0) * 
                                       (trajectory_result.V_x[::step]**2 + 
                                        trajectory_result.V_y[::step]**2)
            })
            trajectory_data.to_excel(writer, sheet_name='Traiettoria_Completa', index=False)
            
            # Sheet 4: Scala mirino
            sight_data.to_excel(writer, sheet_name='Scala_Mirino', index=False)
            
            # Sheet 5: Statistiche Monte Carlo (se disponibili)
            if monte_carlo_stats:
                mc_summary = []
                for metric, stats in monte_carlo_stats.items():
                    if isinstance(stats, dict) and 'mean' in stats:
                        mc_summary.append({
                            'Metrica': metric.replace('_', ' ').title(),
                            'Media': round(stats['mean'], 4),
                            'Deviazione Standard': round(stats['std'], 4),
                            'Mediana': round(stats['median'], 4),
                            'Min': round(stats['min'], 4),
                            'Max': round(stats['max'], 4),
                            'Percentile 5%': round(stats['percentile_5'], 4),
                            'Percentile 95%': round(stats['percentile_95'], 4),
                            'IQR': round(stats['iqr'], 4),
                            'Coeff. Variazione (%)': round(stats['cv'], 2)
                        })
                
                if mc_summary:
                    mc_df = pd.DataFrame(mc_summary)
                    mc_df.to_excel(writer, sheet_name='Statistiche_Monte_Carlo', index=False)
        
        output.seek(0)
        return output
    
    except Exception as e:
        logger.error(f"Errore export Excel: {e}")
        # Fallback: export CSV semplice
        output = io.BytesIO()
        sight_data.to_csv(output, index=False)
        output.seek(0)
        return output

# ==============================
# FUNZIONI DI SUPPORTO GEOMETRICO
# ==============================
def calculate_postural_launch_height(params: SimulationParams, angle_rad: float) -> float:
    """Calcola altezza lancio considerando postura dinamica arciere"""
    anchor_length = params.anchor_length
    neutral_height = params.launch_height_neutral
    pelvis_height = params.pelvis_height
    
    # Geometria corpo arciere
    pivot_height = neutral_height - pelvis_height
    arm_displacement_x = anchor_length * np.cos(angle_rad)
    arm_displacement_y = anchor_length * np.sin(angle_rad)
    
    # Altezza finale lancio
    final_height = pivot_height + pelvis_height + arm_displacement_y
    
    return max(0.1, final_height)  # Altezza minima sicurezza

def iterative_angle_convergence(params: SimulationParams, integrator: AdvancedRK4Integrator,
                               tolerance_angle: float = 0.005, tolerance_height: float = 0.001,
                               max_iterations: int = 20) -> Tuple[float, float]:
    """Convergenza iterativa angolo-postura per massima precisione"""
    
    current_height = params.launch_height_neutral
    current_angle = 0.0
    
    for iteration in range(max_iterations):
        # Aggiorna parametri con altezza corrente
        updated_params = replace(params, launch_height=current_height)
        
        # Trova nuovo angolo ottimale
        new_angle, error = find_optimal_firing_angle(updated_params, integrator)
        
        # Calcola nuova altezza lancio
        new_height = calculate_postural_launch_height(params, np.radians(new_angle))
        
        # Verifica convergenza
        angle_diff = abs(new_angle - current_angle)
        height_diff = abs(new_height - current_height)
        
        if angle_diff < tolerance_angle and height_diff < tolerance_height:
            logger.info(f"Convergenza raggiunta dopo {iteration+1} iterazioni")
            return new_angle, new_height
        
        # Aggiorna valori per iterazione successiva
        current_angle = new_angle
        current_height = new_height
    
    logger.warning(f"Convergenza non raggiunta dopo {max_iterations} iterazioni")
    return current_angle, current_height

# ==============================
# INTERFACCIA STREAMLIT PRINCIPALE
# ==============================
def main():
    st.set_page_config(
        page_title="Simulatore Balistico Avanzato con Mirino",
        page_icon="🏹",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header principale con design moderno
    st.markdown("""
    <div style='text-align: center; padding: 2rem; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; border-radius: 15px; margin-bottom: 2rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
        <h1 style='margin-bottom: 0.5rem; font-size: 2.5rem;'>🏹 Simulatore Balistico Professionale</h1>
        <p style='margin: 0; font-size: 1.2rem; opacity: 0.9;'>
            Simulazione RK4 Adattiva • Generazione Mirino • Analisi Monte Carlo
        </p>
        <p style='margin-top: 0.5rem; font-size: 1rem; opacity: 0.8;'>
            Modello fisico avanzato con correzioni ambientali e geometria posturale
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controlli avanzati
    with st.sidebar:
        st.markdown("### ⚙️ Controlli Simulazione")
        
        # Parametri integrazione
        with st.expander("🔧 Parametri RK4 Adattivo", expanded=False):
            tolerance = st.selectbox("Tolleranza", 
                                   options=[1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
                                   index=2, format_func=lambda x: f"{x:.0e}")
            max_dt = st.slider("Passo massimo (s)", 0.005, 0.05, 0.02, step=0.005)
            min_dt = st.selectbox("Passo minimo (s)",
                                options=[1e-6, 1e-7, 1e-8, 1e-9],
                                index=2, format_func=lambda x: f"{x:.0e}")
            max_steps = st.number_input("Max iterazioni", 10000, 100000, 50000, step=5000)
        
        # Analisi drop
        with st.expander("📊 Analisi Drop", expanded=True):
            dist_min = st.number_input("Distanza min (m)", 5, 50, 10)
            dist_max = st.number_input("Distanza max (m)", 20, 150, 60)
            dist_step = st.number_input("Passo distanza (m)", 1, 10, 5)
            poly_degree = st.slider("Grado interpolazione", 1, 4, 2)
        
        # Monte Carlo
        with st.expander("🎲 Analisi Monte Carlo", expanded=False):
            enable_monte_carlo = st.checkbox("Abilita analisi incertezza")
            if enable_monte_carlo:
                n_mc_sims = st.slider("N° simulazioni", 50, 1000, 200, step=50)
                st.markdown("**Variazioni parametriche:**")
                mass_variation = st.slider("Variazione peso (%)", 0, 10, 3) / 100
                velocity_variation = st.slider("Variazione velocità (%)", 0, 15, 6) / 100
                wind_variation = st.slider("Variazione vento (m/s)", 0, 5, 1.5, step=0.5)
                temp_variation = st.slider("Variazione temperatura (°C)", 0, 15, 8, step=1)
        
        # Opzioni visualizzazione
        with st.expander("🎨 Opzioni Grafiche", expanded=True):
            show_drag_comparison = st.checkbox("Confronto con/senza resistenza")
            show_wind_effects = st.checkbox("Visualizza effetti vento", value=True)
            show_energy_analysis = st.checkbox("Analisi energia", value=True)
            high_quality_plots = st.checkbox("Grafici alta qualità", value=True)
    
    # Layout principale tre colonne
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("### 🎯 Parametri Freccia")
        mass = st.number_input("Peso (g)", 8.0, 80.0, 28.0, step=0.5,
                              help="Peso totale freccia inclusa punta")
        length = st.number_input("Lunghezza (m)", 0.50, 1.20, 0.76, step=0.01)
        diameter = st.number_input("Diametro (mm)", 4.0, 12.0, 6.5, step=0.1)
        spine = st.number_input("Spine", 200, 1500, 700, step=25,
                               help="Rigidità freccia in lb/pollice")
        balance_point = st.number_input("Bilanciamento (m dal nock)", 0.20, 0.60, 0.42, step=0.01)
        tip_type = st.selectbox("Tipo punta", list(TIPO_PUNTA_CD_FACTOR.keys()),
                               help="Influenza coefficiente resistenza")
        
        st.markdown("### 🏹 Parametri Arco")
        draw_force = st.number_input("Forza (lb)", 20.0, 100.0, 42.0, step=1.0)
        draw_length = st.number_input("Allungo (m)", 0.55, 0.85, 0.71, step=0.01)
        brace_height = st.number_input("Brace height (m)", 0.10, 0.25, 0.19, step=0.01)
        bow_type = st.selectbox("Tipo arco", list(BOW_TYPE_DEFAULT_EFF.keys()))
        efficiency = st.number_input("Efficienza", 0.60, 0.95, 
                                    BOW_TYPE_DEFAULT_EFF.get(bow_type, 0.80), 
                                    step=0.01, help="Frazione energia trasferita alla freccia")
    
    with col2:
        st.markdown("### 👤 Parametri Arciere")
        launch_height_neutral = st.number_input("Altezza neutra (m)", 1.0, 2.0, 1.55, step=0.01,
                                               help="Altezza lancio in posizione neutra")
        anchor_length = st.number_input("Lunghezza spalla-aggancio (m)", 0.50, 0.90, 0.72, step=0.01)
        pelvis_height = st.number_input("Altezza bacino (m)", 0.50, 1.30, 1.05, step=0.01)
        eye_offset_v = st.number_input("Offset verticale occhio (m)", 0.02, 0.20, 0.08, step=0.01)
        
        st.markdown("### 🎯 Parametri Bersaglio")
        target_distance = st.number_input("Distanza (m)", 5.0, 100.0, 40.0, step=1.0)
        target_height= st.number_input("Altezza (m)", -10.0, 10.0, 1.4, step=0.1)
        
        st.markdown("### ⚡ Velocità")
        use_measured_v0 = st.checkbox("Usa velocità misurata", 
                                     help="Altrimenti calcolata da parametri arco")
        v0_measured = st.number_input("v₀ misurata (m/s)", 20.0, 120.0, 58.0, step=1.0,
                                     disabled=not use_measured_v0)
    
    with col3:
        st.markdown("### 🌍 Condizioni Ambientali")
        wind_speed = st.number_input("Velocità vento (m/s)", -15.0, 15.0, 0.0, step=0.5,
                                    help="Positivo = favorevole, Negativo = contrario")
        air_temperature = st.number_input("Temperatura (°C)", -20, 45, 18, step=1)
        air_pressure = st.number_input("Pressione (hPa)", 950, 1050, 1013, step=1)
        humidity = st.number_input("Umidità relativa (%)", 10, 95, 55, step=5)
        altitude = st.number_input("Altitudine (m slm)", 0, 3000, 150, step=50,
                                  help="Correzione automatica pressione/temperatura")
        
        st.markdown("### 🔍 Geometria Mirino")
        eye_to_nock = st.number_input("Distanza occhio-cocca (m)", 0.08, 0.30, 0.12, 
                                     step=0.005, format="%.3f")
        nock_to_riser = st.number_input("Distanza cocca-riser (m)", 0.40, 1.20, 0.68, step=0.01)
        
        st.markdown("### 🎛️ Opzioni Avanzate")
        use_postural_correction = st.checkbox("Correzione posturale", value=True,
                                             help="Aggiusta altezza lancio per angolo tiro")
        use_iterative_convergence = st.checkbox("Convergenza iterativa", value=False,
                                               help="Precisione massima (più lento)")
    
    # Pulsante calcolo principale
    st.markdown("---")
    if st.button("🚀 ESEGUI SIMULAZIONE COMPLETA", type="primary", use_container_width=True):
        
        # Validazione e creazione parametri
        try:
            params = SimulationParams(
                mass=mass, length=length, spine=spine, diameter=diameter,
                balance_point=balance_point, tip_type=tip_type,
                draw_force=draw_force, draw_length=draw_length, brace_height=brace_height,
                efficiency=efficiency, bow_type=bow_type,
                launch_height_neutral=launch_height_neutral,
                anchor_length=anchor_length, pelvis_height=pelvis_height, eye_offset_v=eye_offset_v,
                target_distance=target_distance, target_height=params.target_height,
                use_measured_v0=use_measured_v0, v0=v0_measured,
                wind_speed=wind_speed, air_temperature=air_temperature,
                air_pressure=air_pressure, humidity=humidity, altitude=altitude
            )
        except ValueError as e:
            st.error(f"❌ Errore validazione parametri: {e}")
            return
        
        # Inizializzazione integratore
        integrator = AdvancedRK4Integrator(
            tol=tolerance, max_dt=max_dt, min_dt=min_dt, max_steps=max_steps
        )
        
        # Progress tracking
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        try:
            # Fase 1: Calcolo angolo ottimale
            status_text.text("🔍 Ricerca angolo ottimale...")
            progress_bar.progress(15)
            
            if use_iterative_convergence:
                optimal_angle, final_height = iterative_angle_convergence(params, integrator)
                params = replace(params, launch_height=final_height)
            else:
                optimal_angle, optimization_error = find_optimal_firing_angle(params, integrator)
                if optimization_error > 3.0:
                    st.warning(f"⚠️ Errore ottimizzazione elevato: {optimization_error:.2f}m")
                
                if use_postural_correction:
                    corrected_height = calculate_postural_launch_height(params, np.radians(optimal_angle))
                    params = replace(params, launch_height=corrected_height)
            
            # Fase 2: Simulazione principale
            status_text.text("📈 Simulazione traiettoria realistica...")
            progress_bar.progress(30)
            
            max_simulation_range = max(dist_max, target_distance) * 1.3
            main_result = integrator.integrate_trajectory(
                optimal_angle, params, include_drag=True, include_wind=True,
                max_range=max_simulation_range
            )
            
            # Fase 3: Simulazione ideale (opzionale)
            ideal_result = None
            if show_drag_comparison:
                status_text.text("📈 Simulazione traiettoria ideale...")
                progress_bar.progress(45)
                ideal_result = integrator.integrate_trajectory(
                    optimal_angle, params, include_drag=False, include_wind=False,
                    max_range=max_simulation_range
                )
            
            # Fase 4: Analisi drop
            status_text.text("📊 Calcolo curve drop...")
            progress_bar.progress(60)
            
            distances_analysis = np.arange(dist_min, dist_max + 1, dist_step)
            drops_cm_calculated = []
            
            for dist in distances_analysis:
                if dist <= main_result.X.max():
                    y_arrow = interpolate_trajectory_point(main_result.X, main_result.Y, dist)
                    y_sight = (params.launch_height + 
                              np.tan(np.radians(optimal_angle)) * dist)
                    drop_value = (y_sight - y_arrow) * 100.0
                    drops_cm_calculated.append(drop_value)
                else:
                    drops_cm_calculated.append(np.nan)
            
            # Interpolazione curva drop
            valid_drops = ~np.isnan(drops_cm_calculated)
            spline_interpolator = None
            polynomial_fit = None
            
            if np.sum(valid_drops) > 3:
                valid_distances = distances_analysis[valid_drops]
                valid_drop_values = np.array(drops_cm_calculated)[valid_drops]
                
                try:
                    spline_interpolator = UnivariateSpline(valid_distances, valid_drop_values, s=0)
                except:
                    pass
                
                if len(valid_distances) >= poly_degree + 1:
                    try:
                        poly_coeffs = np.polyfit(valid_distances, valid_drop_values, poly_degree)
                        polynomial_fit = np.poly1d(poly_coeffs)
                    except:
                        pass
            
            # Fase 5: Generazione scala mirino
            status_text.text("🎯 Generazione scala mirino...")
            progress_bar.progress(75)
            
            sight_calculator = SightingSystemCalculator(eye_to_nock, nock_to_riser)
            sight_scale_data = sight_calculator.generate_sight_marks(
                distances_analysis[valid_drops], 
                np.array(drops_cm_calculated)[valid_drops]
            )
            
            # Fase 6: Analisi Monte Carlo (opzionale)
            mc_statistics = None
            if enable_monte_carlo:
                status_text.text("🎲 Analisi Monte Carlo...")
                progress_bar.progress(90)
                
                mc_analyzer = MonteCarloAnalyzer(n_mc_sims)
                variations_dict = {
                    'mass': mass_variation,
                    'v0': velocity_variation if use_measured_v0 else 0.04,
                    'wind_speed': wind_variation,
                    'air_temperature': temp_variation,
                    'draw_force': 0.02 if not use_measured_v0 else 0,
                    'diameter': 0.03
                }
                
                try:
                    mc_statistics = mc_analyzer.run_uncertainty_analysis(
                        params, integrator, variations_dict
                    )
                except Exception as e:
                    st.warning(f"⚠️ Errore analisi Monte Carlo: {e}")
            
            progress_bar.progress(100)
            status_text.text("✅ Simulazione completata!")
            
            # ===== VISUALIZZAZIONE RISULTATI =====
            st.markdown("---")
            st.markdown("## 📊 Risultati Simulazione")
            
            # Metriche principali
            metrics_cols = st.columns(5)
            
            with metrics_cols[0]:
                direct_angle = np.degrees(np.arctan2(params.target_height - params.launch_height, 
                                                   target_distance))
                st.metric("Angolo Ottimale", f"{optimal_angle:.2f}°",
                         delta=f"{optimal_angle - direct_angle:+.2f}°")
            
            with metrics_cols[1]:
                st.metric("Velocità Iniziale", f"{main_result.v0:.1f} m/s")
            
            with metrics_cols[2]:
                st.metric("Tempo di Volo", f"{main_result.flight_time:.2f} s")
            
            with metrics_cols[3]:
                target_drop = (params.launch_height + np.tan(np.radians(optimal_angle)) * target_distance -
                              interpolate_trajectory_point(main_result.X, main_result.Y, target_distance)) * 100
                st.metric("Drop al Bersaglio", f"{target_drop:.1f} cm")
            
            with metrics_cols[4]:
                energy_retention = ((main_result.V_x[-1]**2 + main_result.V_y[-1]**2) / 
                                  main_result.v0**2) * 100
                st.metric("Ritenzione Energia", f"{energy_retention:.1f}%")
            
            # Grafico principale traiettoria
            st.markdown("### 📈 Analisi Traiettoria Completa")
            trajectory_figure = create_comprehensive_trajectory_plot(
                main_result, params, ideal_result, show_wind_effects
            , target_distance=target_distance)
            st.pyplot(trajectory_figure, use_container_width=True)
            
            # Statistiche integrazione
            with st.expander("🔧 Statistiche Integrazione Numerica"):
                integration_cols = st.columns(6)
                stats = main_result.integration_stats
                
                with integration_cols[0]:
                    st.metric("Passi Accettati", stats['steps'])
                with integration_cols[1]:
                    st.metric("Passi Rifiutati", stats['rejections'])
                with integration_cols[2]:
                    st.metric("Step Min", f"{stats['min_dt_used']:.2e} s")
                with integration_cols[3]:
                    st.metric("Step Max", f"{stats['max_dt_used']:.2e} s")
                with integration_cols[4]:
                    efficiency = (100 * stats['steps'] / 
                                (stats['steps'] + stats['rejections']) 
                                if (stats['steps'] + stats['rejections']) > 0 else 0)
                    st.metric("Efficienza", f"{efficiency:.1f}%")
                with integration_cols[5]:
                    st.metric("Perdita Energia", f"{main_result.energy_loss:.2f} J")
            
            # Scala mirino
            mirino_cols = st.columns([2, 1])
            
            with mirino_cols[0]:
                st.markdown("### 🎯 Scala Mirino")
                st.dataframe(
                    sight_scale_data.style.format({
                        'Drop (cm)': '{:.1f}',
                        'Proiezione riser (cm)': '{:.2f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            
            with mirino_cols[1]:
                if len(sight_scale_data) > 0:
                    st.markdown("### 🔍 Visualizzazione Mirino")
                    mirino_figure = create_sight_scale_visualization(sight_scale_data, eye_to_nock, nock_to_riser)
                    st.pyplot(mirino_figure, use_container_width=True)
            
            # Analisi Monte Carlo
            if mc_statistics:
                st.markdown("### 🎲 Analisi Statistica Incertezza")
                
                mc_summary_cols = st.columns(4)
                
                with mc_summary_cols[0]:
                    drops_stats = mc_statistics.get('drops_at_target', {})
                    st.metric("Drop Medio", 
                             f"{drops_stats.get('mean', 0):.1f} cm",
                             delta=f"±{drops_stats.get('std', 0):.1f} cm")
                
                with mc_summary_cols[1]:
                    velocity_stats = mc_statistics.get('velocities', {})
                    st.metric("Velocità Media",
                             f"{velocity_stats.get('mean', 0):.1f} m/s",
                             delta=f"±{velocity_stats.get('std', 0):.1f} m/s")
                
                with mc_summary_cols[2]:
                    time_stats = mc_statistics.get('flight_times', {})
                    st.metric("Tempo Medio",
                             f"{time_stats.get('mean', 0):.2f} s",
                             delta=f"±{time_stats.get('std', 0):.2f} s")
                
                with mc_summary_cols[3]:
                    height_stats = mc_statistics.get('max_heights', {})
                    st.metric("Altezza Max Media",
                             f"{height_stats.get('mean', 0):.1f} m",
                             delta=f"±{height_stats.get('std', 0):.1f} m")
                
                # Grafico distribuzione drop
                if 'drops_at_target' in mc_statistics:
                    fig_mc, ax_mc = plt.subplots(figsize=(12, 6))
                    
                    # Simula distribuzione per visualizzazione
                    drop_stats = mc_statistics['drops_at_target']
                    simulated_data = np.random.normal(drop_stats['mean'], drop_stats['std'], n_mc_sims)
                    
                    ax_mc.hist(simulated_data, bins=30, alpha=0.7, color='skyblue',
                              edgecolor='navy', density=True, label='Distribuzione')
                    
                    # Linee statistiche
                    ax_mc.axvline(drop_stats['mean'], color='red', linestyle='-', 
                                 linewidth=3, label=f"Media: {drop_stats['mean']:.1f} cm")
                    ax_mc.axvline(drop_stats['percentile_5'], color='orange', linestyle='--',
                                 linewidth=2, label=f"5° percentile: {drop_stats['percentile_5']:.1f} cm")
                    ax_mc.axvline(drop_stats['percentile_95'], color='orange', linestyle='--',
                                 linewidth=2, label=f"95° percentile: {drop_stats['percentile_95']:.1f} cm")
                    
                    ax_mc.set_xlabel("Drop (cm)", fontsize=12)
                    ax_mc.set_ylabel("Densità Probabilità", fontsize=12)
                    ax_mc.set_title("Distribuzione Drop al Bersaglio - Analisi Monte Carlo", 
                                   fontsize=14, fontweight='bold')
                    ax_mc.legend()
                    ax_mc.grid(True, alpha=0.3)
                    
                    st.pyplot(fig_mc, use_container_width=True)
                
                # Tabella dettagliata MC
                with st.expander("📊 Statistiche Dettagliate Monte Carlo"):
                    mc_detailed_data = []
                    for metric, stats in mc_statistics.items():
                        if isinstance(stats, dict) and 'mean' in stats:
                            mc_detailed_data.append({
                                'Metrica': metric.replace('_', ' ').title(),
                                'Media': f"{stats['mean']:.4f}",
                                'Mediana': f"{stats['median']:.4f}",
                                'Std Dev': f"{stats['std']:.4f}",
                                'Min': f"{stats['min']:.4f}",
                                'Max': f"{stats['max']:.4f}",
                                'IQR': f"{stats['iqr']:.4f}",
                                'CV (%)': f"{stats['cv']:.2f}"
                            })
                    
                    if mc_detailed_data:
                        mc_detailed_df = pd.DataFrame(mc_detailed_data)
                        st.dataframe(mc_detailed_df, use_container_width=True, hide_index=True)
            
            # ===== EXPORT DATI =====
            st.markdown("---")
            st.markdown("### 💾 Export e Download")
            
            export_cols = st.columns(4)
            
            with export_cols[0]:
                # CSV scala mirino
                if len(sight_scale_data) > 0:
                    csv_buffer = io.StringIO()
                    sight_scale_data.to_csv(csv_buffer, index=False)
                    st.download_button(
                        "📊 Download Scala Mirino (CSV)",
                        data=csv_buffer.getvalue(),
                        file_name=f"scala_mirino_{int(target_distance)}m.csv",
                        mime="text/csv"
                    )
            
            with export_cols[1]:
                # Excel analisi completa
                try:
                    excel_buffer = export_comprehensive_analysis(
                        main_result, params, sight_scale_data, mc_statistics
                    )
                    st.download_button(
                        "📈 Download Analisi Completa (Excel)",
                        data=excel_buffer,
                        file_name=f"analisi_balistica_completa_{int(target_distance)}m.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    st.error(f"Errore export Excel: {str(e)}")
            
            
            with export_cols[2]:
                # PDF scala mirino tabellare
                if len(sight_scale_data) > 0:
                    try:
                        pdf_generator = SightScalePDFGenerator()
                        pdf_buffer = pdf_generator.create_sight_scale_pdf(
                            sight_scale_data, sight_calculator, params
                        )
                        st.download_button(
                            "🎯 Download Mirino (PDF Tabellare)",
                            data=pdf_buffer,
                            file_name=f"mirino_tabellare_{int(target_distance)}m.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.error(f"Errore generazione PDF tabellare: {str(e)}")
            
                # PDF mirino grafico con laser
                try:
                    buf, fname = esporta_mirino_pdf_bytes(sight_scale_data, eye_to_nock, nock_to_riser)
                    st.download_button(
                        "📐 Download Mirino (PDF Grafico con Laser)",
                        data=buf,
                        file_name=fname,
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"Errore generazione PDF grafico: {str(e)}")
                        
            with export_cols[3]:
                # JSON configurazione
                config_export = {
                    'parametri_freccia': {
                        'massa_g': mass, 'lunghezza_m': length, 'diametro_mm': diameter,
                        'spine': spine, 'bilanciamento_m': balance_point, 'tipo_punta': tip_type
                    },
                    'parametri_arco': {
                        'forza_lb': draw_force, 'allungo_m': draw_length, 
                        'brace_height_m': brace_height, 'efficienza': efficiency, 'tipo': bow_type
                    },
                    'condizioni_ambientali': {
                        'vento_ms': wind_speed, 'temperatura_c': air_temperature,
                        'pressione_hpa': air_pressure, 'umidita_pct': humidity, 'altitudine_m': altitude
                    },
                    'risultati': {
                        'angolo_ottimale_gradi': optimal_angle, 'velocita_iniziale_ms': main_result.v0,
                        'drop_bersaglio_cm': target_drop, 'tempo_volo_s': main_result.flight_time
                    }
                }
                
                import json
                config_json = json.dumps(config_export, indent=2, ensure_ascii=False)
                st.download_button(
                    "⚙️ Download Configurazione (JSON)",
                    data=config_json,
                    file_name=f"config_simulazione_{int(target_distance)}m.json",
                    mime="application/json"
                )
            
            # ===== RIEPILOGO FINALE =====
            st.markdown("---")
            st.markdown("### 📋 Riepilogo Esecutivo")
            
            summary_data = {
                'Parametro': [
                    'Angolo di tiro ottimale', 'Velocità iniziale calcolata', 'Altezza di lancio effettiva',
                    'Drop al bersaglio', 'Tempo di volo totale', 'Altezza massima raggiunta',
                    'Gittata massima teorica', 'Ritenzione energia finale',
                    'Efficienza integrazione numerica', 'Perdita energia aerodinamica'
                ],
                'Valore': [
                    f"{optimal_angle:.3f}°", f"{main_result.v0:.2f} m/s", f"{params.launch_height:.3f} m",
                    f"{target_drop:.2f} cm", f"{main_result.flight_time:.3f} s", f"{main_result.max_height:.2f} m",
                    f"{main_result.range_distance:.1f} m", f"{energy_retention:.1f}%",
                    f"{efficiency:.1f}%", f"{main_result.energy_loss:.2f} J"
                ],
                'Note Tecniche': [
                    f"Scarto da mira diretta: {optimal_angle - direct_angle:+.2f}°",
                    "Basata su modello energetico arco" if not use_measured_v0 else "Valore inserito dall'utente",
                    "Corretta per geometria posturale arciere" if use_postural_correction else "Altezza neutra",
                    f"Alla distanza di {target_distance}m dal punto di mira",
                    f"Per coprire {target_distance}m di distanza orizzontale",
                    "Apice della parabola balistica",
                    "Distanza teorica impatto al suolo (y=0)",
                    "Percentuale energia cinetica conservata",
                    f"Passi RK4: {stats['steps']}, Rifiutati: {stats['rejections']}",
                    "Energia dissipata per resistenza aerodinamica"
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # Messaggio finale successo
            st.success("🎉 Simulazione balistica completata con successo! "
                      "Tutti i dati sono stati calcolati e sono disponibili per il download.")
            
            
            with st.expander("📚 Note Tecniche e Limitazioni"):
                st.markdown("""
                **Modello Fisico:**
                - Integrazione numerica RK4 con controllo adattivo dell'errore
                - Modello aerodinamico basato su numero di Reynolds variabile
                - Correzioni ambientali per densità aria (temperatura, pressione, umidità)
                - Modello vento semplificato 2D (solo componente orizzontale)
                
                **Assunzioni:**
                - Freccia considerata come corpo rigido puntiforme
                - Traiettoria in piano verticale (no deriva laterale)
                - Condizioni atmosferiche uniformi lungo la traiettoria
                - Resistenza aria proporzionale al quadrato della velocità
                
                **Precisione:**
                - Errore numerico controllato (tolleranza impostata)
                - Validazione attraverso conservazione energia
                - Confronto con modelli analitici semplificati
                
                **Applicabilità:**
                - Tiro con l'arco sportivo e da caccia
                - Distanze tipiche 10-100m
                - Condizioni ambientali moderate
                """)
            
        except Exception as e:
            st.error(f"❌ Errore durante la simulazione: {str(e)}")
            import traceback
            with st.expander("🔍 Dettagli Errore (per debugging)"):
                st.code(traceback.format_exc())
        
        finally:
            # Cleanup progress indicators
            progress_container.empty()


    main()

class SightingSystemCalculator:
    """Calcolatore sistema di mira con geometria 3D completa"""
    
    def __init__(self, eye_to_nock: float, nock_to_riser: float):
        self.eye_to_nock = eye_to_nock
        self.nock_to_riser = nock_to_riser
        self.sight_radius = eye_to_nock + nock_to_riser
    
    def calculate_sight_projection(self, distance: float, drop_meters: float) -> float:
        """Calcola proiezione su riser considerando drop"""
        horizontal_distance = distance
        vertical_offset = drop_meters
        
        correction_angle = np.arctan(vertical_offset / horizontal_distance) if horizontal_distance > 0 else 0
        
        projection_m = horizontal_distance * (self.eye_to_nock + vertical_offset) / (self.nock_to_riser + horizontal_distance)
        projection_cm = projection_m * 100.0 - vertical_offset * 100.0
        
        return projection_cm
    
    def generate_sight_marks(self, distances: np.ndarray, drops_cm: np.ndarray, 
                           trajectory_result: TrajectoryResults, mass_g: float) -> pd.DataFrame:
        """Genera tacche mirino per distanze specifiche CON ENERGIA RESIDUA"""
        sight_data = []
        mass_kg = mass_g / 1000.0
        
        for dist, drop_cm in zip(distances, drops_cm):
            if not np.isnan(drop_cm):
                drop_m = drop_cm / 100.0
                projection = self.calculate_sight_projection(dist, drop_m)
                
                # NUOVO: Calcola energia cinetica residua
                v_x_at_dist = interpolate_trajectory_point(trajectory_result.X, trajectory_result.V_x, dist)
                v_y_at_dist = interpolate_trajectory_point(trajectory_result.X, trajectory_result.V_y, dist)
                v_total_at_dist = np.sqrt(v_x_at_dist**2 + v_y_at_dist**2)
                kinetic_energy = 0.5 * mass_kg * v_total_at_dist**2
                
                sight_data.append({
                    'Distanza (m)': int(dist),
                    'Drop (cm)': round(drop_cm, 1),
                    'Proiezione riser (cm)': round(projection, 2),
                    'Energia (J)': round(kinetic_energy, 1),  # NUOVA COLONNA
                    'Velocità (m/s)': round(v_total_at_dist, 1)  # BONUS: velocità residua
                })
        
        return pd.DataFrame(sight_data)


def display_enhanced_metrics(main_result, params, optimal_angle, target_distance):
    """Visualizza metriche principali con energia residua"""
    
    # Calcoli preliminari
    mass_kg = params.mass / 1000.0
    direct_angle = np.degrees(np.arctan2(params.target_height - params.launch_height, target_distance))
    
    # NUOVO: Calcola energia residua al bersaglio
    v_x_target = interpolate_trajectory_point(main_result.X, main_result.V_x, target_distance)
    v_y_target = interpolate_trajectory_point(main_result.X, main_result.V_y, target_distance)
    v_final_at_target = np.sqrt(v_x_target**2 + v_y_target**2)
    kinetic_energy_residual = 0.5 * mass_kg * v_final_at_target**2
    initial_kinetic_energy = 0.5 * mass_kg * main_result.v0**2
    energy_retention_pct = (kinetic_energy_residual / initial_kinetic_energy) * 100
    
    # Layout metriche
    metrics_cols = st.columns(6)  # Aumentato da 5 a 6 colonne
    
    with metrics_cols[0]:
        st.metric("Angolo Ottimale", f"{optimal_angle:.2f}°",
                 delta=f"{optimal_angle - direct_angle:+.2f}°")
    
    with metrics_cols[1]:
        st.metric("Velocità Iniziale", f"{main_result.v0:.1f} m/s")
    
    with metrics_cols[2]:
        st.metric("Tempo di Volo", f"{main_result.flight_time:.2f} s")
    
    with metrics_cols[3]:
        target_drop = (params.launch_height + np.tan(np.radians(optimal_angle)) * target_distance -
                      interpolate_trajectory_point(main_result.X, main_result.Y, target_distance)) * 100
        st.metric("Drop al Bersaglio", f"{target_drop:.1f} cm")
    
    with metrics_cols[4]:
        # SOSTITUITO: Energia residua invece di ritenzione percentuale
        st.metric("Energia Residua", 
                 f"{kinetic_energy_residual:.1f} J",
                 delta=f"{energy_retention_pct:.1f}%",
                 help="Energia cinetica all'impatto sul bersaglio")


def create_enhanced_summary(main_result, params, optimal_angle, target_distance):
    """Crea riepilogo esecutivo con energia residua"""
    
    # Calcoli energia
    mass_kg = params.mass / 1000.0
    v_x_target = interpolate_trajectory_point(main_result.X, main_result.V_x, target_distance)
    v_y_target = interpolate_trajectory_point(main_result.X, main_result.V_y, target_distance)
    v_final_at_target = np.sqrt(v_x_target**2 + v_y_target**2)
    kinetic_energy_residual = 0.5 * mass_kg * v_final_at_target**2
    initial_kinetic_energy = 0.5 * mass_kg * main_result.v0**2
    
    # Altri calcoli esistenti
    direct_angle = np.degrees(np.arctan2(params.target_height - params.launch_height, target_distance))
    target_drop = (params.launch_height + np.tan(np.radians(optimal_angle)) * target_distance -
                  interpolate_trajectory_point(main_result.X, main_result.Y, target_distance)) * 100
    energy_retention = (v_final_at_target**2 / main_result.v0**2) * 100
    stats = main_result.integration_stats
    efficiency = (100 * stats['steps'] / (stats['steps'] + stats['rejections']) 
                 if (stats['steps'] + stats['rejections']) > 0 else 0)
    
    summary_data = {
        'Parametro': [
            'Angolo di tiro ottimale', 
            'Velocità iniziale calcolata', 
            'Altezza di lancio effettiva',
            'Drop al bersaglio', 
            'Tempo di volo totale', 
            'Altezza massima raggiunta',
            'Gittata massima teorica',
            'Energia cinetica iniziale',           # NUOVO
            'Energia cinetica residua',           # NUOVO
            'Velocità residua al bersaglio',      # NUOVO
            'Ritenzione energia finale',
            'Efficienza integrazione numerica', 
            'Perdita energia aerodinamica'
        ],
        'Valore': [
            f"{optimal_angle:.3f}°", 
            f"{main_result.v0:.2f} m/s", 
            f"{params.launch_height:.3f} m",
            f"{target_drop:.2f} cm", 
            f"{main_result.flight_time:.3f} s", 
            f"{main_result.max_height:.2f} m",
            f"{main_result.range_distance:.1f} m",
            f"{initial_kinetic_energy:.1f} J",           # NUOVO
            f"{kinetic_energy_residual:.1f} J",         # NUOVO
            f"{v_final_at_target:.2f} m/s",             # NUOVO
            f"{energy_retention:.1f}%",
            f"{efficiency:.1f}%",
            f"{main_result.energy_loss:.2f} J"
        ],
        'Note Tecniche': [
            f"Scarto da mira diretta: {optimal_angle - direct_angle:+.2f}°",
            "Basata su modello energetico arco" if not params.use_measured_v0 else "Valore inserito dall'utente",
            "Corretta per geometria posturale arciere",
            f"Alla distanza di {target_distance}m dal punto di mira",
            f"Per coprire {target_distance}m di distanza orizzontale",
            "Apice della parabola balistica",
            "Distanza teorica impatto al suolo (y=0)",
            "Energia al momento del rilascio dalla corda",                    # NUOVO
            f"Energia all'impatto ({energy_retention:.1f}% della iniziale)", # NUOVO
            f"Velocità conservata: {(v_final_at_target/main_result.v0)*100:.1f}%", # NUOVO
            "Percentuale energia cinetica conservata",
            f"Passi RK4: {stats['steps']}, Rifiutati: {stats['rejections']}",
            "Energia dissipata per resistenza aerodinamica"
        ]
    }
    
    return pd.DataFrame(summary_data)


def export_comprehensive_analysis_with_energy(trajectory_result: TrajectoryResults,
                                            params: SimulationParams,
                                            sight_data: pd.DataFrame,
                                            monte_carlo_stats: Optional[Dict] = None) -> io.BytesIO:
    """Export completo analisi in Excel con energia residua"""
    
    output = io.BytesIO()
    
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet 3: Traiettoria completa CON ENERGIA (campionata per Excel)
            max_points = 500
            step = max(1, len(trajectory_result.X) // max_points)
            
            mass_kg = params.mass / 1000.0
            v_total_array = np.sqrt(trajectory_result.V_x[::step]**2 + trajectory_result.V_y[::step]**2)
            kinetic_energy_array = 0.5 * mass_kg * v_total_array**2
            initial_energy = 0.5 * mass_kg * trajectory_result.v0**2
            energy_retention_array = (kinetic_energy_array / initial_energy) * 100
            
            trajectory_data = pd.DataFrame({
                'Distanza (m)': trajectory_result.X[::step],
                'Altezza (m)': trajectory_result.Y[::step],
                'Velocità X (m/s)': trajectory_result.V_x[::step],
                'Velocità Y (m/s)': trajectory_result.V_y[::step],
                'Velocità totale (m/s)': v_total_array,
                'Tempo (s)': trajectory_result.times[::step],
                'Energia cinetica (J)': kinetic_energy_array,
                'Ritenzione energia (%)': energy_retention_array,
                'Perdita energia (J)': initial_energy - kinetic_energy_array
            })
            trajectory_data.to_excel(writer, sheet_name='Traiettoria_Completa', index=False)
            
            # Sheet 4: Scala mirino
            try:
                sight_data.to_excel(writer, sheet_name='Scala_Mirino', index=False)
            except Exception:
                pass
            
            # Sheet MonteCarlo (opzionale)
            if monte_carlo_stats:
                mc_df = pd.DataFrame(monte_carlo_stats)
                mc_df.to_excel(writer, sheet_name='MonteCarlo', index=False)
        
        output.seek(0)
        return output

    except Exception as e:
        logger.error(f"Errore export Excel: {e}")
        # Fallback CSV
        output = io.BytesIO()
        try:
            if isinstance(sight_data, pd.DataFrame):
                sight_data.to_csv(output, index=False)
            else:
                output.write(b"")
        except Exception:
            output.write(b"")
        output.seek(0)
        return output

    except Exception as e:
        logger.error(f"Errore export Excel: {e}")
        # Fallback CSV
        output = io.BytesIO()
        try:
            sight_data.to_csv(output, index=False)
        except Exception:
            output.write(b"")
        output.seek(0)
        return output



if __name__ == '__main__':
    main()
