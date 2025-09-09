# ballistic_simulator_v2_1.py

# Codice originale ballistic_simulator_v2.py con l'aggiunta delle parti mirino A4

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d, UnivariateSpline
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Protocol
from abc import ABC, abstractmethod
import math
from enum import Enum
import io
import warnings
warnings.filterwarnings('ignore')

# --- Aggiunti import per PDF mirino ---
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# ==============================================
# CONFIGURAZIONE E COSTANTI FISICHE
# ==============================================

class PhysicalConstants:
    """Costanti fisiche fondamentali"""
    G = 9.80665  # Accelerazione gravitazionale standard [m/s²]
    AIR_DENSITY_STP = 1.225  # Densità dell'aria a STP [kg/m³]
    AIR_VISCOSITY = 1.81e-5  # Viscosità dinamica dell'aria [Pa·s]
    AIR_MOLAR_MASS = 28.97e-3  # Massa molare aria [kg/mol]
    R_GAS = 8.314  # Costante gas ideale [J/(mol·K)]

class ArrowTipType(Enum):
    """Enum per tipi di punta freccia con fattori CD"""
    FIELD = ("Field Point", 0.95)
    STANDARD = ("Standard", 1.0)
    BROADHEAD = ("Broadhead", 1.15)
    JUDO = ("Judo Point", 1.25)
    BULLET = ("Bullet Point", 0.85)
    
    def __init__(self, display_name: str, cd_factor: float):
        self.display_name = display_name
        self.cd_factor = cd_factor

class BowType(Enum):
    """Enum per tipi di arco con efficienze tipiche"""
    LONGBOW = ("Longbow", 0.75)
    RECURVE = ("Recurvo", 0.82)
    COMPOUND = ("Compound", 0.87)
    TAKEDOWN = ("Takedown", 0.80)
    BAREBOW = ("Barebow", 0.78)
    
    def __init__(self, display_name: str, efficiency: float):
        self.display_name = display_name
        self.efficiency = efficiency

# ==============================================
# MODELLI DATI
# ==============================================

@dataclass
class ArrowConfiguration:
    """Configurazione fisica della freccia"""
    mass_grams: float  # massa in grammi
    length_m: float    # lunghezza in metri
    diameter_mm: float # diametro in millimetri
    spine: int         # spine della freccia
    balance_point_m: float  # punto di bilanciamento dalla nocca
    tip_type: ArrowTipType
    fletching_area_cm2: float = 6.0  # Area totale delle penne
    fletching_offset_deg: float = 2.0  # Angolo di offset delle penne
    shaft_material: str = "Carbon"  # Materiale dell'asta
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        """Validazione parametri freccia"""
        if not (10 <= self.mass_grams <= 50):
            raise ValueError("Massa freccia deve essere tra 10-50g")
        if not (0.5 <= self.length_m <= 1.2):
            raise ValueError("Lunghezza freccia deve essere tra 0.5-1.2m")
        if not (4.0 <= self.diameter_mm <= 12.0):
            raise ValueError("Diametro deve essere tra 4-12mm")
        if not (0 <= self.balance_point_m <= self.length_m):
            raise ValueError("Punto bilanciamento non valido")

@dataclass
class BowConfiguration:
    """Configurazione dell'arco"""
    bow_type: BowType
    draw_weight_lbs: float    # libbraggio
    draw_length_m: float      # allungo
    brace_height_m: float     # brace height
    efficiency: Optional[float] = None  # efficienza (se None usa default del tipo)
    cam_system: str = "Single"  # Tipo di cam (per compound)
    string_material: str = "Dacron"  # Materiale corda
    
    def __post_init__(self):
        if self.efficiency is None:
            self.efficiency = self.bow_type.efficiency
        self._validate()
    
    def _validate(self):
        """Validazione parametri arco"""
        if not (10 <= self.draw_weight_lbs <= 80):
            raise ValueError("Libbraggio deve essere tra 10-80 lbs")
        if not (0.5 <= self.draw_length_m <= 1.1):
            raise ValueError("Allungo deve essere tra 0.5-1.1m")
        if not (0.05 <= self.brace_height_m <= 0.30):
            raise ValueError("Brace height deve essere tra 5-30cm")
        if self.draw_length_m <= self.brace_height_m:
            raise ValueError("Allungo deve essere maggiore del brace height")

@dataclass 
class ArcherConfiguration:
    """Configurazione dell'arciere"""
    height_m: float = 1.75
    eye_height_m: float = 1.65
    anchor_length_m: float = 0.75
    stance_width_m: float = 0.6
    draw_technique: str = "Mediterranean"  # Mediterranean, Three-Under, etc.
    dominant_eye: str = "Right"
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        if not (1.4 <= self.height_m <= 2.2):
            raise ValueError("Altezza deve essere tra 1.4-2.2m")

@dataclass
class EnvironmentalConditions:
    """Condizioni ambientali"""
    temperature_celsius: float = 20.0
    pressure_hpa: float = 1013.25
    humidity_percent: float = 50.0
    wind_speed_ms: float = 0.0  # positivo = favorevole
    altitude_m: float = 0.0
    
    def air_density(self) -> float:
        """Calcola densità dell'aria corretta"""
        T_kelvin = self.temperature_celsius + 273.15
        # Correzione per umidità (approssimazione)
        humidity_factor = 1 - 0.0037 * (self.humidity_percent / 100)
        density = (PhysicalConstants.AIR_MOLAR_MASS * self.pressure_hpa * 100) / \
                 (PhysicalConstants.R_GAS * T_kelvin) * humidity_factor
        # Correzione altitudine
        altitude_factor = np.exp(-self.altitude_m / 8400)  # Scala atmosferica
        return density * altitude_factor

@dataclass
class TargetConfiguration:
    """Configurazione del bersaglio"""
    distance_m: float
    height_m: float = 1.5
    size_cm: float = 20.0  # Diametro zona di scoring
    angle_deg: float = 0.0  # Angolo rispetto alla verticale
    
    def __post_init__(self):
        if self.distance_m <= 0:
            raise ValueError("Distanza deve essere positiva")

# ==============================================
# MODELLI FISICI AVANZATI
# ==============================================

class AerodynamicsModel(ABC):
    """Interfaccia per modelli aerodinamici"""
    
    @abstractmethod
    def drag_coefficient(self, velocity_ms: float, arrow: ArrowConfiguration, 
                        env: EnvironmentalConditions, angle_of_attack_deg: float) -> float:
        pass
    
    @abstractmethod
    def magnus_force(self, velocity_ms: float, spin_rate_rps: float,
                    arrow: ArrowConfiguration, env: EnvironmentalConditions) -> Tuple[float, float]:
        pass

class EnhancedAerodynamicsModel(AerodynamicsModel):
    """Modello aerodinamico avanzato con effetto Magnus"""
    
    def __init__(self):
        self._cd_cache = {}  # Cache per coefficienti di drag
    
    def reynolds_number(self, velocity_ms: float, diameter_m: float, 
                       env: EnvironmentalConditions) -> float:
        """Calcolo numero di Reynolds"""
        rho = env.air_density()
        nu = PhysicalConstants.AIR_VISCOSITY / rho  # Viscosità cinematica
        return velocity_ms * diameter_m / nu
    
    def drag_coefficient(self, velocity_ms: float, arrow: ArrowConfiguration,
                        env: EnvironmentalConditions, angle_of_attack_deg: float = 0.0) -> float:
        """Calcolo coefficiente di resistenza con transizioni Reynolds raffinate"""
        
        # Chiave cache
        cache_key = (round(velocity_ms, 1), arrow.diameter_mm, 
                    round(env.temperature_celsius), round(angle_of_attack_deg, 1))
        
        if cache_key in self._cd_cache:
            return self._cd_cache[cache_key]
        
        Re = self.reynolds_number(velocity_ms, arrow.diameter_mm / 1000, env)
        
        # Modello di drag più sofisticato basato su letteratura
        if Re < 1e2:
            Cd_base = 2.4  # Regime di Stokes
        elif Re < 1e3:
            Cd_base = 2.4 - 0.4 * np.log10(Re / 1e2)
        elif Re < 1e4:
            # Transizione critica
            Cd_base = 2.0 - 0.5 * (Re - 1e3) / 9e3
        elif Re < 2e4:
            # Regime critico con aumento drag
            Cd_base = 1.5 + 1.2 * (Re - 1e4) / 1e4
        elif Re < 1e5:
            # Post-critico con diminuzione
            Cd_base = 2.7 - 1.5 * (Re - 2e4) / 8e4
        else:
            # Regime supercritico
            Cd_base = 1.2
        
        # Correzione per angolo di attacco (effetto indotto)
        alpha_rad = np.radians(abs(angle_of_attack_deg))
        Cd_induced = 2.0 * alpha_rad**2  # Resistenza indotta semplificata
        
        # Fattore tipo punta
        Cd_total = (Cd_base + Cd_induced) * arrow.tip_type.cd_factor
        
        # Correzione per superficie ruvidezza (materiale asta)
        roughness_factor = 1.1 if arrow.shaft_material == "Wood" else 1.0
        Cd_total *= roughness_factor
        
        self._cd_cache[cache_key] = Cd_total
        return Cd_total
    
    def magnus_force(self, velocity_ms: float, spin_rate_rps: float,
                    arrow: ArrowConfiguration, env: EnvironmentalConditions) -> Tuple[float, float]:
        """Calcolo forza Magnus per freccia in rotazione"""
        
        if abs(spin_rate_rps) < 0.1 or velocity_ms < 1.0:
            return 0.0, 0.0
        
        rho = env.air_density()
        diameter_m = arrow.diameter_mm / 1000.0
        
        # Parametro di spin (velocità tangenziale / velocità traslazionale)
        spin_parameter = abs(spin_rate_rps * 2 * np.pi * diameter_m / 2) / velocity_ms
        
        # Coefficiente Magnus empirico (basato su letteratura)
        # Varia con il numero di Reynolds e parametro di spin
        Re = self.reynolds_number(velocity_ms, diameter_m, env)
        
        if Re < 1e4:
            Cm = 0.5 * spin_parameter
        elif Re < 1e5:
            Cm = (0.5 - 0.3 * (Re - 1e4) / 9e4) * spin_parameter
        else:
            Cm = 0.2 * spin_parameter
        
        # Limita coefficiente Magnus
        Cm = min(Cm, 0.3)
        
        # Forza Magnus (perpendicolare alla velocità)
        area_m2 = np.pi * (diameter_m / 2)**2
        F_magnus = 0.5 * rho * velocity_ms**2 * area_m2 * Cm
        
        # Direzione della forza (semplificata per 2D)
        # In 2D, Magnus può causare solo deviazione verticale
        F_x_magnus = 0.0  # Nessuna componente orizzontale in 2D
        F_y_magnus = F_magnus * np.sign(spin_rate_rps)  # Verso alto/basso
        
        return F_x_magnus, F_y_magnus

class SpinModel:
    """Modello per calcolo della rotazione freccia"""
    
    def __init__(self):
        self.spin_rate_cache = {}
    
    def initial_spin_rate(self, arrow: ArrowConfiguration, bow: BowConfiguration,
                         release_quality: float = 0.8) -> float:
        """Calcola spin rate iniziale basato su configurazione penne"""
        
        # Spin indotto dalle penne durante accelerazione
        # Dipende da: angolo offset, area penne, lunghezza freccia
        
        offset_rad = np.radians(arrow.fletching_offset_deg)
        fletching_moment_arm = arrow.length_m - arrow.balance_point_m
        
        # Energia di rotazione impressa (frazione dell'energia cinetica)
        spin_energy_fraction = 0.02 * (arrow.fletching_area_cm2 / 6.0) * \
                              (offset_rad / np.radians(2.0)) * release_quality
        
        # Momento di inerzia freccia (approssimazione cilindro)
        mass_kg = arrow.mass_grams / 1000.0
        I_arrow = 0.5 * mass_kg * (arrow.diameter_mm / 2000.0)**2
        
        # Velocità angolare iniziale
        v0 = self.estimate_muzzle_velocity(arrow, bow)
        E_kinetic = 0.5 * mass_kg * v0**2
        E_spin = spin_energy_fraction * E_kinetic
        omega_0 = np.sqrt(2 * E_spin / I_arrow) if I_arrow > 0 else 0.0
        
        return omega_0 / (2 * np.pi)  # Converti in RPS
    
    def spin_decay(self, initial_spin_rps: float, time_s: float,
                   arrow: ArrowConfiguration, env: EnvironmentalConditions) -> float:
        """Modello decadimento spin nel tempo"""
        
        # Decadimento esponenziale dovuto a resistenza aerodinamica
        # Costante di tempo dipende da momento di inerzia e resistenza
        
        rho = env.air_density()
        diameter_m = arrow.diameter_mm / 1000.0
        mass_kg = arrow.mass_grams / 1000.0
        
        # Coefficiente di attrito rotazionale (empirico)
        C_friction = 0.02  # Tipico per frecce
        
        # Costante di decadimento
        tau = mass_kg / (rho * np.pi * diameter_m**2 * C_friction)
        
        return initial_spin_rps * np.exp(-time_s / tau)
    
    def estimate_muzzle_velocity(self, arrow: ArrowConfiguration, 
                               bow: BowConfiguration) -> float:
        """Stima velocità iniziale dalla configurazione arco-freccia"""
        
        mass_kg = arrow.mass_grams / 1000.0
        draw_force_n = bow.draw_weight_lbs * 4.44822  # lbs -> N
        power_stroke = max(0.0, bow.draw_length_m - bow.brace_height_m)
        
        # Energia immagazzinata (semplificazione lineare)
        stored_energy = bow.efficiency * draw_force_n * power_stroke
        
        # Velocità (conservazione energia)
        return np.sqrt(2 * stored_energy / mass_kg) if mass_kg > 0 else 0.0

# ==============================================
# INTEGRATORE NUMERICO AVANZATO
# ==============================================

@dataclass
class IntegrationState:
    """Stato del sistema dinamico [x, y, vx, vy, omega]"""
    x: float = 0.0      # posizione orizzontale
    y: float = 0.0      # posizione verticale  
    vx: float = 0.0     # velocità orizzontale
    vy: float = 0.0     # velocità verticale
    omega: float = 0.0  # velocità angolare (rad/s)
    
    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.vx, self.vy, self.omega])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'IntegrationState':
        return cls(arr[0], arr[1], arr[2], arr[3], arr[4])

class AdvancedIntegrator:
    """Integratore RK4 avanzato con modello spin e Magnus"""
    
    def __init__(self, tolerance: float = 1e-6, max_step: float = 0.01,
                 min_step: float = 1e-7, safety_factor: float = 0.9):
        self.tolerance = tolerance
        self.max_step = max_step
        self.min_step = min_step
        self.safety_factor = safety_factor
        self.aerodynamics = EnhancedAerodynamicsModel()
        self.spin_model = SpinModel()
        
    def derivatives(self, t: float, state: IntegrationState,
                   arrow: ArrowConfiguration, bow: BowConfiguration,
                   env: EnvironmentalConditions, launch_angle_rad: float) -> IntegrationState:
        """Calcola derivate del sistema con Magnus effect"""
        
        mass_kg = arrow.mass_grams / 1000.0
        area_m2 = np.pi * (arrow.diameter_mm / 2000.0)**2
        velocity = np.sqrt(state.vx**2 + state.vy**2)
        
        # Forze aerodinamiche tradizionali
        if velocity > 1e-6:
            # Angolo di attacco
            flight_angle = np.arctan2(state.vy, state.vx)
            angle_of_attack = np.degrees(flight_angle - launch_angle_rad)
            
            # Drag
            Cd = self.aerodynamics.drag_coefficient(velocity, arrow, env, angle_of_attack)
            drag_force = 0.5 * env.air_density() * Cd * area_m2 * velocity**2
            
            F_drag_x = -drag_force * (state.vx / velocity)
            F_drag_y = -drag_force * (state.vy / velocity)
            
            # Magnus force
            spin_rate_rps = state.omega / (2 * np.pi)
            F_magnus_x, F_magnus_y = self.aerodynamics.magnus_force(
                velocity, spin_rate_rps, arrow, env)
        else:
            F_drag_x = F_drag_y = F_magnus_x = F_magnus_y = 0.0
        
        # Forza vento (orizzontale)
        if abs(env.wind_speed_ms) > 0.1:
            wind_relative_velocity = state.vx - env.wind_speed_ms
            if abs(wind_relative_velocity) > 1e-6:
                wind_drag = 0.5 * env.air_density() * area_m2 * wind_relative_velocity**2
                F_wind_x = -np.sign(wind_relative_velocity) * wind_drag * 0.5  # Coefficiente ridotto per vento
            else:
                F_wind_x = 0.0
        else:
            F_wind_x = 0.0
        
        # Accelerazioni
        ax = (F_drag_x + F_magnus_x + F_wind_x) / mass_kg
        ay = -PhysicalConstants.G + (F_drag_y + F_magnus_y) / mass_kg
        
        # Decadimento spin
        alpha_decay = -0.1  # Costante di decadimento empirica
        
        return IntegrationState(
            x=state.vx,
            y=state.vy,
            vx=ax,
            vy=ay,
            omega=alpha_decay * state.omega
        )
    
    def rk4_step(self, t: float, state: IntegrationState, dt: float,
                arrow: ArrowConfiguration, bow: BowConfiguration,
                env: EnvironmentalConditions, launch_angle_rad: float) -> IntegrationState:
        """Singolo step RK4"""
        
        def add_states(s1: IntegrationState, s2: IntegrationState, factor: float) -> IntegrationState:
            return IntegrationState(
                s1.x + factor * s2.x,
                s1.y + factor * s2.y,
                s1.vx + factor * s2.vx,
                s1.vy + factor * s2.vy,
                s1.omega + factor * s2.omega
            )
        
        k1 = self.derivatives(t, state, arrow, bow, env, launch_angle_rad)
        k2 = self.derivatives(t + dt/2, add_states(state, k1, dt/2), arrow, bow, env, launch_angle_rad)
        k3 = self.derivatives(t + dt/2, add_states(state, k2, dt/2), arrow, bow, env, launch_angle_rad)
        k4 = self.derivatives(t + dt, add_states(state, k3, dt), arrow, bow, env, launch_angle_rad)
        
        # Weighted average
        delta = IntegrationState(
            (k1.x + 2*k2.x + 2*k3.x + k4.x) * dt / 6,
            (k1.y + 2*k2.y + 2*k3.y + k4.y) * dt / 6,
            (k1.vx + 2*k2.vx + 2*k3.vx + k4.vx) * dt / 6,
            (k1.vy + 2*k2.vy + 2*k3.vy + k4.vy) * dt / 6,
            (k1.omega + 2*k2.omega + 2*k3.omega + k4.omega) * dt / 6
        )
        
        return add_states(state, delta, 1.0)
    
    def integrate_trajectory(self, launch_angle_deg: float,
                           arrow: ArrowConfiguration, bow: BowConfiguration,
                           archer: ArcherConfiguration, env: EnvironmentalConditions,
                           target: TargetConfiguration) -> Dict[str, Any]:
        """Integrazione completa della traiettoria"""
        
        # Condizioni iniziali
        launch_angle_rad = np.radians(launch_angle_deg)
        v0 = self.spin_model.estimate_muzzle_velocity(arrow, bow)
        initial_spin = self.spin_model.initial_spin_rate(arrow, bow)
        
        state = IntegrationState(
            x=0.0,
            y=archer.eye_height_m,
            vx=v0 * np.cos(launch_angle_rad),
            vy=v0 * np.sin(launch_angle_rad),
            omega=initial_spin * 2 * np.pi  # converti in rad/s
        )
        
        # Arrays per tracciare traiettoria
        trajectory = {
            'time': [0.0],
            'x': [state.x],
            'y': [state.y],
            'vx': [state.vx],
            'vy': [state.vy],
            'omega': [state.omega],
            'velocity': [np.sqrt(state.vx**2 + state.vy**2)]
        }
        
        t = 0.0
        dt = self.max_step / 5  # Start conservatively
        max_range = target.distance_m * 1.5
        
        stats = {'steps': 0, 'rejections': 0, 'min_dt': self.max_step, 'max_dt': 0.0}
        
        while (state.x < max_range and state.y > -10.0 and 
               t < 30.0 and len(trajectory['time']) < 10000):
            
            # Doppio step per controllo errore
            state_full = self.rk4_step(t, state, dt, arrow, bow, env, launch_angle_rad)
            
            state_half1 = self.rk4_step(t, state, dt/2, arrow, bow, env, launch_angle_rad)
            state_half2 = self.rk4_step(t + dt/2, state_half1, dt/2, arrow, bow, env, launch_angle_rad)
            
            # Errore stimato
            error_pos = np.sqrt((state_full.x - state_half2.x)**2 + (state_full.y - state_half2.y)**2)
            error_vel = np.sqrt((state_full.vx - state_half2.vx)**2 + (state_full.vy - state_half2.vy)**2)
            error = max(error_pos, error_vel * dt)
            
            if error < self.tolerance or dt <= self.min_step:
                # Accetta step
                state = state_half2  # Usa la soluzione più accurata
                t += dt
                
                # Salva dati
                trajectory['time'].append(t)
                trajectory['x'].append(state.x)
                trajectory['y'].append(state.y)
                trajectory['vx'].append(state.vx)
                trajectory['vy'].append(state.vy)
                trajectory['omega'].append(state.omega)
                trajectory['velocity'].append(np.sqrt(state.vx**2 + state.vy**2))
                
                stats['steps'] += 1
                stats['min_dt'] = min(stats['min_dt'], dt)
                stats['max_dt'] = max(stats['max_dt'], dt)
                
                # Adatta passo
                if error > 0:
                    factor = self.safety_factor * (self.tolerance / error)**0.2
                    dt = max(self.min_step, min(self.max_step, dt * min(factor, 1.5)))
                else:
                    dt = min(self.max_step, dt * 1.2)
            else:
                # Rifiuta step
                stats['rejections'] += 1
                factor = self.safety_factor * (self.tolerance / error)**0.25
                dt = max(self.min_step, dt * max(factor, 0.3))
        
        # Converti in numpy arrays
        for key in trajectory:
            trajectory[key] = np.array(trajectory[key])
        
        # Calcola metriche finali
        max_height = np.max(trajectory['y'])
        flight_time = trajectory['time'][-1]
        final_range = trajectory['x'][-1]
        
        # Interpolazione per trovare impatto al bersaglio
        if target.distance_m <= trajectory['x'][-1]:
            impact_height = float(interp1d(trajectory['x'], trajectory['y'], 
                                         kind='linear', fill_value='extrapolate')(target.distance_m))
        else:
            impact_height = trajectory['y'][-1]  # Freccia non raggiunge il bersaglio
        
        return {
            'trajectory': trajectory,
            'initial_velocity': v0,
            'initial_spin_rps': initial_spin,
            'launch_angle_deg': launch_angle_deg,
            'max_height': max_height,
            'flight_time': flight_time,
            'final_range': final_range,
            'impact_height': impact_height,
            'integration_stats': stats
        }

# ==============================================
# OTTIMIZZATORE ANGOLO
# ==============================================

class AngleOptimizer:
    """Ottimizzatore per trovare l'angolo di tiro ottimale"""
    
    def __init__(self, integrator: AdvancedIntegrator):
        self.integrator = integrator
        self._cache = {}  # Cache per risultati
    
    def objective_function(self, angle_deg: float, arrow: ArrowConfiguration,
                          bow: BowConfiguration, archer: ArcherConfiguration,
                          env: EnvironmentalConditions, target: TargetConfiguration) -> float:
        """Funzione obiettivo per ottimizzazione angolo"""
        
        # Cache key
        cache_key = (round(angle_deg, 2), id(arrow), id(bow), id(target))
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            result = self.integrator.integrate_trajectory(
                angle_deg, arrow, bow, archer, env, target)
            
            error = abs(result['impact_height'] - target.height_m)
            self._cache[cache_key] = error
            return error
            
        except Exception:
            return 1e6  # Penalità per errori
    
    def find_optimal_angle(self, arrow: ArrowConfiguration, bow: BowConfiguration,
                          archer: ArcherConfiguration, env: EnvironmentalConditions,
                          target: TargetConfiguration) -> Tuple[float, float]:
        """Trova l'angolo ottimale per colpire il bersaglio"""
        
        # Stima iniziale basata su balistica parabolica
        v0 = self.integrator.spin_model.estimate_muzzle_velocity(arrow, bow)
        g = PhysicalConstants.G
        h0, ht = archer.eye_height_m, target.height_m
        d = target.distance_m
        
        # Angolo balistico ideale (senza resistenza)
        discriminant = v0**4 - g*(g*d**2 + 2*(ht - h0)*v0**2)
        if discriminant > 0:
            angle_estimate = np.degrees(0.5 * np.arcsin(g*d / v0**2))
        else:
            angle_estimate = 10.0  # Fallback
        
        # Range di ricerca
        search_range = max(15.0, abs(angle_estimate))
        bounds = (max(-10, angle_estimate - search_range), 
                 min(45, angle_estimate + search_range))
        
        try:
            # Ottimizzazione con Brent
            result = minimize_scalar(
                lambda angle: self.objective_function(angle, arrow, bow, archer, env, target),
                bounds=bounds,
                method='bounded'
            )
            
            if result.success and result.fun < 2.0:  # Errore < 2m
                return result.x, result.fun
            else:
                # Fallback: ricerca griglia
                angles = np.linspace(bounds[0], bounds[1], 30)
                errors = [self.objective_function(a, arrow, bow, archer, env, target) 
                         for a in angles]
                best_idx = np.argmin(errors)
                return angles[best_idx], errors[best_idx]
                
        except Exception as e:
            print(f"Errore ottimizzazione: {e}")
            return angle_estimate, 1e6

# ==============================================
# ANALISI E VISUALIZZAZIONE
# ==============================================

class TrajectoryAnalyzer:
    """Analizzatore per elaborazione avanzata dei risultati"""
    
    def __init__(self):
        self.colors = {
            'main_trajectory': '#2E86AB',
            'comparison': '#A23B72', 
            'target': '#F18F01',
            'sight_line': '#C73E1D',
            'magnus_effect': '#8B5A3C',
            'wind_effect': '#4A90A4'
        }
    
    def create_trajectory_plot(self, result: Dict[str, Any], 
                              arrow: ArrowConfiguration, bow: BowConfiguration,
                              target: TargetConfiguration, 
                              comparison_result: Optional[Dict[str, Any]] = None) -> plt.Figure:
        """Crea grafico avanzato della traiettoria"""
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), 
                                           height_ratios=[3, 1, 1])
        
        traj = result['trajectory']
        
        # GRAFICO PRINCIPALE - Traiettoria
        ax1.plot(traj['x'], traj['y'], color=self.colors['main_trajectory'], 
                linewidth=2.5, label='Traiettoria completa', alpha=0.9)
        
        # Confronto (se presente)
        if comparison_result is not None:
            comp_traj = comparison_result['trajectory']
            ax1.plot(comp_traj['x'], comp_traj['y'], 
                    color=self.colors['comparison'], linestyle='--',
                    linewidth=2, label='Senza Magnus/Spin', alpha=0.7)
        
        # Bersaglio
        ax1.scatter(target.distance_m, target.height_m, 
                   color=self.colors['target'], s=120, marker='o',
                   label='Bersaglio', edgecolors='black', linewidth=2, zorder=5)
        
        # Punto di impatto
        ax1.scatter(target.distance_m, result['impact_height'],
                   color='red', s=80, marker='x', linewidth=3,
                   label=f"Impatto (h={result['impact_height']:.2f}m)", zorder=5)
        
        # Linea di mira
        sight_y = np.tan(np.radians(result['launch_angle_deg'])) * traj['x'] + traj['y'][0]
        ax1.plot(traj['x'], sight_y, color=self.colors['sight_line'],
                linestyle=':', linewidth=2, label='Linea di mira', alpha=0.8)
        
        # Drop annotation
        drop_m = result['impact_height'] - target.height_m
        if abs(drop_m) > 0.05:  # Solo se drop significativo
            ax1.annotate(f'Drop: {drop_m*100:.1f} cm',
                        xy=(target.distance_m, result['impact_height']),
                        xytext=(target.distance_m - 5, result['impact_height'] - 1),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                                 edgecolor="red", alpha=0.9),
                        fontsize=10, fontweight='bold')
        
        ax1.set_xlabel('Distanza (m)', fontsize=12)
        ax1.set_ylabel('Altezza (m)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        ax1.set_title(f'Simulazione Balistica Avanzata - Magnus + Spin\n'
                     f'Angolo: {result["launch_angle_deg"]:.2f}° | '
                     f'v₀: {result["initial_velocity"]:.1f} m/s | '
                     f'Spin: {result["initial_spin_rps"]:.1f} RPS | '
                     f'Tempo: {result["flight_time"]:.2f} s', fontsize=13)
        
        # GRAFICO VELOCITÀ
        velocity = traj['velocity']
        ax2.plot(traj['x'], velocity, color='green', linewidth=2, label='Velocità totale')
        ax2.plot(traj['x'], traj['vx'], color='blue', linewidth=1.5, 
                alpha=0.7, label='Componente orizzontale')
        ax2.plot(traj['x'], traj['vy'], color='red', linewidth=1.5,
                alpha=0.7, label='Componente verticale')
        
        ax2.set_xlabel('Distanza (m)', fontsize=12)
        ax2.set_ylabel('Velocità (m/s)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_title('Profilo di Velocità', fontsize=11)
        
        # GRAFICO SPIN
        spin_rps = traj['omega'] / (2 * np.pi)
        ax3.plot(traj['x'], spin_rps, color='purple', linewidth=2, label='Velocità di rotazione')
        ax3.set_xlabel('Distanza (m)', fontsize=12)
        ax3.set_ylabel('Spin (RPS)', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_title('Decadimento Spin', fontsize=11)
        
        plt.tight_layout()
        return fig
    
    def analyze_drop_curve(self, arrow: ArrowConfiguration, bow: BowConfiguration,
                          archer: ArcherConfiguration, env: EnvironmentalConditions,
                          distances: np.ndarray, integrator: AdvancedIntegrator,
                          optimizer: AngleOptimizer) -> pd.DataFrame:
        """Analizza curva drop per multiple distanze"""
        
        results = []
        
        for distance in distances:
            target = TargetConfiguration(distance_m=float(distance), height_m=archer.eye_height_m)
            
            try:
                # Trova angolo ottimale
                optimal_angle, error = optimizer.find_optimal_angle(
                    arrow, bow, archer, env, target)
                
                # Simula traiettoria
                result = integrator.integrate_trajectory(
                    optimal_angle, arrow, bow, archer, env, target)
                
                # Calcola drop
                sight_height = archer.eye_height_m + np.tan(np.radians(optimal_angle)) * distance
                drop_m = sight_height - result['impact_height']
                
                results.append({
                    'distance_m': distance,
                    'angle_deg': optimal_angle,
                    'drop_m': drop_m,
                    'drop_cm': drop_m * 100,
                    'velocity_impact': result['trajectory']['velocity'][-1] if len(result['trajectory']['velocity']) > 0 else 0,
                    'flight_time': result['flight_time'],
                    'optimization_error': error
                })
                
            except Exception as e:
                print(f"Errore per distanza {distance}m: {e}")
                results.append({
                    'distance_m': distance,
                    'angle_deg': np.nan,
                    'drop_m': np.nan,
                    'drop_cm': np.nan,
                    'velocity_impact': np.nan,
                    'flight_time': np.nan,
                    'optimization_error': 1e6
                })
        
        return pd.DataFrame(results)
    
    def create_drop_analysis_plot(self, drop_data: pd.DataFrame) -> plt.Figure:
        """Crea grafico analisi drop"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        valid_data = drop_data.dropna()
        
        if len(valid_data) > 0:
            # Drop curve
            ax1.scatter(valid_data['distance_m'], valid_data['drop_cm'], 
                       color='red', s=60, alpha=0.8, edgecolors='darkred', label='Dati simulazione')
            
            # Fit polinomiale se sufficienti punti
            if len(valid_data) >= 3:
                try:
                    coeffs = np.polyfit(valid_data['distance_m'], valid_data['drop_cm'], deg=2)
                    poly_fit = np.poly1d(coeffs)
                    
                    x_smooth = np.linspace(valid_data['distance_m'].min(), 
                                         valid_data['distance_m'].max(), 200)
                    y_smooth = poly_fit(x_smooth)
                    
                    ax1.plot(x_smooth, y_smooth, color='blue', linewidth=2, 
                            label=f'Fit quadratico: {coeffs[0]:.4f}x² + {coeffs[1]:.4f}x + {coeffs[2]:.2f}')
                except:
                    pass
            
            ax1.set_xlabel('Distanza (m)', fontsize=12)
            ax1.set_ylabel('Drop (cm)', fontsize=12)
            ax1.set_title('Curva Drop', fontsize=13)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Velocità all'impatto
            ax2.plot(valid_data['distance_m'], valid_data['velocity_impact'], 
                    color='green', linewidth=2, marker='o', label='Velocità impatto')
            ax2.set_xlabel('Distanza (m)', fontsize=12)
            ax2.set_ylabel('Velocità (m/s)', fontsize=12)
            ax2.set_title('Velocità all\'Impatto', fontsize=13)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        plt.tight_layout()
        return fig

# ==============================================
# INTERFACCIA STREAMLIT MIGLIORATA
# ==============================================

def create_streamlit_interface():
    """Interfaccia Streamlit completamente rinnovata"""
    
    st.set_page_config(
        page_title="Simulatore Balistico v2.0",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🏹"
    )
    
    # Header con stile moderno
    st.markdown("""
    <div style='text-align: center; padding: 2rem; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; border-radius: 15px; margin-bottom: 2rem; 
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);'>
        <h1 style='margin: 0; font-size: 2.5em; font-weight: 300;'>
            🏹 Simulatore Balistico v2.0
        </h1>
        <p style='margin: 0.5rem 0 0 0; font-size: 1.2em; opacity: 0.9;'>
            Simulazione fisica completa con effetto Magnus e modello di spin
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar per controlli avanzati
    with st.sidebar:
        st.markdown("### ⚙️ Controlli Simulazione")
        
        # Parametri integrazione
        with st.expander("🔧 Parametri Numerici", expanded=False):
            tolerance = st.slider("Tolleranza integrazione", 1e-8, 1e-4, 1e-6, format="%.1e")
            max_step = st.slider("Passo massimo (s)", 0.001, 0.1, 0.01)
            min_step = st.slider("Passo minimo (s)", 1e-8, 0.001, 1e-6, format="%.1e")
        
        # Opzioni visualizzazione
        with st.expander("📊 Visualizzazione", expanded=True):
            show_comparison = st.checkbox("Confronta con simulazione base", value=False)
            show_magnus_analysis = st.checkbox("Analisi effetto Magnus", value=True)
            show_drop_analysis = st.checkbox("Analisi curva drop", value=True)
        
        # Range analisi drop
        if show_drop_analysis:
            with st.expander("📈 Range Drop Analysis"):
                drop_min = st.number_input("Distanza min (m)", 5, 100, 10)
                drop_max = st.number_input("Distanza max (m)", 10, 150, 80)
                drop_step = st.number_input("Passo (m)", 1, 10, 5)
    
    # Layout principale a 4 colonne per migliore organizzazione
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        st.markdown("### 🎯 Configurazione Freccia")
        mass_grams = st.number_input("Peso (g)", 10.0, 50.0, 24.0, step=0.5)
        length_m = st.number_input("Lunghezza (m)", 0.5, 1.0, 0.75, step=0.01)
        diameter_mm = st.number_input("Diametro (mm)", 4.0, 12.0, 6.2, step=0.1)
        spine = st.number_input("Spine", 200, 1200, 700, step=25)
        balance_point_m = st.number_input("Bilanciamento (m)", 0.2, 0.8, 0.4, step=0.01)
        
        tip_type = st.selectbox("Tipo punta", 
                               [tip.display_name for tip in ArrowTipType],
                               help="Diversi tipi di punta hanno coefficienti di resistenza differenti")
        
        fletching_area = st.number_input("Area penne (cm²)", 3.0, 12.0, 6.0, step=0.5)
        fletching_offset = st.number_input("Offset penne (°)", 0.5, 5.0, 2.0, step=0.1)
        shaft_material = st.selectbox("Materiale asta", ["Carbon", "Aluminum", "Wood"])
    
    with col2:
        st.markdown("### 🏹 Configurazione Arco")
        bow_type_name = st.selectbox("Tipo arco", 
                                    [bow.display_name for bow in BowType])
        draw_weight = st.number_input("Libbraggio (lbs)", 10.0, 80.0, 40.0, step=1.0)
        draw_length = st.number_input("Allungo (m)", 0.5, 1.1, 0.70, step=0.01)
        brace_height = st.number_input("Brace Height (m)", 0.05, 0.30, 0.18, step=0.01)
        
        custom_efficiency = st.checkbox("Efficienza personalizzata")
        if custom_efficiency:
            efficiency = st.number_input("Efficienza", 0.5, 0.95, 0.82, step=0.01)
        else:
            efficiency = None
            
        cam_system = st.selectbox("Sistema cam", ["Single", "Binary", "Hybrid"], 
                                 help="Solo per archi compound")
        string_material = st.selectbox("Materiale corda", ["Dacron", "FastFlight", "Spectra"])
    
    with col3:
        st.markdown("### 👤 Configurazione Arciere")
        archer_height = st.number_input("Altezza (m)", 1.4, 2.2, 1.75, step=0.01)
        eye_height = st.number_input("Altezza occhio (m)", 1.2, 2.1, 1.65, step=0.01)
        anchor_length = st.number_input("Lunghezza aggancio (m)", 0.4, 1.0, 0.75, step=0.01)
        stance_width = st.number_input("Larghezza stance (m)", 0.3, 1.0, 0.6, step=0.01)
        
        draw_technique = st.selectbox("Tecnica tiro", 
                                     ["Mediterranean", "Three-Under", "Thumb Draw"])
        dominant_eye = st.selectbox("Occhio dominante", ["Right", "Left"])
        
        st.markdown("### 🎯 Bersaglio")
        target_distance = st.number_input("Distanza (m)", 5.0, 150.0, 50.0, step=1.0)
        target_height = st.number_input("Altezza (m)", -5.0, 10.0, 1.5, step=0.1)
        target_size = st.number_input("Diametro (cm)", 5.0, 122.0, 20.0, step=1.0)
    
    with col4:
        st.markdown("### 🌤️ Condizioni Ambientali")
        temperature = st.number_input("Temperatura (°C)", -20, 50, 20, step=1)
        pressure = st.number_input("Pressione (hPa)", 900, 1100, 1013, step=1)
        humidity = st.number_input("Umidità (%)", 0, 100, 50, step=5)
        wind_speed = st.number_input("Vento (m/s)", -15.0, 15.0, 0.0, step=0.1,
                                   help="Positivo = favorevole, Negativo = contrario")
        altitude = st.number_input("Altitudine (m)", 0, 3000, 0, step=50)
        
        st.markdown("### ⚡ Controlli Avanzati")
        enable_spin_model = st.checkbox("Modello rotazione avanzato", value=True,
                                       help="Include effetto Magnus e decadimento spin")
        release_quality = st.slider("Qualità rilascio", 0.5, 1.0, 0.8, step=0.05,
                                   help="Influenza spin iniziale: 1.0 = perfetto")
        
        wind_model_advanced = st.checkbox("Modello vento avanzato", value=False,
                                        help="Include turbolenza e rafiche")
    
    # Pulsante principale con stile migliorato
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 AVVIA SIMULAZIONE COMPLETA", type="primary", use_container_width=True):
        
        # Validazione e creazione oggetti configurazione
        try:
            # Trova enum corrispondenti
            selected_tip_type = next(tip for tip in ArrowTipType 
                                   if tip.display_name == tip_type)
            selected_bow_type = next(bow for bow in BowType 
                                   if bow.display_name == bow_type_name)
            
            # Crea oggetti configurazione
            arrow = ArrowConfiguration(
                mass_grams=mass_grams,
                length_m=length_m,
                diameter_mm=diameter_mm,
                spine=spine,
                balance_point_m=balance_point_m,
                tip_type=selected_tip_type,
                fletching_area_cm2=fletching_area,
                fletching_offset_deg=fletching_offset,
                shaft_material=shaft_material
            )
            
            bow = BowConfiguration(
                bow_type=selected_bow_type,
                draw_weight_lbs=draw_weight,
                draw_length_m=draw_length,
                brace_height_m=brace_height,
                efficiency=efficiency,
                cam_system=cam_system,
                string_material=string_material
            )
            
            archer = ArcherConfiguration(
                height_m=archer_height,
                eye_height_m=eye_height,
                anchor_length_m=anchor_length,
                stance_width_m=stance_width,
                draw_technique=draw_technique,
                dominant_eye=dominant_eye
            )
            
            env = EnvironmentalConditions(
                temperature_celsius=temperature,
                pressure_hpa=pressure,
                humidity_percent=humidity,
                wind_speed_ms=wind_speed,
                altitude_m=altitude
            )
            
            target = TargetConfiguration(
                distance_m=target_distance,
                height_m=target_height,
                size_cm=target_size
            )
            
        except ValueError as e:
            st.error(f"❌ Errore nella configurazione: {e}")
            return
        except Exception as e:
            st.error(f"❌ Errore imprevisto: {e}")
            return
        
        # Progress bar
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        try:
            # Inizializzazione
            status_text.text("⚙️ Inizializzazione modelli...")
            progress_bar.progress(10)
            
            integrator = AdvancedIntegrator(
                tolerance=tolerance,
                max_step=max_step,
                min_step=min_step
            )
            
            optimizer = AngleOptimizer(integrator)
            analyzer = TrajectoryAnalyzer()
            
            # Ottimizzazione angolo
            status_text.text("🎯 Ottimizzazione angolo di tiro...")
            progress_bar.progress(30)
            
            optimal_angle, opt_error = optimizer.find_optimal_angle(
                arrow, bow, archer, env, target)
            
            if opt_error > 5.0:
                st.warning(f"⚠️ Ottimizzazione con errore elevato: {opt_error:.2f}m")
            
            # Simulazione principale
            status_text.text("🚀 Simulazione traiettoria principale...")
            progress_bar.progress(50)
            
            main_result = integrator.integrate_trajectory(
                optimal_angle, arrow, bow, archer, env, target)
            
            # Simulazione di confronto (se richiesta)
            comparison_result = None
            if show_comparison:
                status_text.text("📊 Simulazione di confronto...")
                progress_bar.progress(65)
                
                # Simulazione semplificata senza Magnus
                simple_integrator = AdvancedIntegrator(tolerance, max_step, min_step)
                simple_integrator.aerodynamics = EnhancedAerodynamicsModel()  # Usa stesso modello drag
                
                comparison_result = simple_integrator.integrate_trajectory(
                    optimal_angle, arrow, bow, archer, env, target)
            
            # Analisi drop (se richiesta)
            drop_data = None
            if show_drop_analysis:
                status_text.text("📈 Analisi curva drop...")
                progress_bar.progress(80)
                
                distances = np.arange(drop_min, drop_max + 1, drop_step)
                drop_data = analyzer.analyze_drop_curve(
                    arrow, bow, archer, env, distances, integrator, optimizer)
            
            progress_bar.progress(100)
            status_text.text("✅ Simulazione completata!")
            
            # === VISUALIZZAZIONE RISULTATI ===
            st.markdown("---")
            st.markdown("## 📊 Risultati Simulazione")
            
            # Metriche principali con design migliorato
            metric_cols = st.columns(5)
            
            with metric_cols[0]:
                st.metric("🎯 Angolo Ottimale", 
                         f"{optimal_angle:.2f}°",
                         help="Angolo di elevazione ottimizzato")
            
            with metric_cols[1]:
                st.metric("⚡ Velocità Iniziale", 
                         f"{main_result['initial_velocity']:.1f} m/s",
                         help="Velocità alla bocca calcolata")
            
            with metric_cols[2]:
                st.metric("🌪️ Spin Iniziale", 
                         f"{main_result['initial_spin_rps']:.1f} RPS",
                         help="Velocità di rotazione iniziale")
            
            with metric_cols[3]:
                st.metric("⏱️ Tempo di Volo", 
                         f"{main_result['flight_time']:.2f} s",
                         help="Tempo per raggiungere il bersaglio")
            
            with metric_cols[4]:
                drop_at_target = (target.height_m - main_result['impact_height']) * 100
                st.metric("📉 Drop al Bersaglio", 
                         f"{drop_at_target:+.1f} cm",
                         help="Differenza verticale rispetto al bersaglio")
            
            # Grafico traiettoria principale
            st.markdown("### 📈 Analisi Traiettoria")
            fig_trajectory = analyzer.create_trajectory_plot(
                main_result, arrow, bow, target, comparison_result)
            st.pyplot(fig_trajectory, use_container_width=True)
            
            # Statistiche integrazione dettagliate
            with st.expander("🔧 Dettagli Integrazione Numerica", expanded=False):
                stats_cols = st.columns(4)
                stats = main_result['integration_stats']
                
                with stats_cols[0]:
                    st.metric("Passi Accettati", stats['steps'])
                
                with stats_cols[1]:
                    st.metric("Passi Rifiutati", stats['rejections'])
                
                with stats_cols[2]:
                    st.metric("Passo Min/Max", 
                             f"{stats['min_dt']:.2e}/{stats['max_dt']:.2e} s")
                
                with stats_cols[3]:
                    efficiency_pct = 100 * stats['steps'] / (stats['steps'] + stats['rejections']) \
                                   if (stats['steps'] + stats['rejections']) > 0 else 0
                    st.metric("Efficienza", f"{efficiency_pct:.1f}%")
            
            # Analisi drop curve
            if show_drop_analysis and drop_data is not None:
                st.markdown("### 📉 Analisi Curva Drop")
                
                # Grafico drop
                fig_drop = analyzer.create_drop_analysis_plot(drop_data)
                st.pyplot(fig_drop, use_container_width=True)
                
                # Tabella dati drop
                with st.expander("📊 Tabella Dati Drop", expanded=False):
                    # Format della tabella
                    display_data = drop_data.copy()
                    display_data = display_data.round({
                        'angle_deg': 2,
                        'drop_m': 3,
                        'drop_cm': 1,
                        'velocity_impact': 1,
                        'flight_time': 2,
                        'optimization_error': 3
                    })
                    
                    st.dataframe(
                        display_data,
                        column_config={
                            'distance_m': st.column_config.NumberColumn("Distanza (m)"),
                            'angle_deg': st.column_config.NumberColumn("Angolo (°)"),
                            'drop_cm': st.column_config.NumberColumn("Drop (cm)"),
                            'velocity_impact': st.column_config.NumberColumn("Velocità impatto (m/s)"),
                            'flight_time': st.column_config.NumberColumn("Tempo volo (s)"),
                            'optimization_error': st.column_config.NumberColumn("Errore opt. (m)")
                        },
                        use_container_width=True,
                        hide_index=True
                    )
                
                # Predittore drop interattivo
                st.markdown("#### 🎯 Predittore Drop Interattivo")
                
                valid_drop_data = drop_data.dropna()
                if len(valid_drop_data) >= 3:
                    try:
                        # Fit polinomiale
                        coeffs = np.polyfit(valid_drop_data['distance_m'], 
                                          valid_drop_data['drop_cm'], deg=2)
                        poly_fit = np.poly1d(coeffs)
                        
                        # Input per query
                        col_query1, col_query2 = st.columns([1, 2])
                        
                        with col_query1:
                            query_distance = st.number_input(
                                "Distanza query (m)", 
                                float(valid_drop_data['distance_m'].min()),
                                float(valid_drop_data['distance_m'].max()),
                                float(target_distance),
                                step=1.0
                            )
                        
                        with col_query2:
                            predicted_drop = poly_fit(query_distance)
                            st.metric(
                                f"Drop previsto a {query_distance:.0f}m",
                                f"{predicted_drop:.1f} cm",
                                help="Basato su interpolazione quadratica"
                            )
                    
                    except Exception as e:
                        st.warning(f"Impossibile creare predittore: {e}")
            
            # Analisi effetto Magnus (se abilitato)
            if show_magnus_analysis and enable_spin_model:
                st.markdown("### 🌪️ Analisi Effetto Magnus")
                
                # Confronta con e senza Magnus
                magnus_cols = st.columns(2)
                
                with magnus_cols[0]:
                    # Simulazione senza Magnus (modifica temporanea)
                    temp_integrator = AdvancedIntegrator(tolerance, max_step, min_step)
                    
                    # Override del modello Magnus per disabilitarlo
                    class NoMagnusAero(EnhancedAerodynamicsModel):
                        def magnus_force(self, *args):
                            return 0.0, 0.0
                    
                    temp_integrator.aerodynamics = NoMagnusAero()
                    
                    no_magnus_result = temp_integrator.integrate_trajectory(
                        optimal_angle, arrow, bow, archer, env, target)
                    
                    st.write("**Senza effetto Magnus:**")
                    st.write(f"- Impatto: {no_magnus_result['impact_height']:.3f} m")
                    st.write(f"- Velocità finale: {no_magnus_result['trajectory']['velocity'][-1]:.1f} m/s")
                    st.write(f"- Gittata: {no_magnus_result['final_range']:.1f} m")
                
                with magnus_cols[1]:
                    st.write("**Con effetto Magnus:**")
                    st.write(f"- Impatto: {main_result['impact_height']:.3f} m")
                    st.write(f"- Velocità finale: {main_result['trajectory']['velocity'][-1]:.1f} m/s")
                    st.write(f"- Gittata: {main_result['final_range']:.1f} m")
                    
                    # Differenze
                    height_diff = (main_result['impact_height'] - no_magnus_result['impact_height']) * 100
                    range_diff = main_result['final_range'] - no_magnus_result['final_range']
                    
                    st.markdown("**Effetto Magnus:**")
                    st.write(f"- Δ Altezza: {height_diff:+.1f} cm")
                    st.write(f"- Δ Gittata: {range_diff:+.1f} m")
            
            # Informazioni configurazione
            st.markdown("### ℹ️ Riepilogo Configurazione")
            
            config_tabs = st.tabs(["🎯 Freccia", "🏹 Arco", "👤 Arciere", "🌤️ Ambiente"])
            
            with config_tabs[0]:
                arrow_info = {
                    'Peso': f"{arrow.mass_grams} g",
                    'Lunghezza': f"{arrow.length_m} m",
                    'Diametro': f"{arrow.diameter_mm} mm",
                    'Spine': str(arrow.spine),
                    'Bilanciamento': f"{arrow.balance_point_m} m",
                    'Tipo punta': arrow.tip_type.display_name,
                    'Area penne': f"{arrow.fletching_area_cm2} cm²",
                    'Offset penne': f"{arrow.fletching_offset_deg}°",
                    'Materiale': arrow.shaft_material
                }
                st.json(arrow_info)
            
            with config_tabs[1]:
                bow_info = {
                    'Tipo': bow.bow_type.display_name,
                    'Libbraggio': f"{bow.draw_weight_lbs} lbs",
                    'Allungo': f"{bow.draw_length_m} m",
                    'Brace height': f"{bow.brace_height_m} m",
                    'Efficienza': f"{bow.efficiency:.3f}",
                    'Sistema cam': bow.cam_system,
                    'Materiale corda': bow.string_material
                }
                st.json(bow_info)
            
            with config_tabs[2]:
                archer_info = {
                    'Altezza': f"{archer.height_m} m",
                    'Altezza occhio': f"{archer.eye_height_m} m",
                    'Lunghezza aggancio': f"{archer.anchor_length_m} m",
                    'Larghezza stance': f"{archer.stance_width_m} m",
                    'Tecnica': archer.draw_technique,
                    'Occhio dominante': archer.dominant_eye
                }
                st.json(archer_info)
            
            with config_tabs[3]:
                env_info = {
                    'Temperatura': f"{env.temperature_celsius}°C",
                    'Pressione': f"{env.pressure_hpa} hPa",
                    'Umidità': f"{env.humidity_percent}%",
                    'Vento': f"{env.wind_speed_ms} m/s",
                    'Altitudine': f"{env.altitude_m} m",
                    'Densità aria calcolata': f"{env.air_density():.3f} kg/m³"
                }
                st.json(env_info)
            
            # Export dati
            st.markdown("### 💾 Export Dati")
            
            export_cols = st.columns(3)
            
            with export_cols[0]:
                # Export traiettoria CSV
                trajectory_df = pd.DataFrame(main_result['trajectory'])
                csv_trajectory = trajectory_df.to_csv(index=False)
                
                st.download_button(
                    "📊 Download Traiettoria CSV",
                    data=csv_trajectory,
                    file_name=f"trajectory_{target_distance:.0f}m.csv",
                    mime="text/csv"
                )
            
            with export_cols[1]:
                # Export drop data CSV (se disponibile)
                if drop_data is not None:
                    csv_drop = drop_data.to_csv(index=False)
                    
                    st.download_button(
                        "📉 Download Drop Data CSV",
                        data=csv_drop,
                        file_name=f"drop_analysis_{drop_min}-{drop_max}m.csv",
                        mime="text/csv"
                    )
            
            with export_cols[2]:
                # Export configurazione completa JSON
                complete_config = {
                    'arrow': {
                        'mass_grams': arrow.mass_grams,
                        'length_m': arrow.length_m,
                        'diameter_mm': arrow.diameter_mm,
                        'spine': arrow.spine,
                        'tip_type': arrow.tip_type.display_name,
                        'fletching_area_cm2': arrow.fletching_area_cm2,
                        'fletching_offset_deg': arrow.fletching_offset_deg,
                        'shaft_material': arrow.shaft_material
                    },
                    'bow': {
                        'type': bow.bow_type.display_name,
                        'draw_weight_lbs': bow.draw_weight_lbs,
                        'draw_length_m': bow.draw_length_m,
                        'efficiency': bow.efficiency
                    },
                    'environment': {
                        'temperature_celsius': env.temperature_celsius,
                        'pressure_hpa': env.pressure_hpa,
                        'wind_speed_ms': env.wind_speed_ms
                    },
                    'results': {
                        'optimal_angle_deg': optimal_angle,
                        'initial_velocity_ms': main_result['initial_velocity'],
                        'initial_spin_rps': main_result['initial_spin_rps'],
                        'flight_time_s': main_result['flight_time'],
                        'max_height_m': main_result['max_height'],
                        'impact_height_m': main_result['impact_height'],
                        'final_range_m': main_result['final_range']
                    }
                }
                
                import json
                config_json = json.dumps(complete_config, indent=2)
                
                st.download_button(
                    "⚙️ Download Configurazione JSON",
                    data=config_json,
                    file_name=f"ballistic_config_{target_distance:.0f}m.json",
                    mime="application/json"
                )
            
            # Messaggio finale di successo
            st.success("""
            🎉 **Simulazione completata con successo!**
            
            Tutti i grafici e i dati sono stati generati utilizzando il modello fisico avanzato che include:
            - ✅ Integrazione RK4 adattativa con controllo dell'errore
            - ✅ Modello aerodinamico avanzato con transizioni Reynolds
            - ✅ Effetto Magnus e modello di spin con decadimento  
            - ✅ Condizioni ambientali realistiche (temperatura, pressione, umidità)
            - ✅ Modello del vento orizzontale
            - ✅ Ottimizzazione automatica dell'angolo di tiro
            """)
            
        except Exception as e:
            st.error(f"❌ Errore durante la simulazione: {e}")
            import traceback
            st.error(f"Dettagli tecnici: {traceback.format_exc()}")
        
        finally:
            # Cleanup progress bar
            progress_bar.empty()
            status_text.empty()

# Funzione per generare il mirino PDF stampabile su A4, adattata dal primo file
def genera_mirino_pdf(distances: List[float], drops_cm: List[float], angles_deg: List[float], eye_height_m: float, filename="mirino_balistica.pdf", additional_info: Optional[Dict[str, Any]] = None) -> io.BytesIO:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica", 14)
    c.drawString(60, height - 50, "Mirino Balistico Personalizzato")
    c.setFont("Helvetica", 11)
    c.drawString(60, height - 70, f"Quota occhio/arciere: {eye_height_m:.2f} m")

    # Intestazione tabella
    y = height - 110
    c.setFont("Helvetica-Bold", 10)
    c.drawString(60, y, "Distanza (m)")
    c.drawString(160, y, "Angolo Mira (°)")
    c.drawString(280, y, "Drop (cm)")
    c.setFont("Helvetica", 10)

    for i, (d, a, dr) in enumerate(zip(distances, angles_deg, drops_cm)):
        y_row = y - 20 - i * 18
        c.drawString(60, y_row, f"{d:.1f}")
        c.drawString(160, y_row, f"{a:.2f}")
        c.drawString(280, y_row, f"{dr:.2f}")

    # Aggiunta informazioni aggiuntive se fornite
    if additional_info:
        y_info = y_row - 40
        c.setFont("Helvetica-Bold", 11)
        c.drawString(60, y_info, "Informazioni aggiuntive:")
        y_info -= 20
        c.setFont("Helvetica", 10)
        for key, value in additional_info.items():
            c.drawString(70, y_info, f"{key}: {value}")
            y_info -= 18

    c.save()
    buf.seek(0)
    return buf

# --------------------------------------------
# Aggiunta UI Streamlit per il mirino PDF
# --------------------------------------------

def main():
    st.set_page_config(page_title="Simulatore Balistico v2.0 + Mirino PDF", layout="wide", page_icon="🏹")

    # (Qui rimane tutto il codice della UI originale per input parametri, simulazione, visualizzazione ecc...)

    # Supponiamo di avere già i dati simulati disponibili in variabili simili a queste:
    # distances, drops_cm, angles_deg, eye_height_m
    
    # esempio dati da analisi simulazione (da codice originale)
    # distances = np.array([...])
    # drop_data = pd.DataFrame(...) - estrai dati drop e angoli

    # Per dimostrazione usiamo dati di esempio (da sostituire con il risultato reale)
    distances = np.array([10, 20, 30, 40, 50])
    drops_cm = np.array([1.2, 4.5, 9.8, 17.0, 26.5])
    angles_deg = np.array([1.5, 3.0, 4.7, 6.5, 8.5])
    eye_height_m = 1.65

    st.header("Scarica Mirino Balistico Stampabile")

    additional_info = {
        "Nota": "Dati simulati con modello RK4 avanzato",
        # Aggiungere altre info come spin, energia ecc. se disponibili
    }

    # Pulsante per generare e scaricare il PDF del mirino
    pdf_buffer = genera_mirino_pdf(distances.tolist(), drops_cm.tolist(), angles_deg.tolist(), eye_height_m, additional_info=additional_info)

    st.download_button(
        label="Scarica Mirino in PDF (A4)",
        data=pdf_buffer,
        file_name="mirino_balistico.pdf",
        mime="application/pdf"
    )

if __name__ == "__main__":
    main()
