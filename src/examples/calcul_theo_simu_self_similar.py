import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
import pynlo.light.DerivedPulses as DPL
import pynlo.media.fibers.fiber as FIB
import pynlo.interactions.FourWaveMixing.SSFM as SSFM

# --- Définitions des Fonctions de la Phase 1 ---

def calculate_fiber_parameters(MFD_um, D_ps_nm_km, n2_m2_W, gain_dB_m, lambda0_nm):
    lambda0_m = lambda0_nm * 1e-9
    c_m_s = constants.c
    A_eff_m2 = np.pi * (MFD_um * 1e-6 / 2)**2
    print(f"  Surface effective du mode (A_eff): {A_eff_m2:.2e} m^2")
    gamma_W_m = (2 * np.pi * n2_m2_W) / (lambda0_m * A_eff_m2)
    print(f"  Coefficient non-linéaire effectif (gamma): {gamma_W_m:.2e} W^-1 m^-1")
    g_m_inv = gain_dB_m * (np.log(10) / 10)
    print(f"  Coefficient de gain distribué (g): {g_m_inv:.3f} m^-1")
    D_SI_s_m2 = D_ps_nm_km * 1e-6
    beta2_SI_s2_m = - (D_SI_s_m2 * lambda0_m**2) / (2 * np.pi * c_m_s)
    print(f"  Paramètre D fourni: {D_ps_nm_km} ps/(nm*km)")
    print(f"  Beta2 calculé en unités SI (beta2_SI): {beta2_SI_s2_m:.2e} s^2/m")
    beta2_for_pynlo_ps2_km = beta2_SI_s2_m * 1e27
    if beta2_SI_s2_m > 0:
        print(f"  Dispersion Normale (beta2 > 0), compatible avec la théorie de Kruglov.")
    else:
        print(f"  ATTENTION: Dispersion Anormale (beta2 < 0). La théorie de Kruglov pour les impulsions paraboliques auto-similaires ne s'applique pas directement. Attendez-vous à une dynamique de type soliton.")
    return A_eff_m2, gamma_W_m, g_m_inv, beta2_SI_s2_m, beta2_for_pynlo_ps2_km

def calculate_input_pulse_params(P_avg_W, f_rep_Hz, delta_lambda_FWHM_nm, lambda0_nm, GDD_input_fs2):
    U_in_reel_J = P_avg_W / f_rep_Hz
    print(f"  Énergie de l'impulsion d'entrée (U_in_reel): {U_in_reel_J * 1e12:.2f} pJ ({U_in_reel_J * 1e9:.2f} nJ)")
    lambda0_m = lambda0_nm * 1e-9
    delta_lambda_FWHM_m = delta_lambda_FWHM_nm * 1e-9
    c_m_s = constants.c
    delta_nu_FWHM_Hz = (c_m_s / lambda0_m**2) * delta_lambda_FWHM_m
    if delta_nu_FWHM_Hz == 0:
        print("  ATTENTION: Largeur spectrale FWHM nulle, la durée limitée par transformée sera infinie.")
        delta_t_TL_s = np.inf
    else:
        delta_t_TL_s = 0.441 / delta_nu_FWHM_Hz 
    print(f"  Largeur spectrale FWHM (delta_nu_FWHM): {delta_nu_FWHM_Hz*1e-12:.2f} THz")
    print(f"  Durée FWHM limitée par transformée (delta_t_TL): {delta_t_TL_s * 1e15:.2f} fs")
    GDD_input_ps2 = GDD_input_fs2 * 1e-6 
    print(f"  GDD de l'impulsion d'entrée (phi2): {GDD_input_ps2:.4e} ps^2 ({GDD_input_fs2:.2f} fs^2)")
    return U_in_reel_J, delta_t_TL_s, GDD_input_ps2

def kruglov_optimal_params(U_in_J, g_m_inv, gamma_W_m, beta2_SI_s2_m, N_ss=100):
    if g_m_inv <= 0: print("  ERREUR: Le coefficient de gain 'g' doit être positif pour la théorie de Kruglov."); return np.nan, np.nan, np.nan
    if gamma_W_m <= 0: print("  ERREUR: Le coefficient non-linéaire 'gamma' doit être positif."); return np.nan, np.nan, np.nan
    if beta2_SI_s2_m <= 0: 
        print("  ERREUR KRUGLOV: Beta2 doit être positif (dispersion normale) pour la formation d'impulsions paraboliques auto-similaires selon Kruglov.")
        print("                   Les paramètres optimaux de Kruglov ne peuvent pas être calculés.")
        return np.nan, np.nan, np.nan
    term_sqrt_A0 = np.sqrt(gamma_W_m * beta2_SI_s2_m / 2.0)
    if term_sqrt_A0 == 0: print("  ERREUR: Terme sous la racine pour A0 est nul."); return np.nan, np.nan, np.nan
    A0 = 0.5 * ( (g_m_inv * U_in_J) / term_sqrt_A0 )**(1/3)
    print(f"  Amplitude asymptotique normalisée (A0): {A0:.2e} (unités Kruglov, liées à sqrt(W))")
    Tp0_opt_s = (6 * term_sqrt_A0 * A0) / g_m_inv
    delta_TFWHM_opt_s = Tp0_opt_s 
    print(f"  Durée FWHM initiale 'optimale' de Kruglov pour U_in (delta_TFWHM_opt): {delta_TFWHM_opt_s * 1e15:.2f} fs")
    log_argument = (N_ss * g_m_inv) / (6 * gamma_W_m * A0**2)
    if log_argument <= 0: print(f"  AVERTISSEMENT: Argument du logarithme pour z_c non positif ({log_argument:.2e}). z_c ne peut être calculé."); z_c_m = np.nan
    else: z_c_m = (3 / (2 * g_m_inv)) * np.log(log_argument)
    print(f"  Longueur de fibre caractéristique de Kruglov (z_c) pour N_ss={N_ss}: {z_c_m:.3f} m")
    return A0, delta_TFWHM_opt_s, z_c_m

# --- Fonctions Utilitaires pour les Graphiques (Phase 2) ---
def calculate_fwhm_with_interp(x_axis, y_intensity):
    """Calcule la FWHM d'un profil avec interpolation."""
    if len(y_intensity) == 0 or np.all(y_intensity <= 1e-20) or len(x_axis) < 2:
        return 0.0

    y_max = np.max(y_intensity)
    if y_max == 0: return 0.0
    half_max = y_max / 2.0

    # Trouver les indices où l'intensité croise half_max
    try:
        # Recherche des indices où y_intensity est supérieur ou égal à half_max
        above_half_indices = np.where(y_intensity >= half_max)[0]
        if len(above_half_indices) < 2: # Pas assez de points pour définir une FWHM
            return 0.0

        # Interpolation pour x1 (bord gauche de la FWHM)
        first_above_idx = above_half_indices[0]
        if first_above_idx == 0 or y_intensity[first_above_idx] == half_max: # Le premier point est exactement half_max ou le début du tableau
            x1 = x_axis[first_above_idx]
        else: # Interpoler entre le point avant et le premier point au-dessus
            x_interp_pts = [x_axis[first_above_idx - 1], x_axis[first_above_idx]]
            y_interp_pts = [y_intensity[first_above_idx - 1], y_intensity[first_above_idx]]
            # S'assurer que les y_interp_pts encadrent half_max et ne sont pas égaux (évite division par zéro)
            if y_interp_pts[0] < half_max < y_interp_pts[1] and y_interp_pts[1] != y_interp_pts[0]:
                 x1 = np.interp(half_max, y_interp_pts, x_interp_pts)
            else: # Fallback si l'interpolation n'est pas simple
                 x1 = x_axis[first_above_idx]


        # Interpolation pour x2 (bord droit de la FWHM)
        last_above_idx = above_half_indices[-1]
        if last_above_idx == len(x_axis) - 1 or y_intensity[last_above_idx] == half_max: # Le dernier point est exactement half_max ou la fin du tableau
            x2 = x_axis[last_above_idx]
        else: # Interpoler entre le dernier point au-dessus et le point après
            x_interp_pts = [x_axis[last_above_idx], x_axis[last_above_idx + 1]]
            y_interp_pts = [y_intensity[last_above_idx], y_intensity[last_above_idx + 1]]
            if y_interp_pts[1] < half_max < y_interp_pts[0] and y_interp_pts[0] != y_interp_pts[1]:
                 x2 = np.interp(half_max, [y_interp_pts[1], y_interp_pts[0]], [x_interp_pts[1], x_interp_pts[0]]) # Inverser pour interp croissante
            else: # Fallback
                 x2 = x_axis[last_above_idx]
        
        fwhm = np.abs(x2 - x1)
        if fwhm < 1e-9 * (x_axis[-1] - x_axis[0]): # Si FWHM est excessivement petite (bruit numérique)
            return 0.0 
        return fwhm
    except Exception as e:
        # print(f"Debug FWHM calc error: {e}")
        return 0.0 # En cas d'erreur inattendue

def get_plot_limits_v4(axis_values, intensity_values, N_fwhm_zoom, min_total_width_plot_unit):
    if len(intensity_values) == 0 or np.all(intensity_values <= 1e-20) or len(axis_values) < 2:
        if len(axis_values) >= 2: return axis_values.min(), axis_values.max()
        else: return -1, 1

    peak_idx = np.argmax(intensity_values)
    center_val = axis_values[peak_idx]
    fwhm_val = calculate_fwhm_with_interp(axis_values, intensity_values)

    desired_total_width = N_fwhm_zoom * fwhm_val
    actual_total_width = max(desired_total_width, min_total_width_plot_unit)
    
    if fwhm_val == 0.0 and min_total_width_plot_unit == 0.0 : 
         actual_total_width = (axis_values.max() - axis_values.min()) * 0.5 
    elif fwhm_val == 0.0: # Si FWHM est nulle mais min_total_width est défini
        actual_total_width = min_total_width_plot_unit


    lim_min, lim_max = center_val - actual_total_width / 2.0, center_val + actual_total_width / 2.0
    lim_min, lim_max = max(axis_values.min(), lim_min), min(axis_values.max(), lim_max)
    
    if lim_min >= lim_max : 
        # Si les limites se croisent, tenter de centrer sur le pic avec min_total_width
        lim_min = center_val - min_total_width_plot_unit / 2.0
        lim_max = center_val + min_total_width_plot_unit / 2.0
        lim_min = max(axis_values.min(), lim_min)
        lim_max = min(axis_values.max(), lim_max)
        if lim_min >= lim_max : # Fallback final
            lim_min, lim_max = axis_values.min(), axis_values.max()
            
    return lim_min, lim_max

# --- Définition des Entrées Utilisateur (MISES À JOUR) ---
print("--- Définition des Entrées Utilisateur ---")
MFD_um_input = 6.5
D_ps_nm_km_input = -22.0 
n2_m2_W_input = 2.6e-20
gain_dB_m_input = 14.0
lambda0_nm_input = 1555.0
P_avg_W_input = 1.0e-3 # 1mW
f_rep_Hz_input = 40e6
delta_lambda_FWHM_nm_input = 20.0 
GDD_input_fs2_input = 0.0 

print(f"Paramètres Fibre: MFD={MFD_um_input} µm, D={D_ps_nm_km_input} ps/(nm*km), n2={n2_m2_W_input:.1e} m^2/W, Gain={gain_dB_m_input:.2f} dB/m")
print(f"Paramètres Impulsion: Lambda0={lambda0_nm_input} nm, P_avg={P_avg_W_input*1e3:.2f} mW, f_rep={f_rep_Hz_input*1e-6:.1f} MHz, DeltaLambda_FWHM={delta_lambda_FWHM_nm_input} nm, GDD_input={GDD_input_fs2_input} fs^2")

print("\n--- Phase 1: Calculs Théoriques ---")
print("\n1. Calcul des paramètres de la fibre...")
A_eff_m2_calc, gamma_W_m_calc, g_m_inv_calc, beta2_SI_s2_m_calc, beta2_for_pynlo_ps2_km_calc = \
    calculate_fiber_parameters(MFD_um_input, D_ps_nm_km_input, n2_m2_W_input, gain_dB_m_input, lambda0_nm_input)
print("\n2. Calcul des paramètres de l'impulsion d'entrée réelle...")
U_in_reel_J_calc, delta_t_TL_s_calc, GDD_input_ps2_calc = \
    calculate_input_pulse_params(P_avg_W_input, f_rep_Hz_input, delta_lambda_FWHM_nm_input, lambda0_nm_input, GDD_input_fs2_input)
print("\n3. Application de la théorie de Kruglov et al. ...")
A0_calc, delta_TFWHM_opt_s_calc, z_c_m_calc = \
    kruglov_optimal_params(U_in_reel_J_calc, g_m_inv_calc, gamma_W_m_calc, beta2_SI_s2_m_calc, N_ss=100)
print("\n4. Estimation de la puissance moyenne de sortie (basée sur le gain simple)...")
if not np.isnan(z_c_m_calc) and g_m_inv_calc > 0 :
    P_avg_out_estim_W = P_avg_W_input * np.exp(g_m_inv_calc * z_c_m_calc)
    print(f"  Puissance moyenne de sortie estimée (P_avg_out_estim): {P_avg_out_estim_W * 1e3:.2f} mW ({P_avg_out_estim_W:.2f} W)")
    print(f"  Gain total estimé : {10 * np.log10(P_avg_out_estim_W / P_avg_W_input):.2f} dB")
elif g_m_inv_calc <= 0 and not np.isnan(z_c_m_calc) : 
    P_avg_out_estim_W = P_avg_W_input * np.exp(g_m_inv_calc * z_c_m_calc) 
    print(f"  Puissance moyenne de sortie estimée (avec perte/sans gain) (P_avg_out_estim): {P_avg_out_estim_W * 1e3:.2f} mW ({P_avg_out_estim_W:.2f} W)")
else: 
    print(f"  Estimation de la puissance de sortie non applicable car la longueur de fibre z_c n'a pas pu être déterminée (ou gain nul/négatif).")
print("\n--- Fin de la Phase 1 ---")

print("\n--- Phase 2: Simulation PyNLO et Visualisation ---")

USER_SIMULATION_LENGTH_M = 1.5 
FALLBACK_SIMULATION_LENGTH_M = 1.0 

print("\n1. Définition de l'impulsion d'entrée pour PyNLO...")
EPP_J_pynlo = U_in_reel_J_calc
FWHM_input_TL_ps_pynlo = delta_t_TL_s_calc * 1e12
GDD_input_ps2_pynlo = GDD_input_ps2_calc 
time_window_factor = 30.0 
min_time_window_ps = 50.0 # Fenêtre temporelle minimale pour la simulation
time_window_pynlo_ps = max(min_time_window_ps, 
                           time_window_factor * FWHM_input_TL_ps_pynlo if not np.isinf(FWHM_input_TL_ps_pynlo) else min_time_window_ps)
NPTS_pynlo = 2**14 
pulse_in_pynlo = DPL.GaussianPulse(
    power=1.0, T0_ps=FWHM_input_TL_ps_pynlo, center_wavelength_nm=lambda0_nm_input,
    time_window_ps=time_window_pynlo_ps, GDD=GDD_input_ps2_pynlo, TOD=0.0, 
    NPTS=NPTS_pynlo, frep_MHz=f_rep_Hz_input * 1e-6, power_is_avg=False)
pulse_in_pynlo.set_epp(EPP_J_pynlo)
print(f"  Impulsion PyNLO définie: EPP={pulse_in_pynlo.calc_epp()*1e12:.2f} pJ, FWHM_TL={FWHM_input_TL_ps_pynlo:.3f} ps, GDD={GDD_input_ps2_pynlo:.2e} ps^2, Fenêtre T={time_window_pynlo_ps:.2f} ps")

print("\n2. Définition de la fibre pour PyNLO...")
if USER_SIMULATION_LENGTH_M is not None and USER_SIMULATION_LENGTH_M > 0:
    fiber_length_pynlo_m = USER_SIMULATION_LENGTH_M
    print(f"  Utilisation de la longueur de fibre spécifiée par l'utilisateur : {fiber_length_pynlo_m:.3f} m")
elif not np.isnan(z_c_m_calc) and beta2_SI_s2_m_calc > 0:
    fiber_length_pynlo_m = z_c_m_calc
    print(f"  Utilisation de la longueur de fibre z_c calculée (Kruglov applicable) : {fiber_length_pynlo_m:.3f} m")
else:
    fiber_length_pynlo_m = FALLBACK_SIMULATION_LENGTH_M
    print(f"  ATTENTION: z_c ou beta2 non valide pour la théorie de Kruglov, ou USER_SIMULATION_LENGTH_M non spécifiée.")
    print(f"             Utilisation d'une longueur de fibre de repli : {fiber_length_pynlo_m:.3f} m")
fiber_pynlo = FIB.FiberInstance()
fiber_pynlo.generate_fiber(
    length=fiber_length_pynlo_m, center_wl_nm=lambda0_nm_input,
    betas=(beta2_for_pynlo_ps2_km_calc, 0.0, 0.0), gamma_W_m=gamma_W_m_calc, 
    gain=g_m_inv_calc, gvd_units='ps^n/km')
print(f"  Fibre PyNLO définie: L={fiber_pynlo.length:.3f} m, beta2={beta2_for_pynlo_ps2_km_calc:.2f} ps^2/km, gamma={gamma_W_m_calc:.2e} 1/(W*m), g={g_m_inv_calc:.3f} 1/m")

print("\n3. Configuration du solveur SSFM...")
simulation_steps = 200 
ssfm_solver = SSFM.SSFM(
    local_error=0.005, disable_Raman=False, disable_self_steepening=False, USE_SIMPLE_RAMAN=False)
print(f"  Solveur SSFM configuré.")
print(f"  NOTE: PyNLO SSFM avec 'generate_fiber' utilise un gain linéaire constant g.")
print(f"        Il ne modélise pas directement une énergie de saturation (Esat) comme peut le faire votre script Julia.")

print("\n4. Exécution de la propagation SSFM...")
y_pynlo_m, AW_pynlo, AT_pynlo, pulse_out_pynlo = ssfm_solver.propagate(
    pulse_in=pulse_in_pynlo, fiber=fiber_pynlo, n_steps=simulation_steps)
print(f"  Propagation terminée.")

print("\n5. Analyse et visualisation des résultats...")
P_avg_out_J_pynlo = pulse_out_pynlo.calc_epp() * pulse_out_pynlo.frep_mks
print(f"  Puissance moyenne d'entrée (pour comparaison): {P_avg_W_input * 1e3:.2f} mW")
print(f"  Puissance moyenne de sortie (simulée): {P_avg_out_J_pynlo * 1e3:.2f} mW")
print(f"  Gain total simulé : {10 * np.log10(P_avg_out_J_pynlo / P_avg_W_input):.2f} dB")

# --- Paramètres de Zoom pour les Graphiques (FACILEMENT MODIFIABLES) ---
N_FWHM_T_ZOOM_EVOL = 7.0 
MIN_TOTAL_WIDTH_T_PS_EVOL = FWHM_input_TL_ps_pynlo * 15 if not np.isinf(FWHM_input_TL_ps_pynlo) else 30.0 
if MIN_TOTAL_WIDTH_T_PS_EVOL == 0 : MIN_TOTAL_WIDTH_T_PS_EVOL = 30.0 

N_FWHM_W_ZOOM_EVOL = 7.0  
MIN_TOTAL_WIDTH_W_NM_EVOL = delta_lambda_FWHM_nm_input * 6 if delta_lambda_FWHM_nm_input > 0 else 150.0 
if MIN_TOTAL_WIDTH_W_NM_EVOL == 0 : MIN_TOTAL_WIDTH_W_NM_EVOL = 150.0
# --------------------------------------------------------------------

# --- Graphiques d'évolution 2D (Intensité en dB) ---
fig_evol, (ax_evol_T, ax_evol_W) = plt.subplots(1, 2, figsize=(15, 6))
AT_pynlo_abs_sq_dB = 10 * np.log10(np.abs(AT_pynlo)**2 + 1e-30)
lim_T_min_evol, lim_T_max_evol = get_plot_limits_v4(pulse_in_pynlo.T_ps, np.abs(AT_pynlo[:,0])**2, N_FWHM_T_ZOOM_EVOL, MIN_TOTAL_WIDTH_T_PS_EVOL)
im_T = ax_evol_T.imshow( AT_pynlo_abs_sq_dB.T, extent=[pulse_in_pynlo.T_ps.min(), pulse_in_pynlo.T_ps.max(), y_pynlo_m.min()*1e3, y_pynlo_m.max()*1e3], aspect='auto', origin='lower', cmap='viridis', vmin = np.max(AT_pynlo_abs_sq_dB) - 40)
ax_evol_T.set_xlabel("Temps (ps)"); ax_evol_T.set_ylabel("Distance (mm)"); ax_evol_T.set_title("Évolution Temporelle (dB)")
ax_evol_T.set_xlim(lim_T_min_evol, lim_T_max_evol); fig_evol.colorbar(im_T, ax=ax_evol_T, label="Intensité (dB)")

AW_pynlo_abs_sq_dB = 10 * np.log10(np.abs(AW_pynlo)**2 + 1e-30)
lim_W_min_evol, lim_W_max_evol = get_plot_limits_v4(pulse_in_pynlo.wl_nm, np.abs(AW_pynlo[:,0])**2, N_FWHM_W_ZOOM_EVOL, MIN_TOTAL_WIDTH_W_NM_EVOL)
im_W = ax_evol_W.imshow( AW_pynlo_abs_sq_dB.T, extent=[pulse_in_pynlo.wl_nm.min(), pulse_in_pynlo.wl_nm.max(), y_pynlo_m.min()*1e3, y_pynlo_m.max()*1e3], aspect='auto', origin='lower', cmap='viridis', vmin = np.max(AW_pynlo_abs_sq_dB) - 40)
ax_evol_W.set_xlabel("Longueur d'onde (nm)"); ax_evol_W.set_ylabel("Distance (mm)"); ax_evol_W.set_title("Évolution Spectrale (dB)")
ax_evol_W.set_xlim(lim_W_min_evol, lim_W_max_evol); fig_evol.colorbar(im_W, ax=ax_evol_W, label="Intensité (dB)")
fig_evol.tight_layout(); plt.show()

# --- Graphiques d'entrée/sortie 1D et étapes intermédiaires (ÉCHELLE LINÉAIRE) ---
fig_io_evol, (ax_io_T_evol, ax_io_W_evol) = plt.subplots(2, 1, figsize=(10, 10))
num_total_steps_output = AW_pynlo.shape[1]
indices_intermediaires = [0] 
if num_total_steps_output > 1 : # S'assurer qu'il y a au moins un pas de sortie
    step_indices = np.linspace(0, num_total_steps_output - 1, min(5, num_total_steps_output), dtype=int)
    indices_intermediaires = step_indices
else: # Cas où il n'y a qu'un seul pas de sortie (ou aucun après l'entrée)
    indices_intermediaires = np.array([0], dtype=int) if num_total_steps_output > 0 else np.array([])


# Valeurs de zoom pour les graphiques d'évolution 1D basées sur le profil de SORTIE
intensity_out_T_final_lin = np.abs(AT_pynlo[:,-1])**2 if num_total_steps_output > 0 else np.abs(AT_pynlo[:,0])**2
lim_T_min_io_evol_lin, lim_T_max_io_evol_lin = get_plot_limits_v4(pulse_in_pynlo.T_ps, intensity_out_T_final_lin, N_FWHM_T_ZOOM_EVOL, MIN_TOTAL_WIDTH_T_PS_EVOL)

intensity_out_W_final_lin = np.abs(AW_pynlo[:,-1])**2 if num_total_steps_output > 0 else np.abs(AW_pynlo[:,0])**2
lim_W_min_io_evol_lin, lim_W_max_io_evol_lin = get_plot_limits_v4(pulse_in_pynlo.wl_nm, intensity_out_W_final_lin, N_FWHM_W_ZOOM_EVOL, MIN_TOTAL_WIDTH_W_NM_EVOL)

colors = plt.cm.viridis(np.linspace(0, 1, max(1,len(indices_intermediaires)))) # max(1,..) pour éviter erreur si indices_intermediaires est vide

if len(indices_intermediaires) > 0:
    for k, idx in enumerate(indices_intermediaires):
        intensity_T_step_lin = np.abs(AT_pynlo[:,idx])**2
        fwhm_T_step_ps = calculate_fwhm_with_interp(pulse_in_pynlo.T_ps, intensity_T_step_lin) * 1e3 # en fs
        label = f"z = {y_pynlo_m[idx]*1e3:.2f} mm (FWHM: {fwhm_T_step_ps:.1f} fs)"
        if idx == 0 and len(indices_intermediaires) > 1 : label = f"Entrée (z=0 mm, FWHM: {fwhm_T_step_ps:.1f} fs)"
        elif idx == indices_intermediaires[-1] and len(indices_intermediaires) > 1: label = f"Sortie (z={y_pynlo_m[idx]*1e3:.2f} mm, FWHM: {fwhm_T_step_ps:.1f} fs)"
        
        ax_io_T_evol.plot(pulse_in_pynlo.T_ps, intensity_T_step_lin / np.max(intensity_T_step_lin + 1e-30), 
                          label=label, color=colors[k]) 
ax_io_T_evol.set_xlabel("Temps (ps)"); ax_io_T_evol.set_ylabel("Intensité Normalisée (lin.)"); 
ax_io_T_evol.set_title(f"Profils Temporels à Différentes Distances (lin.)"); ax_io_T_evol.legend(fontsize='small')
ax_io_T_evol.set_xlim(lim_T_min_io_evol_lin, lim_T_max_io_evol_lin); ax_io_T_evol.set_ylim(-0.05, 1.1); ax_io_T_evol.grid(True)

if len(indices_intermediaires) > 0:
    for k, idx in enumerate(indices_intermediaires):
        intensity_W_step_lin = np.abs(AW_pynlo[:,idx])**2
        fwhm_W_step_nm = calculate_fwhm_with_interp(pulse_in_pynlo.wl_nm, intensity_W_step_lin)
        label = f"z = {y_pynlo_m[idx]*1e3:.2f} mm (FWHM: {fwhm_W_step_nm:.1f} nm)"
        if idx == 0 and len(indices_intermediaires) > 1: label = f"Entrée (z=0 mm, FWHM: {fwhm_W_step_nm:.1f} nm)"
        elif idx == indices_intermediaires[-1] and len(indices_intermediaires) > 1: label = f"Sortie (z={y_pynlo_m[idx]*1e3:.2f} mm, FWHM: {fwhm_W_step_nm:.1f} nm)"
        
        ax_io_W_evol.plot(pulse_in_pynlo.wl_nm, intensity_W_step_lin / np.max(intensity_W_step_lin + 1e-30), 
                          label=label, color=colors[k]) 
ax_io_W_evol.set_xlabel("Longueur d'onde (nm)"); ax_io_W_evol.set_ylabel("Intensité Spectrale Normalisée (lin.)"); 
ax_io_W_evol.set_title(f"Spectres à Différentes Distances (lin.)"); ax_io_W_evol.legend(fontsize='small')
ax_io_W_evol.set_xlim(lim_W_min_io_evol_lin, lim_W_max_io_evol_lin); ax_io_W_evol.set_ylim(-0.05, 1.1); ax_io_W_evol.grid(True)
fig_io_evol.tight_layout(pad=2.5); plt.show()

fig_phase_gd, (ax_phase_T_out, ax_phase_W_out) = plt.subplots(1, 2, figsize=(15, 5))
phase_temporal_out = np.unwrap(np.angle(pulse_out_pynlo.AT)); delta_T_ps = pulse_out_pynlo.T_ps[1] - pulse_out_pynlo.T_ps[0]
inst_freq_THz = pulse_out_pynlo.center_frequency_THz - (np.gradient(phase_temporal_out) / delta_T_ps) / (2 * np.pi)
ax_phase_T_out_twin = ax_phase_T_out.twinx()
p1_phase_T, = ax_phase_T_out.plot(pulse_out_pynlo.T_ps, phase_temporal_out, color='green', label="Phase Temporelle (rad)")
p2_phase_T, = ax_phase_T_out_twin.plot(pulse_out_pynlo.T_ps, inst_freq_THz, color='purple', linestyle='--', label="Fréq. Inst. (THz)")
ax_phase_T_out.set_xlabel("Temps (ps)"); ax_phase_T_out.set_ylabel("Phase Temporelle (rad)", color='green')
ax_phase_T_out_twin.set_ylabel("Fréquence Instantanée (THz)", color='purple')
ax_phase_T_out.set_title("Phase Temporelle et Fréq. Inst. (Sortie)"); ax_phase_T_out.set_xlim(lim_T_min_io_evol_lin, lim_T_max_io_evol_lin)
ax_phase_T_out.grid(True); ax_phase_T_out.legend(handles=[p1_phase_T, p2_phase_T], loc='upper right')

phase_spectral_out_unwrapped = np.unwrap(np.angle(pulse_out_pynlo.AW)) # Version directe
group_delay_s = np.gradient(phase_spectral_out_unwrapped, pulse_out_pynlo.W_mks) 
ax_phase_W_out_twin = ax_phase_W_out.twinx()
p1_phase_W, = ax_phase_W_out.plot(pulse_out_pynlo.wl_nm, phase_spectral_out_unwrapped, color='green', label="Phase Spectrale (rad)")
p2_phase_W, = ax_phase_W_out_twin.plot(pulse_out_pynlo.wl_nm, group_delay_s * 1e12, color='purple', linestyle='--', label="Délai de Groupe (ps)")
mean_phase, std_phase = np.nanmean(phase_spectral_out_unwrapped), np.nanstd(phase_spectral_out_unwrapped)
if not (np.isnan(mean_phase) or np.isnan(std_phase) or std_phase < 1e-9): ax_phase_W_out.set_ylim(mean_phase - 3*std_phase, mean_phase + 3*std_phase)
else: ax_phase_W_out.set_ylim( (-np.pi if np.all(np.isnan(phase_spectral_out_unwrapped)) else np.nanmin(phase_spectral_out_unwrapped))-0.1, (np.pi if np.all(np.isnan(phase_spectral_out_unwrapped)) else np.nanmax(phase_spectral_out_unwrapped))+0.1)
mean_gd, std_gd = np.nanmean(group_delay_s*1e12), np.nanstd(group_delay_s*1e12)
if not (np.isnan(mean_gd) or np.isnan(std_gd) or std_gd < 1e-9): ax_phase_W_out_twin.set_ylim(mean_gd - 3*std_gd, mean_gd + 3*std_gd)
else: ax_phase_W_out_twin.set_ylim( (np.nanmin(group_delay_s*1e12) if not np.all(np.isnan(group_delay_s)) else -1.0)-0.1, (np.nanmax(group_delay_s*1e12) if not np.all(np.isnan(group_delay_s)) else 1.0)+0.1)
ax_phase_W_out.set_xlabel("Longueur d'onde (nm)"); ax_phase_W_out.set_ylabel("Phase Spectrale (rad)", color='green')
ax_phase_W_out_twin.set_ylabel("Délai de Groupe (ps)", color='purple')
ax_phase_W_out.set_title("Phase Spectrale et Délai de Groupe (Sortie)"); ax_phase_W_out.set_xlim(lim_W_min_io_evol_lin, lim_W_max_io_evol_lin)
ax_phase_W_out.grid(True); ax_phase_W_out.legend(handles=[p1_phase_W, p2_phase_W], loc='upper right')
fig_phase_gd.tight_layout(pad=2.5); plt.show()