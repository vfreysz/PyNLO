import numpy as np
import matplotlib.pyplot as plt
import pynlo
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.constants import speed_of_light # c_mks

# Constantes globales pour les fonctions du compresseur portées
c_mks = speed_of_light # m/s

# --- Paramètres de l'impulsion et de la simulation ---
FWHM_initial_fs = 250.0
EPP_nj = 150.0
pulse_wavelength_nm = 1555.0
time_window_ps = 1500.0 
NPTS = 2**16 # Haute résolution

beta2_cfbg_ps2 = 28.69
beta3_cfbg_ps3 = -1.096
FOD_cfbg_ps4 = 0.0

fiber_length_m = 10.0
beta2_fiber_ps2_km = -23.0
beta3_fiber_ps3_km = 0.1
beta4_fiber_ps4_km = 0.0
gamma_fiber_W_km = 1.5
ssfm_steps = 200
ssfm_local_error = 0.005

# Paramètres du compresseur à réseaux
grating_lines_per_mm = 1200.5
m_order_compressor = -1 
N_passes_compressor = 2    

fwhm_initial_ps = FWHM_initial_fs / 1000.0
t0_initial_ps = fwhm_initial_ps / 1.7627
epp_J = EPP_nj * 1e-9

# --- Fonctions utilitaires portées et adaptées pour PyNLO ---
def calculate_grating_dispersion_custom(pulse_obj_pynlo, grating_lines_per_mm, L_eff_m,
                                     theta_i_deg, m_input=-1, N_passes_formula=2):
    if m_input == 0:
        raise ValueError("m_input ne peut pas être zéro pour un compresseur à réseaux.")
    lambda_0_m = pulse_obj_pynlo.center_wavelength_nm * 1e-9
    d_m = 1e-3 / grating_lines_per_mm
    theta_i_rad = np.deg2rad(theta_i_deg)
    sin_theta_d_calc = (m_input * lambda_0_m / d_m) - np.sin(theta_i_rad)
    if abs(sin_theta_d_calc) > 1.000001:
        return (np.inf, np.inf, np.inf)
    sin_theta_d = np.clip(sin_theta_d_calc, -1.0, 1.0)
    cos_theta_d_sq = 1.0 - sin_theta_d**2
    if np.isclose(cos_theta_d_sq, 0):
        return (np.inf, np.inf, np.inf)
    cos_theta_d = np.sqrt(cos_theta_d_sq)
    L_g_m = N_passes_formula * L_eff_m
    gdd_s2 = ((-L_g_m * (m_input**2) * (lambda_0_m**3)) /
              (2 * np.pi * (c_mks**2) * (d_m**2) * (cos_theta_d**3)))
    term_order_lambda_d = (m_input * lambda_0_m / d_m)
    tod_gdd_ratio_angular_factor = 1.0 + (term_order_lambda_d * sin_theta_d / cos_theta_d_sq)
    tod_s3 = gdd_s2 * (-3 * lambda_0_m / (2 * np.pi * c_mks)) * tod_gdd_ratio_angular_factor
    fod_tod_ratio_angular_term1 = 1.5 * term_order_lambda_d * sin_theta_d / cos_theta_d_sq
    fod_tod_ratio_angular_term2 = 0.5 * (d_m * sin_theta_d) / (m_input * lambda_0_m) 
    fod_tod_ratio_angular_factor = 1.0 + fod_tod_ratio_angular_term1 - fod_tod_ratio_angular_term2
    fod_s4 = tod_s3 * (-4 * lambda_0_m / (2 * np.pi * c_mks)) * fod_tod_ratio_angular_factor
    return gdd_s2 * 1e24, tod_s3 * 1e36, fod_s4 * 1e48

def get_littrow_angle_custom(pulse_obj_pynlo, grating_lines_per_mm, m_order=1):
    if m_order == 0: return float('nan')
    lambda_0_m = pulse_obj_pynlo.center_wavelength_nm * 1e-9
    d_m = 1e-3 / grating_lines_per_mm
    sin_littrow_val = (m_order * lambda_0_m) / (2 * d_m)
    if -1 <= sin_littrow_val <= 1:
        return np.rad2deg(np.arcsin(sin_littrow_val))
    return float('nan')

def calculate_fwhm(x_axis, y_axis, is_spectral_wavelength=False):
    if is_spectral_wavelength:
        valid_indices = np.where(~np.isnan(x_axis) & ~np.isinf(x_axis))
        x_axis_filt, y_axis_filt = x_axis[valid_indices], y_axis[valid_indices]
        if len(x_axis_filt) == 0: return 0, 0, (0,1), np.array([0]), np.array([0])
        sorted_indices = np.argsort(x_axis_filt)
        x_sorted = x_axis_filt[sorted_indices]
        y_sorted = y_axis_filt[sorted_indices]
    else:
        x_sorted, y_sorted = x_axis, y_axis
    if len(x_sorted) < 2 or np.max(y_sorted) < 1e-12:
        min_x, max_x = (np.min(x_axis), np.max(x_axis)) if len(x_axis) > 0 else (0,1)
        return 0, (min_x + max_x) / 2, (min_x, max_x), x_sorted, y_sorted
    y_max = np.max(y_sorted); half_max = y_max / 2.0
    center_x_val = x_sorted[np.argmax(y_sorted)]
    try:
        above_half_max_indices = np.where(y_sorted >= half_max)[0]
        if not above_half_max_indices.any(): return 0, center_x_val, (np.min(x_sorted), np.max(x_sorted)), x_sorted, y_sorted
        continuous_segments = np.split(above_half_max_indices, np.where(np.diff(above_half_max_indices) != 1)[0]+1)
        largest_segment = max(continuous_segments, key=len, default=np.array([]))
        if largest_segment.size == 0: return 0, center_x_val, (np.min(x_sorted), np.max(x_sorted)), x_sorted, y_sorted
        first_idx, last_idx = largest_segment[0], largest_segment[-1]
        if first_idx == 0 or y_sorted[first_idx] == half_max: x1 = x_sorted[first_idx]
        else:
            xp, yp, xc, yc = x_sorted[first_idx-1], y_sorted[first_idx-1], x_sorted[first_idx], y_sorted[first_idx]
            x1 = xp + (half_max - yp) * (xc - xp) / (yc - yp) if (yc - yp) != 0 else xc
        if last_idx == len(x_sorted)-1 or y_sorted[last_idx] == half_max: x2 = x_sorted[last_idx]
        else:
            xc, yc, xn, yn = x_sorted[last_idx], y_sorted[last_idx], x_sorted[last_idx+1], y_sorted[last_idx+1]
            x2 = xc + (half_max - yc) * (xn - xc) / (yn - yc) if (yn - yc) != 0 else xc
        fwhm = abs(x2 - x1)
        zoom_factor = 5
        if fwhm < 1e-6 : 
            default_half_width = 2.0 if not is_spectral_wavelength else 20.0 
            zoom_min, zoom_max = center_x_val - default_half_width, center_x_val + default_half_width
        else:
            zoom_min, zoom_max = center_x_val - zoom_factor*fwhm, center_x_val + zoom_factor*fwhm
        zoom_min, zoom_max = max(np.min(x_sorted), zoom_min), min(np.max(x_sorted), zoom_max)
        if zoom_min >= zoom_max: zoom_min, zoom_max = np.min(x_sorted), np.max(x_sorted)
        return fwhm, center_x_val, (zoom_min, zoom_max), x_sorted, y_sorted
    except Exception:
        return 0, center_x_val, (np.min(x_sorted), np.max(x_sorted)), x_sorted, y_sorted

def plot_pulse(pulse, title_prefix="", central_wl_nm_spectrum=1555.0):
    fig, (ax_time, ax_freq) = plt.subplots(1, 2, figsize=(14, 5))
    intensity_time = np.abs(pulse.AT)**2
    fwhm_t, center_t, xlim_t, _, _ = calculate_fwhm(pulse.T_ps, intensity_time)
    ax_time.plot(pulse.T_ps, intensity_time); ax_time.set_xlabel("Temps (ps)"); ax_time.set_ylabel("Intensité (ua)")
    ax_time.set_title(f"{title_prefix} - Forme Temporelle\nFWHM: {fwhm_t*1000:.2f} fs"); ax_time.set_xlim(xlim_t); ax_time.grid(True)
    if fwhm_t > 1e-6 :
        max_it = np.max(intensity_time)
        if max_it > 0 : ax_time.axhline(max_it/2, color='r', linestyle='--', lw=0.8, alpha=0.7)
        ax_time.axvline(center_t - fwhm_t/2, color='r', linestyle='--', lw=0.8, alpha=0.7)
        ax_time.axvline(center_t + fwhm_t/2, color='r', linestyle='--', lw=0.8, alpha=0.7)

    fwhm_wl, center_wl, xlim_wl, wl_sorted, spec_sorted = calculate_fwhm(pulse.wl_nm, np.abs(pulse.AW)**2, True)
    ax_freq.plot(wl_sorted, spec_sorted); ax_freq.set_xlabel("Longueur d'onde (nm)"); ax_freq.set_ylabel("Intensité Spectrale (ua)")
    ax_freq.set_title(f"{title_prefix} - Spectre\nFWHM: {fwhm_wl:.2f} nm")
    if fwhm_wl > 1e-3 and len(wl_sorted)>1 and fwhm_wl < (wl_sorted[-1] - wl_sorted[0]):
        spec_xlim_min = max(wl_sorted[0], center_wl - 3*fwhm_wl); spec_xlim_max = min(wl_sorted[-1], center_wl + 3*fwhm_wl)
        spec_xlim_min = min(spec_xlim_min, central_wl_nm_spectrum - 10); spec_xlim_max = max(spec_xlim_max, central_wl_nm_spectrum + 10)
        ax_freq.set_xlim(spec_xlim_min, spec_xlim_max)
    elif len(wl_sorted)>1 : ax_freq.set_xlim(central_wl_nm_spectrum - 50, central_wl_nm_spectrum + 200)
    ax_freq.grid(True)
    if fwhm_wl > 1e-6 and len(spec_sorted)>0:
        max_is = np.max(spec_sorted)
        if max_is > 0: ax_freq.axhline(max_is/2, color='r', linestyle='--', lw=0.8, alpha=0.7)
        ax_freq.axvline(center_wl - fwhm_wl/2, color='r', linestyle='--', lw=0.8, alpha=0.7)
        ax_freq.axvline(center_wl + fwhm_wl/2, color='r', linestyle='--', lw=0.8, alpha=0.7)
    plt.tight_layout(); plt.show()

# --- Étape 1 : Génération de l'impulsion ---
print("Étape 1: Génération de l'impulsion initiale...")
print(f"DEBUG: NPTS={NPTS}, time_window_ps={time_window_ps}, t0_initial_ps={t0_initial_ps:.4f}")
with np.errstate(over='ignore'): 
    pulse_initial = pynlo.light.DerivedPulses.SechPulse(
        power=1.0, T0_ps=t0_initial_ps, center_wavelength_nm=pulse_wavelength_nm,
        time_window_ps=time_window_ps, NPTS=NPTS, frep_MHz=100.0, power_is_avg=False)
pulse_initial.set_epp(epp_J)
print(f"  Énergie: {pulse_initial.calc_epp() * 1e9:.2f} nJ, dt: {pulse_initial.dT_ps * 1000:.3f} fs")
plot_pulse(pulse_initial, "Impulsion Initiale", pulse_wavelength_nm)

# --- Étape 2 : Application de la phase (CFBG) ---
print("\nÉtape 2: Application de la phase du CFBG...")
pulse_stretched = pulse_initial.create_cloned_pulse()
pulse_stretched.chirp_pulse_W(GDD=beta2_cfbg_ps2, TOD=beta3_cfbg_ps3, FOD=FOD_cfbg_ps4)
plot_pulse(pulse_stretched, "Après CFBG (Étirée)", pulse_wavelength_nm)

# --- Étape 3 : Propagation dans la fibre PM1550 ---
print("\nÉtape 3: Propagation dans la fibre PM1550...")
pm1550_fiber = pynlo.media.fibers.fiber.FiberInstance()
pm1550_fiber.generate_fiber(fiber_length_m, pulse_wavelength_nm,
    (beta2_fiber_ps2_km, beta3_fiber_ps3_km, beta4_fiber_ps4_km),
    gamma_fiber_W_km * 1e-3, gvd_units='ps^n/km', label="PM1550")
ssfm_solver = pynlo.interactions.FourWaveMixing.SSFM.SSFM(ssfm_local_error, 
                                                          disable_Raman = False, 
                                                          disable_self_steepening = False,
                                                          suppress_iteration = True, 
                                                          USE_SIMPLE_RAMAN = False)
z_coords, AW_fiber, AT_fiber, pulse_after_fiber = ssfm_solver.propagate(pulse_stretched, pm1550_fiber, ssfm_steps)
print(f"  Énergie en sortie de fibre: {pulse_after_fiber.calc_epp() * 1e9:.2f} nJ")
plot_pulse(pulse_after_fiber, f"Après {fiber_length_m}m de PM1550", pulse_wavelength_nm)

# --- Étape 4 : Compression et Optimisation du Compresseur ---
print("\n--- Étape 4: Compression et Optimisation du Compresseur ---")
gdd_fiber_ps2 = beta2_fiber_ps2_km * 1e-3 * fiber_length_m 
tod_fiber_ps3 = beta3_fiber_ps3_km * 1e-3 * fiber_length_m 
fod_fiber_ps4 = beta4_fiber_ps4_km * 1e-3 * fiber_length_m 
gdd_total_to_compensate_ps2 = beta2_cfbg_ps2 + gdd_fiber_ps2
tod_total_to_compensate_ps3 = beta3_cfbg_ps3 + tod_fiber_ps3
fod_total_to_compensate_ps4 = FOD_cfbg_ps4 + fod_fiber_ps4
print(f"Dispersion Totale à compenser (Étireur + Fibre):\n  GDD_total = {gdd_total_to_compensate_ps2:.4f} ps^2\n  TOD_total = {tod_total_to_compensate_ps3:.4f} ps^3\n  FOD_total = {fod_total_to_compensate_ps4:.4f} ps^4")

# --- 4.A Optimisation Analytique (GDD seul) ---
print("\n--- 4.A Optimisation Analytique du Compresseur (GDD Seul) ---")
pulse_gdd_optimized_analytic = pulse_after_fiber.create_cloned_pulse()
theta_i_deg_analytic_gdd = get_littrow_angle_custom(pulse_gdd_optimized_analytic, grating_lines_per_mm, m_order=m_order_compressor)
L_eff_m_analytic_gdd = 1.0 # Valeur par défaut si Littrow non physique ou calcul échoue

if not np.isnan(theta_i_deg_analytic_gdd):
    GDD_target_comp_s2 = -gdd_total_to_compensate_ps2 * 1e-24
    lambda_0_m = pulse_gdd_optimized_analytic.center_wavelength_nm * 1e-9
    d_m = 1e-3 / grating_lines_per_mm
    cos_theta_d_analytic = np.cos(np.deg2rad(theta_i_deg_analytic_gdd))
    if not (cos_theta_d_analytic**3 == 0 or m_order_compressor**2 == 0 or lambda_0_m**3 ==0):
        L_g_m_analytic = -GDD_target_comp_s2 * (2*np.pi*c_mks**2*d_m**2*cos_theta_d_analytic**3) / (m_order_compressor**2*lambda_0_m**3)
        L_eff_m_analytic_gdd = L_g_m_analytic / N_passes_compressor
    else: print("  Avertissement: division par zéro évitée dans L_g_m_analytic.")
else: print("  Angle de Littrow non physique pour l'optimisation analytique du GDD.")

print(f"  Angle de Littrow (m={m_order_compressor}): {theta_i_deg_analytic_gdd:.3f} deg")
print(f"  Distance L_eff calculée pour GDD seul: {L_eff_m_analytic_gdd * 100:.3f} cm")

gdd_comp_an, tod_comp_an, fod_comp_an = calculate_grating_dispersion_custom(
    pulse_gdd_optimized_analytic, grating_lines_per_mm, L_eff_m_analytic_gdd, theta_i_deg_analytic_gdd, m_order_compressor, N_passes_compressor)
pulse_gdd_optimized_analytic.chirp_pulse_W(GDD=gdd_comp_an, TOD=tod_comp_an, FOD=fod_comp_an)
plot_pulse(pulse_gdd_optimized_analytic, "Après Compresseur (Opt. GDD Analytique)", pulse_wavelength_nm)
print(f"  Dispersion du compresseur (Opt. GDD An.): GDD={gdd_comp_an:.2e}, TOD={tod_comp_an:.2e}, FOD={fod_comp_an:.2e} (ps^n)")
print(f"  Dispersion Résiduelle (Opt. GDD An.): GDD={(gdd_total_to_compensate_ps2 + gdd_comp_an):.2e}, TOD={(tod_total_to_compensate_ps3 + tod_comp_an):.2e}, FOD={(fod_total_to_compensate_ps4 + fod_comp_an):.2e} (ps^n)")

# --- (Ancienne Étape 4.B RETIRÉE) ---

# --- 4.B (Anciennement 4.C) Optimisation Numérique (Maximisation de la puissance crête) ---
# Point de départ: résultats de l'optimisation analytique 4.A
print("\n--- 4.B Optimisation Numérique du Compresseur (Maximisation Puissance Crête) ---")
print("Utilisation des résultats de l'optimisation analytique GDD comme point de départ.")

def cost_function_peak_power_pynlo(params, pulse_to_compress_ref):
    L_eff_m, theta_i_deg = params
    temp_pulse = pulse_to_compress_ref.create_cloned_pulse()
    gdd_c, tod_c, fod_c = calculate_grating_dispersion_custom(
        temp_pulse, grating_lines_per_mm, L_eff_m, theta_i_deg, m_order_compressor, N_passes_compressor)
    if np.isinf(gdd_c): return 1e12 
    temp_pulse.chirp_pulse_W(GDD=gdd_c, TOD=tod_c, FOD=fod_c)
    return -np.max(np.abs(temp_pulse.AT)**2)

# Utiliser les résultats de 4.A comme point de départ. S'assurer qu'ils sont valides.
initial_guess_peak = [L_eff_m_analytic_gdd if (not np.isnan(L_eff_m_analytic_gdd) and 0.01 < L_eff_m_analytic_gdd < 2.0) else 0.5,
                      theta_i_deg_analytic_gdd if (not np.isnan(theta_i_deg_analytic_gdd) and -88 < theta_i_deg_analytic_gdd < -30) else -60.0]

bounds_peak = [(0.01, 2.0), (-88, -30)] # Limites physiques (L_eff > 0, angle raisonnable)
result_peak_power = minimize(cost_function_peak_power_pynlo, initial_guess_peak,
                             args=(pulse_after_fiber,), method='L-BFGS-B',
                             bounds=bounds_peak, options={'ftol': 1e-9, 'gtol': 1e-7, 'eps':1e-9})

L_eff_m_peak_opt, theta_i_deg_peak_opt = result_peak_power.x
print(f"  Optimisation (Puissance Crête) terminée.")
print(f"    Puissance crête max estimée: {-result_peak_power.fun:.3e} W")
print(f"    Angle optimisé: {theta_i_deg_peak_opt:.4f} deg, Distance optimisée: {L_eff_m_peak_opt * 100:.4f} cm")

pulse_peak_power_optimized = pulse_after_fiber.create_cloned_pulse()
gdd_c_peak, tod_c_peak, fod_c_peak = calculate_grating_dispersion_custom(
    pulse_peak_power_optimized, grating_lines_per_mm, L_eff_m_peak_opt, theta_i_deg_peak_opt, m_order_compressor, N_passes_compressor)
pulse_peak_power_optimized.chirp_pulse_W(GDD=gdd_c_peak, TOD=tod_c_peak, FOD=fod_c_peak)
plot_pulse(pulse_peak_power_optimized, "Après Compresseur (Opt. Puissance Crête)", pulse_wavelength_nm)
print(f"  Dispersion du compresseur (Opt. Crête): GDD={gdd_c_peak:.2e}, TOD={tod_c_peak:.2e}, FOD={fod_c_peak:.2e} (ps^n)")
print(f"  Dispersion Résiduelle (Opt. Crête): GDD={(gdd_total_to_compensate_ps2 + gdd_c_peak):.2e}, TOD={(tod_total_to_compensate_ps3 + tod_c_peak):.2e}, FOD={(fod_total_to_compensate_ps4 + fod_c_peak):.2e} (ps^n)")

# --- 5. Tracé Comparatif Final ---
print("\n--- 5. Tracé Comparatif Final ---")
fig_comp, ax_comp = plt.subplots(figsize=(12, 7))
norm_init_peak = np.max(np.abs(pulse_initial.AT)**2)
if norm_init_peak == 0: norm_init_peak = 1 
ax_comp.plot(pulse_initial.T_ps, np.abs(pulse_initial.AT)**2 / norm_init_peak, 'k-', label=f'Initiale (FWHM: {calculate_fwhm(pulse_initial.T_ps, np.abs(pulse_initial.AT)**2)[0]*1000:.1f} fs)')
ax_comp.plot(pulse_stretched.T_ps, np.abs(pulse_stretched.AT)**2 / norm_init_peak, 'gray', linestyle=':', label=f'Étirée (CFBG)')
ax_comp.plot(pulse_after_fiber.T_ps, np.abs(pulse_after_fiber.AT)**2 / norm_init_peak, 'm-', alpha=0.6, label=f'Après Fibre')

norm_final_peak = np.max(np.abs(pulse_peak_power_optimized.AT)**2)
if norm_final_peak == 0: norm_final_peak = norm_init_peak 

ax_comp.plot(pulse_gdd_optimized_analytic.T_ps, np.abs(pulse_gdd_optimized_analytic.AT)**2 / norm_final_peak, 'c-.', alpha=0.8, label=f'Opt. GDD An. (FWHM: {calculate_fwhm(pulse_gdd_optimized_analytic.T_ps, np.abs(pulse_gdd_optimized_analytic.AT)**2)[0]*1000:.1f} fs)')
# pulse_disp_coeffs_optimized n'existe plus, donc on ne la trace pas.
ax_comp.plot(pulse_peak_power_optimized.T_ps, np.abs(pulse_peak_power_optimized.AT)**2 / norm_final_peak, 'r-', lw=2, label=f'Opt. P. Crête (FWHM: {calculate_fwhm(pulse_peak_power_optimized.T_ps, np.abs(pulse_peak_power_optimized.AT)**2)[0]*1000:.1f} fs)')
ax_comp.set_title('Comparaison des Étapes de Simulation et d\'Optimisation du Compresseur'); ax_comp.set_xlabel('Temps (ps)'); ax_comp.set_ylabel('Intensité Normalisée (ua)')
fwhm_f, center_f, xlim_f, _, _ = calculate_fwhm(pulse_peak_power_optimized.T_ps, np.abs(pulse_peak_power_optimized.AT)**2)
ax_comp.set_xlim(xlim_f); ax_comp.set_ylim(0, 1.1)
ax_comp.legend(fontsize='small'); ax_comp.grid(True); plt.tight_layout(); plt.show()

# --- Affichage de l'évolution dans la fibre (Optionnel) ---
# print("\nAffichage de l'évolution dans la fibre (mis à jour)...")
# plot_evolution_in_fiber(pulse_after_fiber, z_coords, AT_fiber, AW_fiber, fiber_length_m) # Décommentez et assurez-vous que la fonction est définie