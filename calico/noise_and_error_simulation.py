import numpy as np
import sys

def simulate_thermal_noise(caldata_obj, seed):
    np.random.seed(seed)
    try:
        thermal_noise_real = np.random.normal(
            0.0,
            caldata_obj.sigma_t_0,
            size=(caldata_obj.Nbls),
        )
        thermal_noise_imag = np.random.normal(
            0.0,
            caldata_obj.sigma_t_0,
            size=(caldata_obj.Nbls),
        )
        caldata_obj.data_visibilities[0,:,0,0] += thermal_noise_real + 1.0j * thermal_noise_imag
    except:
        print(sys.exc_info())
        print("Initial thermal noise failed. Was sigma_t set correctly?")

def simulate_model_error(caldata_obj, seed, scaling_factor):
    np.random.seed(seed)
    # try:
    model_error_real_hi = np.zeros(caldata_obj.Nbls)
    model_error_imag_hi = np.zeros(caldata_obj.Nbls)
    model_error_real_lo = np.zeros(caldata_obj.Nbls)
    model_error_imag_lo = np.zeros(caldata_obj.Nbls)

    model_error_real_hi[~caldata_obj.threshold_mask] += np.random.normal(
        0.0,
        caldata_obj.sigma_e_0,
        size=(caldata_obj.Nbls),
    )[~caldata_obj.threshold_mask]
    model_error_imag_hi[~caldata_obj.threshold_mask] += np.random.normal(
        0.0,
        caldata_obj.sigma_e_0,
        size=(caldata_obj.Nbls),
    )[~caldata_obj.threshold_mask]
    model_error_real_lo[caldata_obj.threshold_mask] += np.random.normal(
        0.0,
        caldata_obj.sigma_e_0 * scaling_factor,
        size=(caldata_obj.Nbls),
    )[caldata_obj.threshold_mask]
    model_error_imag_lo[caldata_obj.threshold_mask] += np.random.normal(
        0.0,
        caldata_obj.sigma_e_0 * scaling_factor,
        size=(caldata_obj.Nbls),
    )[caldata_obj.threshold_mask]
    caldata_obj.data_visibilities[0,:,0,0] += (model_error_real_hi + model_error_real_lo
                                                + 1.0j * (model_error_imag_hi + model_error_imag_lo))
    # except:
    #     print(sys.exc_info())
    #     print("Initial model error failed. Was sigma_m set correctly?")

def format_sim_weights_per_baseline(caldata_obj, scaling_factor, threshold_length=50):
    simulation_sigma_per_baseline = np.heaviside(caldata_obj.uv_norm - threshold_length, 1)
    simulation_sigma_per_baseline[caldata_obj.threshold_mask] += scaling_factor
    return simulation_sigma_per_baseline

def plot_weights_per_baseline(caldata_obj, weight_array, scaling_factor, threshold_length=50):
    if scaling_factor != 1:
        import dev_tools
        dev = dev_tools.DevTools()
        dev.plot_weights_per_baseline(
            caldata_obj.uv_norm,
            weight_array,
            weighting_function="Step Down",
            scaling_factor=scaling_factor,
            threshold_length=threshold_length,
            sigma="sigma_e",
            ylim=11,
        )