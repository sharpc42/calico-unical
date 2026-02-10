import numpy as np
import sys

def simulate_thermal_noise(sigma_t_0, 
                           Nbls, 
                           seed,
                           verbose=True,):
    np.random.seed(seed)
    try:
        thermal_noise_real = np.random.normal(
            0.0,
            sigma_t_0,
            size=(Nbls),
        )
        thermal_noise_imag = np.random.normal(
            0.0,
            sigma_t_0,
            size=(Nbls),
        )
        return thermal_noise_real, thermal_noise_imag
    except:
        print(sys.exc_info())
        if verbose:
            print("Initial thermal noise failed. Was sigma_t set correctly?")

def simulate_model_error(Nbls,
                         sigma_e_0,
                         uv_norm_array, 
                         threshold_length, 
                         weighting_function, 
                         scaling_factor,
                         seed=42,
                         verbose=True,):
    np.random.seed(seed)
    if sigma_e_0 is not None and weighting_function == 'step_down_weights':
        if verbose:
            print("***STEP DOWN WEIGHTS IN SIMULATION***")
        # try:
        model_error_real_hi = np.zeros(Nbls)
        model_error_imag_hi = np.zeros(Nbls)
        model_error_real_lo = np.zeros(Nbls)
        model_error_imag_lo = np.zeros(Nbls)

        threshold_mask = uv_norm_array < threshold_length

        model_error_real_hi[~threshold_mask] += np.random.normal(
            0.0,
            1,
            size=(Nbls),
        )[~threshold_mask]
        model_error_imag_hi[~threshold_mask] += np.random.normal(
            0.0,
            1,
            size=(Nbls),
        )[~threshold_mask]
        model_error_real_lo[threshold_mask] += np.random.normal(
            0.0,
            1,
            size=(Nbls),
        )[threshold_mask]
        model_error_imag_lo[threshold_mask] += np.random.normal(
            0.0,
            1,
            size=(Nbls),
        )[threshold_mask]
        model_error_real = (model_error_real_hi + model_error_real_lo / np.sqrt(scaling_factor)) * sigma_e_0
        model_error_imag = (model_error_imag_hi + model_error_imag_lo / np.sqrt(scaling_factor)) * sigma_e_0
        model_error_real_long = model_error_real_hi[~threshold_mask] * sigma_e_0
        model_error_real_short = model_error_real_lo[threshold_mask] * sigma_e_0 / np.sqrt(scaling_factor)

        model_error_real_hi = np.random.normal(
            0.0,
            sigma_e_0,
            size=(Nbls),
        )
        model_error_imag_hi = np.random.normal(
            0.0,
            sigma_e_0,
            size=(Nbls),
        )
        return model_error_real_hi, model_error_imag_hi, model_error_real_long, model_error_real_short
        # except:
        #     print(sys.exc_info())
        #     print("Initial model error failed. Was sigma_e set correctly?")
    elif sigma_e_0 is not None and weighting_function == 'constant_weights':
        if verbose:
            print("***CONSTANT WEIGHTING FUNCTION IN SIMULATION***")
        model_error_real = np.random.normal(
            0.0,
            sigma_e_0,
            size=(Nbls),
        )
        model_error_imag = np.random.normal(
            0.0,
            sigma_e_0,
            size=(Nbls),
        )
        return model_error_real, model_error_imag, np.array([]), np.array([])
    else:
        print("Can't do model simulation - sigma_e_0 is not set")

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

def simulate_visibilities(caldata_obj, 
                          time_ind,
                          sigma_m=14,
                          sigma_vT=14,
                          seed=42,
                          true_vis_equals_model=True):
    np.random.seed(seed)
    caldata_obj.model_visibilities[time_ind, :, :, :] = np.random.normal(
                0,
                sigma_m,
                size=(
                    1,
                    caldata_obj.Nbls,
                    caldata_obj.Nfreqs,
                    caldata_obj.N_vis_pols,
                ),
            ) + 1.0j * np.random.normal(
                0,
                sigma_m,
                size=(
                    1,
                    caldata_obj.Nbls,
                    caldata_obj.Nfreqs,
                    caldata_obj.N_vis_pols,
                ),
            )
    if true_vis_equals_model:
        caldata_obj.data_visibilities = caldata_obj.model_visibilities.copy()
    else:
        caldata_obj.data_visibilities[time_ind, :, :, :] = np.random.normal(
                0,
                sigma_vT,
                size=(
                    1,
                    caldata_obj.Nbls,
                    caldata_obj.Nfreqs,
                    caldata_obj.N_vis_pols,
                ),
            ) + 1.0j * np.random.normal(
                0,
                sigma_vT,
                size=(
                    1,
                    caldata_obj.Nbls,
                    caldata_obj.Nfreqs,
                    caldata_obj.N_vis_pols,
                ),
            )