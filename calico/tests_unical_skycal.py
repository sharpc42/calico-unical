import numpy as np
import calibration_optimization
import calibration_wrappers
import cost_function_calculations
import calibration_qa
import caldata
import pyuvdata
import os
import unittest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class TestStringMethods(unittest.TestCase):

    def test_common_setup_handling(self):
        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/tutorial_full_onetime_unflagged.uvfits")
        data = model.copy()
        caldata_obj = caldata.CalData()
        caldata_obj.load_data(
            data,
            model,
        )
        gains_flattened_skycal = calibration_optimization.run_skycal_optimization_per_pol_single_freq(
            caldata_obj=caldata_obj,
            xtol=1e-5,
            maxiter=200,
            dev_type="test gains flattened",
        )
        gains_flattened_unical = calibration_optimization.run_unical_optimization(
            caldata_obj=caldata_obj,
            xtol=1e-5,
            maxiter=200,
            dev_type="test gains flattened",
        )
        assert np.any(gains_flattened_skycal)
        assert np.any(gains_flattened_unical)
        np.testing.assert_allclose(
            gains_flattened_skycal,
            gains_flattened_unical,
        )
        gains_rolled_skycal = calibration_optimization.run_skycal_optimization_per_pol_single_freq(
            caldata_obj=caldata_obj,
            xtol=1e-5,
            maxiter=200,
            dev_type="test gains rolled",
        )
        gains_rolled_unical = calibration_optimization.run_unical_optimization(
            caldata_obj=caldata_obj,
            xtol=1e-5,
            maxiter=200,
            dev_type="test gains rolled",
        )
        assert np.any(gains_rolled_skycal)
        assert np.any(gains_rolled_unical)
        np.testing.assert_allclose(
            gains_rolled_skycal,
            gains_rolled_unical,
        )
    
    def test_gains_param_rolling(self):
        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/tutorial_full_onetime_unflagged.uvfits")
        data = model.copy()
        caldata_obj = caldata.CalData()
        caldata_obj.load_data(
            data,
            model,
        )
        gains_rolled_unical = calibration_optimization.run_unical_optimization(
            caldata_obj=caldata_obj,
            xtol=1e-5,
            maxiter=200,
            dev_type="test gains rolled",
        )
        orig_gains = caldata_obj.gains[caldata_obj.ant_inds, 0, 0]
        assert np.any(orig_gains)
        assert np.any(gains_rolled_unical)
        np.testing.assert_allclose(
            orig_gains,
            gains_rolled_unical,
        )

    def test_fit_model_param_rolling(self):
        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/tutorial_full_onetime_unflagged.uvfits")
        data = model.copy()
        caldata_obj = caldata.CalData()
        caldata_obj.load_data(
            data,
            model,
        )
        fit_vis_rolled = calibration_optimization.run_unical_optimization(
            caldata_obj=caldata_obj,
            xtol=1e-5,
            maxiter=200,
            dev_type="test fit vis rolled",
        )
        orig_fit_vis = caldata_obj.fit_vis[0, caldata_obj.bl_inds, 0, 0].copy()
        assert np.any(orig_fit_vis)
        assert np.any(fit_vis_rolled)
        np.testing.assert_allclose(
            orig_fit_vis,
            fit_vis_rolled,
        )

    def test_cost_function_return_same_for_same_form(self):
        import time
        import noise_and_error_simulation as sim
        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/tutorial_full_onetime_unflagged.uvfits")
        data = model.copy()
        caldata_obj = caldata.CalData()
        caldata_obj.load_data(
            data,
            model,
        )
        n_real, n_imag = sim.simulate_thermal_noise(
            sigma_t_0=3, 
            Nbls=caldata_obj.Nbls, 
            seed=int(time.time()),
        )
        e_real, e_imag, _, _ = sim.simulate_model_error(
            Nbls=caldata_obj.Nbls, 
            sigma_e_0=4, 
            uv_norm_array=caldata_obj.uv_norm,
            threshold_length=0,
            weighting_function="constant_weights",
            scaling_factor=1,
            seed=int(time.time()),
        )
        caldata_obj.data_visibilities[0,:,0,0] += n_real + 1j*n_imag
        caldata_obj.model_visibilities[0,:,0,0] += e_real + 1j*e_imag

        cost_one_run_skycal = calibration_optimization.run_skycal_optimization_per_pol_single_freq(
            caldata_obj=caldata_obj,
            xtol=1e-5,
            maxiter=200,
            dev_type="test gains one run skycal",
        )
        cost_one_run_unical = calibration_optimization.run_unical_optimization(
            caldata_obj=caldata_obj,
            xtol=1e-5,
            maxiter=200,
            dev_type="test gains one run skycal",
        )

        np.testing.assert_allclose(cost_one_run_skycal, cost_one_run_unical)

    def calibration_grid_search():
        import time
        import noise_and_error_simulation as sim
        import matplotlib.pyplot as plt
        import dev_tools
        import subprocess
        from datetime import datetime
        import variable_weights

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/tutorial_full_onetime_unflagged.uvfits")
        data = model.copy()
        # Ensure data and model are phased the same
        data.phase_to_time(np.mean(data.time_array))
        model.phase_to_time(np.mean(data.time_array))
        caldata_obj = caldata.CalData()
        caldata_obj.load_data(
            data,
            model,
            gain_init_calfile=None,
            gain_init_to_vis_ratio=True,
            gains_multiply_model=True,
            gain_init_stddev=0.0,
            glim=None,
            ulim=None,
            weighting_function="constant_weights",
            simulate_visibilities=True,
            sigma_t_0=1,
            sigma_m_0=1,
            scaling_factor_cost=1,
            threshold_length=0,
        )
        original_model = caldata_obj.model_visibilities[0,:,0,0].copy()
        original_data = caldata_obj.data_visibilities[0,:,0,0].copy()
        sigma_t_scales  = np.arange(  0, 10, 7/3)
        sigma_m_scales  = np.arange(-10, 10, 7/3)
        real_avg_g_minus_1_unical_truth = []
        real_avg_g_minus_1_unical_skyapprox = []
        real_avg_g_minus_1_skycal = []
        calc_avg_abs_model_errors = []
        calc_avg_abs_thermal_noise = []
        for sigma_t in sigma_t_scales:
            for sigma_m in sigma_m_scales:
                print(f"\n\n***Testing***\n\t{sigma_t=:.2f} {sigma_m=:.2f}\n\n")
                caldata_obj.gains[:,0,0] = 1
                caldata_obj.data_visibilities[0,:,0,0] = original_data
                caldata_obj.model_visibilities[0,:,0,0] = original_model
                e_real, e_imag, _, _ = sim.simulate_model_error(
                    Nbls=caldata_obj.Nbls, 
                    sigma_e_0=max(np.abs(sigma_m),0.1), 
                    uv_norm_array=caldata_obj.uv_norm,
                    threshold_length=0,
                    weighting_function="constant_weights",
                    scaling_factor=1,
                    # seed=int(time.time()),
                    seed=1,
                )
                if sigma_m < 0:
                    caldata_obj.model_visibilities[0,:,0,0] += e_real + 1j*e_imag
                else:
                    caldata_obj.data_visibilities[0,:,0,0] += e_real + 1j*e_imag
                n_real, n_imag = sim.simulate_thermal_noise(
                    sigma_t_0=max(sigma_t,0.1), 
                    Nbls=caldata_obj.Nbls, 
                    seed=1,
                )
                data_vT = caldata_obj.data_visibilities[0,:,0,0].copy()
                caldata_obj.data_visibilities[0,:,0,0] += n_real + 1j*n_imag
                gains = calibration_optimization.run_skycal_optimization_per_pol_single_freq(
                    caldata_obj=caldata_obj,
                    xtol=1e-5,
                    maxiter=200,
                )
                real_avg_g_minus_1_skycal.append(np.mean(gains).real - 1)
                vwa = variable_weights.VariableWeightsArray()
                scaling_factors = [1,100000]
                for scaling_factor in scaling_factors:
                    vwa.set_algorithm_weights(
                        caldata_obj,
                        sigma_t_0 = max(sigma_t, 0.1),
                        sigma_m_0 = sigma_m,
                        threshold_length = 0,
                        weighting_function = "constant_weights",
                        scaling_factor = scaling_factor,
                    )
                    gains, _ = calibration_optimization.run_unical_optimization(
                        caldata_obj=caldata_obj,
                        xtol=1e-5,
                        maxiter=200,
                    )
                    if scaling_factor == 1:
                        real_avg_g_minus_1_unical_truth.append(np.mean(gains).real - 1)
                        vT_minus_m = data_vT - caldata_obj.model_visibilities[0,:,0,0]
                        thermal_noise = caldata_obj.data_visibilities[0,:,0,0] - data_vT
                        avg_abs_model_error = np.mean(np.abs(vT_minus_m))
                        if sigma_m < 0: 
                            avg_abs_model_error *= -1
                        calc_avg_abs_model_errors.append(avg_abs_model_error)
                        calc_avg_abs_thermal_noise.append(np.std(thermal_noise.real))
                    else:
                        real_avg_g_minus_1_unical_skyapprox.append(np.mean(gains).real - 1)
        start_time = time.time()
        start_time_suffix = str(start_time)
        git_hash = str(subprocess.check_output(['git','rev-parse','--short','HEAD']).decode('ascii').strip())
        git_time_suffix = f"g{git_hash}_t{start_time_suffix}"
        file_suffix = git_time_suffix
        start_time_dt = datetime.fromtimestamp(start_time)
        metadata = {
            "Date"                   : f"{start_time_dt:%B %d, %Y}",
            "Time"                   : f"{start_time_dt:%H:%M:%S}",
            "Sigma_t Vals"           : ",".join(f"{noise:.3f}" for noise in sigma_t_scales.tolist()),
            "Sigma_m Vals"           : ",".join(f"{error:.3f}" for error in sigma_m_scales.tolist()),
            "Scaling Factors (Cost)" : "None",
            "Scaling Factor (Sim)"   : "1",
            "Git ID"                 : git_hash,
            "Time ID"                : start_time_suffix,
            "Weighting Function"     : "constant_weights",
            "Optimization Function"  : "None",
        }
        print(f"Unical gains arrays equal for both scaling factors? {np.array_equal(real_avg_g_minus_1_unical_truth, real_avg_g_minus_1_unical_skyapprox)}")
        dev_tools.plot_3d_data_as_2d_hist(
            x_array     = np.asarray(calc_avg_abs_model_errors),
            y_array     = np.asarray(calc_avg_abs_thermal_noise),
            z_array     = np.asarray(real_avg_g_minus_1_unical_truth),
            plot_title  = f"<Re(g)>-1 vs $\\sigma_t$ & $\\sigma_m$ GMM: {caldata_obj.gains_multiply_model}\n(Calculated, Unical - Scalefactor: 1)",
            plot_xlabel = "$<|m - v_T|>$",
            plot_ylabel = "$<|v - v_T|>$",
            plot_vmax   = 0.3,
            plot_vmin   = -0.3,
            # plot_xlim_h = 1,
            # plot_ylim_h = 1,
            # plot_ylim_l = -1,
            # plot_xlim_l = -1,
            filename    = f'{THIS_DIR}/images/grid_search_{file_suffix}_unical_truth.png',
            plot_cmap   = "PuOr",
            cmap_label  = "$Re(g)-1$",
            suffix      = file_suffix,
            metadata    = metadata,
        )
        dev_tools.plot_3d_data_as_2d_hist(
            x_array     = np.asarray(calc_avg_abs_model_errors),
            y_array     = np.asarray(calc_avg_abs_thermal_noise),
            z_array     = np.asarray(real_avg_g_minus_1_unical_skyapprox),
            plot_title  = f"<Re(g)>-1 vs $\\sigma_t$ & $\\sigma_m$ GMM: {caldata_obj.gains_multiply_model}\n(Calculated, Unical - Scalefactor: 100000)",
            plot_xlabel = "$<|m - v_T|>$",
            plot_ylabel = "$<|v - v_T|>$",
            plot_vmax   = 0.3,
            plot_vmin   = -0.3,
            # plot_xlim_h = 1,
            # plot_ylim_h = 1,
            # plot_ylim_l = -1,
            # plot_xlim_l = -1,
            filename    = f'{THIS_DIR}/images/grid_search_{file_suffix}_unical_skyapprox.png',
            plot_cmap   = "PuOr",
            cmap_label  = "$Re(g)-1$",
            suffix      = file_suffix,
            metadata    = metadata,
        )
        dev_tools.plot_3d_data_as_2d_hist(
            x_array     = np.asarray(calc_avg_abs_model_errors),
            y_array     = np.asarray(calc_avg_abs_thermal_noise),
            z_array     = np.asarray(real_avg_g_minus_1_skycal),
            plot_title  = f"<Re(g)>-1 vs $\\sigma_t$ & $\\sigma_m$ GMM: {caldata_obj.gains_multiply_model}\n(Calculated, Skycal)",
            plot_xlabel = "$<|m - v_T|>$",
            plot_ylabel = "$<|v - v_T|>$",
            plot_vmax   = 0.3,
            plot_vmin   = -0.3,
            # plot_xlim_h = 1,
            # plot_ylim_h = 1,
            # plot_ylim_l = -1,
            # plot_xlim_l = -1,
            filename    = f'{THIS_DIR}/images/grid_search_{file_suffix}_skycal.png',
            plot_cmap   = "PuOr",
            cmap_label  = "$Re(g)-1$",
            suffix      = file_suffix,
            metadata    = metadata,
        )
        

    def plot_skycal_unical_diff_per_scaling_factor():
        import time
        import noise_and_error_simulation as sim
        import variable_weights
        import matplotlib.pyplot as plt
        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/tutorial_full_onetime_unflagged.uvfits")
        data = model.copy()
        caldata_obj = caldata.CalData()
        caldata_obj.load_data(
            data,
            model,
        )
        n_real, n_imag = sim.simulate_thermal_noise(
            sigma_t_0=3, 
            Nbls=caldata_obj.Nbls, 
            seed=int(time.time()),
        )
        e_real, e_imag, _, _ = sim.simulate_model_error(
            Nbls=caldata_obj.Nbls, 
            sigma_e_0=8, 
            uv_norm_array=caldata_obj.uv_norm,
            threshold_length=0,
            weighting_function="constant_weights",
            scaling_factor=1,
            seed=int(time.time()),
        )
        caldata_obj.data_visibilities[0,:,0,0] += n_real + 1j*n_imag
        caldata_obj.model_visibilities[0,:,0,0] += e_real + 1j*e_imag
        # caldata_obj.data_visibilities[0,:,0,0] += e_real + 1j*e_imag
        caldata_obj.gains_multiply_model = True
        gains_skycal = calibration_optimization.run_skycal_optimization_per_pol_single_freq(
            caldata_obj=caldata_obj,
            xtol=1e-5,
            maxiter=200,
        )
        # avg_mag_skycal = np.mean(np.abs(gains_skycal))
        avg_real_skycal = np.mean(gains_skycal).real
        gains_diff = []
        # iterate through algorithm weights for unical
        vwa = variable_weights.VariableWeightsArray()
        scaling_factors_lo = [scaling_factor for scaling_factor in range(1,100,5)]
        scaling_factors_hi = [scaling_factor for scaling_factor in range(100,100000,19980)]
        scaling_factors = scaling_factors_lo + scaling_factors_hi
        sigma_m = 1
        for scaling_factor in scaling_factors:
            vwa.set_algorithm_weights(
                caldata_obj,
                sigma_t_0 = 1,
                sigma_m_0 = sigma_m,
                threshold_length = 0,
                weighting_function = "constant_weights",
                scaling_factor = scaling_factor,
            )
            gains_unical, _ = calibration_optimization.run_unical_optimization(
                caldata_obj=caldata_obj,
                xtol=1e-5,
                maxiter=200,
            )
            print(f"***Avg Mag Gains - Unical: {np.mean(gains_unical).real:.3f}***")
            gains_diff.append(np.mean(gains_unical.real) - avg_real_skycal)
        print(f"{caldata_obj.gains_multiply_model=}")
        plt.plot(scaling_factors, gains_diff)
        plt.title(f"Unical - Skycal Avg Mag Gains vs Scaling Factor (~$1/\\sigma_m^2$)\n$\\sigma_m = {sigma_m}$, $m < v_T$")
        plt.xlabel("Scaling Factor (~$1/\\sigma_m^2$)")
        plt.ylabel("$Re<g_u> - Re<g_s>$")
        plt.show()

    def compare_optimizers():
        import copy
        import time
        import noise_and_error_simulation as sim
        import variable_weights
        import matplotlib.pyplot as plt
        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/tutorial_full_onetime_unflagged.uvfits")
        data = model.copy()
        caldata_obj = caldata.CalData()
        caldata_obj.load_data(
            data,
            model,
        )
        org_gains = copy.deepcopy(caldata_obj.gains[:,0,0])
        org_fit_vis = copy.deepcopy(caldata_obj.fit_vis[0,:,0,0])
        n_real, n_imag = sim.simulate_thermal_noise(
            sigma_t_0=3, 
            Nbls=caldata_obj.Nbls, 
            seed=int(time.time()),
        )
        e_real, e_imag, _, _ = sim.simulate_model_error(
            Nbls=caldata_obj.Nbls, 
            sigma_e_0=3, 
            uv_norm_array=caldata_obj.uv_norm,
            threshold_length=0,
            weighting_function="constant_weights",
            scaling_factor=1,
            seed=int(time.time()),
        )
        caldata_obj.data_visibilities[0,:,0,0] += n_real + 1j*n_imag
        # caldata_obj.model_visibilities[0,:,0,0] += e_real + 1j*e_imag  # m > v_T
        caldata_obj.data_visibilities[0,:,0,0] += e_real + 1j*e_imag  # m < v_T
        caldata_obj.gains_multiply_model = True
        vwa = variable_weights.VariableWeightsArray()
        vwa.set_algorithm_weights(
            caldata_obj,
            sigma_t_0 = 0.45,
            sigma_m_0 = 0.6,
            threshold_length = 0,
            weighting_function = "constant_weights",
            scaling_factor = 1e4,
        )
        optimizers = [
            "pytorch",
            "powell",
        ]
        opt_gains = []
        for optimizer in optimizers:
            caldata_obj.gains[:,0,0] = org_gains
            caldata_obj.fit_vis[0,:,0,0] = org_fit_vis
            gains, _ = calibration_optimization.run_unical_optimization(
                caldata_obj=caldata_obj,
                xtol=1e-5,
                maxiter=200,
                optimization_scheme=optimizer,
            )
            print(f"{optimizer=}\n\n{gains=}")
            opt_gains.append(gains)
        lbfgs_gains = opt_gains[0]
        powell_gains = opt_gains[1]  # make sure these track the correct optimizer order...
        plot_time = int(time.time())
        abs_powell_minus_lbfgs = np.abs(powell_gains) - np.abs(lbfgs_gains)
        print(f"Plotting abs diff plot")
        x_arr = [x for x in range(abs_powell_minus_lbfgs.size)]
        plt.plot(x_arr, abs_powell_minus_lbfgs)
        plt.title("$|g_P| - |g_L|$ per antenna")
        plt.ylabel("$|g_P| - |g_L|$")
        plt.xlabel("Antennas")
        plt.savefig(f"calico/images/powell_lbfgs_diff_abs_{plot_time}.png")
        plt.close()
        real_powell_minus_lbfgs = powell_gains.real - lbfgs_gains.real
        print(f"Plotting real diff plot")
        x_arr = [x for x in range(real_powell_minus_lbfgs.size)]
        plt.plot(x_arr, real_powell_minus_lbfgs)
        plt.title("$Re(g_P) - Re(g_L)$ per antenna")
        plt.ylabel("$Re(g_P) - Re(g_L)$")
        plt.xlabel("Antennas")
        plt.savefig(f"calico/images/powell_lbfgs_diff_real_{plot_time}.png")
        plt.close()
        imag_powell_minus_lbfgs = powell_gains.imag - lbfgs_gains.imag
        print(f"Plotting imag diff plot")
        x_arr = [x for x in range(imag_powell_minus_lbfgs.size)]
        plt.plot(x_arr, imag_powell_minus_lbfgs)
        plt.title("$Im(g_P) - Im(g_L)$ per antenna")
        plt.ylabel("$Im(g_P) - Im(g_L)$")
        plt.xlabel("Antennas")
        plt.savefig(f"calico/images/powell_lbfgs_diff_imag_{plot_time}.png")
        plt.close()
        print(f"Plotting scatter plot in nsew-plane (abs diff)")
        en_plane = caldata_obj.antenna_positions[:,:-1]
        print(f"en-plane shape - {en_plane.shape}")
        max_diff = np.max([np.max(abs_powell_minus_lbfgs), np.abs(np.min(abs_powell_minus_lbfgs))])
        scatter_plot = plt.scatter(
            x=en_plane[:,0], 
            y=en_plane[:,1],
            c=abs_powell_minus_lbfgs,
            vmax=max_diff,
            vmin=-max_diff,
        )
        cbar = plt.colorbar(scatter_plot)
        cbar.set_label("$|g_P| - |g_L|$")
        plt.title("$|g_P| - |g_L|$ per antenna in en-plane")
        plt.xlabel("East")
        plt.ylabel("North")
        plt.savefig(f"calico/images/powell_lbfgs_en_plane_{plot_time}.png")
        plt.close()

    def plot_aggregate_montecarlos():
        import matplotlib.pyplot as plt
        ne5_arr = [
            1.022,
            1.003,
            1.017,
            1.033,
            1.007,
            1.006,
            1.020,
            1.013,	
            1.008,
            1.010,
            1.015,
            1.000,
        ]
        ne10_arr = [
            1.264,
            1.019,
            1.053,
            1.026,
            1.035,
            1.078,
            1.019,
            1.040,
            1.053,
            1.040,
            1.022,
            1.063,
            1.078,
        ]
        ne20_arr = [
            2.397,
            1.562,
            1.866,
            1.081,
            1.161,
            1.232,
            1.136,
            1.674,
            2.160,
            3.230,
        ]
        plt.hist(ne5_arr, bins=5, color="blue")
        plt.title("Averages of Monte Carlos with 1e7 samples (5 Jy error/noise)")
        plt.xlim(0.99,1.05)
        plt.savefig("calico/images/hist_ne5means")
        plt.close()

        plt.hist(ne10_arr, bins=7, color="orange")
        plt.title("Averages of Monte Carlos with 1e7 samples (10 Jy error/noise)")
        plt.xlim(0.95,1.3)
        plt.savefig("calico/images/hist_ne10means")
        plt.close()

        plt.hist(ne20_arr, bins=10, color="green")
        plt.title("Averages of Monte Carlos with 1e7 samples (20 Jy error/noise)")
        plt.xlim(0.9,3.5)
        plt.savefig("calico/images/hist_ne20means")
        plt.close()

    def montecarlo():
        import numpy as np
        import matplotlib.pyplot as plt
        import time

        n_samples = 1e7
        abs_avg_sum_vals = []  # |<|v|^2 + n*e + vn* + v*e>|
        sum_abs_mag_vals = []  # |<|v|^2>| + |<n*e>| + |<vn*>| + |<v*e>|
        subtract_off_v   = []  # |<v^2 + n*e + vn* + v*e>| - |<|v|^2>|
        n_and_e_terms    = []  # |<n*e + v*e + vn*>|
        avg_sum          = []  # <|v|^2 + n*e + v* + v*e>
        avg_v_squared    = []  # <|v|^2>
        avg_ne_arr       = []  # |<n*e>|
        avg_vn_arr       = []  # |<vn*>|
        avg_ve_arr       = []  # |<v*e>|
        e_arr            = []

        # roll data vis once
        org_true_data_visibilities = np.random.normal(
            loc=0,
            scale=14,
            size=(1,6,1,1),
        ) + 1.0j * np.random.normal(
            loc=0,
            scale=14,
            size=(1,6,1,1),
        )

        for i in range(int(n_samples)):
            np.random.seed(int(time.time()))
            model_error = np.random.normal(
                loc=0, 
                scale=5,
                size=(1,6,1,1),
            ) + 1.0j * np.random.normal(
                loc=0,
                scale=5,
                size=(1,6,1,1),
            )
            thermal_noise = np.random.normal(
                loc=0,
                scale=5,
                size=(1,6,1,1),
            ) + 1.0j * np.random.normal(
                loc=0,
                scale=5,
                size=(1,6,1,1),
            )
            # quantities
            ne = np.conj(thermal_noise) * model_error
            vn = org_true_data_visibilities * np.conj(thermal_noise)
            ve = np.conj(org_true_data_visibilities) * model_error
            abs_v_squared = np.abs(org_true_data_visibilities)**2
            avg_ne = np.mean(ne)
            avg_vn = np.mean(vn)
            avg_ve = np.mean(ve)
            avg_abs_v_squared = np.mean(abs_v_squared)
            # expressions
            abs_avg_sum = np.abs(
                np.mean(abs_v_squared + ne + ve + vn)
            )
            sum_abs_mag = np.abs(avg_abs_v_squared) + np.abs(avg_ne) + \
                        np.abs(avg_vn) + np.abs(avg_ve)
            denominator = avg_abs_v_squared + np.mean(np.abs(model_error)**2)
            # fill arrays
            abs_avg_sum_vals.append(
                abs_avg_sum / avg_abs_v_squared
            )
            sum_abs_mag_vals.append(
                sum_abs_mag / 1
            )
            subtract_off_v.append(
                abs_avg_sum - avg_abs_v_squared
            )
            n_and_e_terms.append(
                np.abs(np.mean(ne + vn + ve))
            )
            avg_sum.append(
                np.mean(avg_abs_v_squared + ve + vn + ne)
            )
            avg_v_squared.append(
                avg_abs_v_squared
            )
            avg_ne_arr.append(
                np.abs(avg_ne)
            )
            avg_ve_arr.append(
                np.abs(avg_ve)
            )
            avg_vn_arr.append(
                np.abs(avg_vn)
            )
            e_arr.append(
                model_error
            )

        abs_avg_sum_mean = np.mean(abs_avg_sum_vals)
        sum_abs_mag_mean = np.mean(sum_abs_mag_vals)
        subtract_off_v_mean = np.mean(subtract_off_v)
        n_and_e_terms_mean = np.mean(n_and_e_terms)
        avg_sum_mean = np.mean(avg_sum)
        avg_ne_mean = np.mean(avg_ne_arr)
        avg_ve_mean = np.mean(avg_ve_arr)
        avg_vn_mean = np.mean(avg_vn_arr)
        avg_v_squared_mean = np.mean(avg_v_squared)
        model_error_mean = np.mean(e_arr)

        print(f"\n\n***RESULTS***\n{abs_avg_sum_mean=:.3f}\t{sum_abs_mag_mean=:.3f}")
        print(f"{subtract_off_v_mean=:.3f}\t{n_and_e_terms_mean=:.3f}")
        print(f"{avg_v_squared_mean=:.3f}\t{avg_ne_mean=:.3f}")
        print(f"{avg_ve_mean=:.3f}\t{avg_vn_mean=:.3f}")

        new_model_error = np.random.normal(
            loc=0,
            scale=5,
            size=(1,6,1,1),
        ) + 1.0j * np.random.normal(
            loc=0,
            scale=5,
            size=(1,6,1,1),
        )
        avg_model_error_squared = np.mean(np.abs(new_model_error)**2)
        org_model_visibilities = org_true_data_visibilities.copy()

        print(f"\n***MODEL VISIBILITIES - CASE m = vT + e***")
        new_model_visibilities = org_true_data_visibilities + new_model_error
        avg_abs_m_squared = np.mean(np.abs(new_model_visibilities)**2)
        avg_abs_vT_squared = np.mean(np.abs(org_true_data_visibilities)**2)
        print(f"$<|m|^2>$ {avg_abs_m_squared:.3f}")
        print(f"$<|v_T|^2>$ {avg_abs_vT_squared:.3f}\t$<|e|^2>$ {avg_model_error_squared:.3f}")
        print(f"$<|v_T|^2> + <|e|^2>$ {avg_abs_vT_squared + avg_model_error_squared:.3f}")
        print(f"$<|m|^2> - <|v_T|^2> - <|e|^2>$ {avg_abs_m_squared - avg_abs_vT_squared - avg_model_error_squared:.3f}")
        print(f"$<|v_T|^2> - <|m|^2> - <|e|^2>$ {avg_abs_vT_squared - avg_abs_m_squared - avg_model_error_squared:.3f}")

        print(f"\n***MODEL VISIBILITIES - CASE vT = m + e***")
        new_true_data_visibilities = org_model_visibilities + new_model_error
        avg_abs_m_squared = np.mean(np.abs(org_model_visibilities)**2)
        avg_abs_vT_squared = np.mean(np.abs(new_true_data_visibilities)**2)
        print(f"$<|m|^2>$ {avg_abs_m_squared:.3f}")
        print(f"$<|v_T|^2>$ {avg_abs_vT_squared:.3f}\t$<|e|^2>$ {avg_model_error_squared:.3f}")
        print(f"$<|v_T|^2> + <|e|^2>$ {avg_abs_vT_squared + avg_model_error_squared:.3f}")
        print(f"$<|m|^2> - <|v_T|^2> - <|e|^2>$ {avg_abs_m_squared - avg_abs_vT_squared - avg_model_error_squared:.3f}")
        print(f"$<|v_T|^2> - <|m|^2> - <|e|^2>$ {avg_abs_vT_squared - avg_abs_m_squared - avg_model_error_squared:.3f}")

        fig, ax = plt.subplots()
        plt.hist(abs_avg_sum_vals, bins=50)
        plt.title("$\\frac{|< |v_T|^2 + v_T^* e + n^* v_T + n^* e >|}{<|v_T|^2>}$")
        plt.xlabel("Realizations")
        props = dict(boxstyle='round', color="wheat", alpha=0.7)
        plt.text(x=0.85, 
                y=0.95, 
                s=f"Mean: {np.mean(abs_avg_sum_vals):.3f}\nStd: {np.std(abs_avg_sum_vals):.3f}", 
                fontsize=12,
                verticalalignment='top', 
                bbox=props, 
                ha="center", 
                va="top",
                transform=ax.transAxes,
        )
        # plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # unittest.main()
    TestStringMethods.compare_optimizers()
    # TestStringMethods.plot_skycal_unical_diff_per_scaling_factor()
    # TestStringMethods.plot_montecarlos()