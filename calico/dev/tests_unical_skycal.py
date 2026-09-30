import copy
import os
import subprocess
import time
import unittest
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np

import pyuvdata
from calico import caldata, calibration_optimization as calopt
from calico.dev import dev_tools, noise_and_error_simulation as sim, variable_weights

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestStringMethods(unittest.TestCase):
    def test_common_setup_handling(self):
        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/tutorial_full_onetime_unflagged.uvfits")
        data = model.copy()
        caldata_obj = caldata.CalData()
        caldata_obj.load_data(data, model)
        gains_flattened_skycal = calopt.run_skycal_optimization_per_pol_single_freq(
            caldata_obj=caldata_obj,
            xtol=1e-5,
            maxiter=200,
            dev_type="test gains flattened",
        )
        gains_flattened_unical = calopt.run_unical_optimization(
            caldata_obj=caldata_obj,
            xtol=1e-5,
            maxiter=200,
            dev_type="test gains flattened",
        )
        assert np.any(gains_flattened_skycal)
        assert np.any(gains_flattened_unical)
        np.testing.assert_allclose(gains_flattened_skycal, gains_flattened_unical)
        gains_rolled_skycal = calopt.run_skycal_optimization_per_pol_single_freq(
            caldata_obj=caldata_obj,
            xtol=1e-5,
            maxiter=200,
            dev_type="test gains rolled",
        )
        gains_rolled_unical = calopt.run_unical_optimization(
            caldata_obj=caldata_obj,
            xtol=1e-5,
            maxiter=200,
            dev_type="test gains rolled",
        )
        assert np.any(gains_rolled_skycal)
        assert np.any(gains_rolled_unical)
        np.testing.assert_allclose(gains_rolled_skycal, gains_rolled_unical)

    def test_gains_param_rolling(self):
        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/tutorial_full_onetime_unflagged.uvfits")
        data = model.copy()
        caldata_obj = caldata.CalData()
        caldata_obj.load_data(data, model)
        gains_rolled_unical = calopt.run_unical_optimization(
            caldata_obj=caldata_obj,
            xtol=1e-5,
            maxiter=200,
            dev_type="test gains rolled",
        )
        orig_gains = caldata_obj.gains[caldata_obj.ant_inds, 0, 0]
        assert np.any(orig_gains)
        assert np.any(gains_rolled_unical)
        np.testing.assert_allclose(orig_gains, gains_rolled_unical)

    def test_fit_model_param_rolling(self):
        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/tutorial_full_onetime_unflagged.uvfits")
        data = model.copy()
        caldata_obj = caldata.CalData()
        caldata_obj.load_data(data, model)
        fit_vis_rolled = calopt.run_unical_optimization(
            caldata_obj=caldata_obj,
            xtol=1e-5,
            maxiter=200,
            dev_type="test fit vis rolled",
        )
        orig_fit_vis = caldata_obj.fit_vis[0, caldata_obj.bl_inds, 0, 0].copy()
        assert np.any(orig_fit_vis)
        assert np.any(fit_vis_rolled)
        np.testing.assert_allclose(orig_fit_vis, fit_vis_rolled)

    def test_cost_function_return_same_for_same_form(self):
        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/tutorial_full_onetime_unflagged.uvfits")
        data = model.copy()
        caldata_obj = caldata.CalData()
        caldata_obj.load_data(data, model)
        n_real, n_imag = sim.simulate_thermal_noise(
            sigma_t_0=3,
            n_bls=caldata_obj.Nbls,
            n_times=caldata_obj.Ntimes,
            seed=int(time.time()),
        )
        e_real, e_imag, _, _ = sim.simulate_model_error(
            n_bls=caldata_obj.Nbls,
            n_times=caldata_obj.Ntimes,
            sigma_e_0=4,
            uv_norm_array=caldata_obj.uv_norm,
            threshold_length=0,
            weighting_function="constant_weights",
            scaling_factor=1,
            seed=int(time.time()),
        )
        caldata_obj.data_visibilities[0, :, 0, 0] += n_real + 1j * n_imag
        caldata_obj.model_visibilities[0, :, 0, 0] += e_real + 1j * e_imag

        cost_one_run_skycal = calopt.run_skycal_optimization_per_pol_single_freq(
            caldata_obj=caldata_obj,
            xtol=1e-5,
            maxiter=200,
            dev_type="test gains one run skycal",
        )
        cost_one_run_unical = calopt.run_unical_optimization(
            caldata_obj=caldata_obj,
            xtol=1e-5,
            maxiter=200,
            dev_type="test gains one run skycal",
        )

        np.testing.assert_allclose(cost_one_run_skycal, cost_one_run_unical)

    def calibration_grid_search():
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
        original_model = caldata_obj.model_visibilities[0, :, 0, 0].copy()
        original_data = caldata_obj.data_visibilities[0, :, 0, 0].copy()
        sigma_t_scales = np.arange(0, 10, 7 / 3)
        sigma_m_scales = np.arange(-10, 10, 7 / 3)
        real_avg_g_minus_1_unical_truth = []
        real_avg_g_minus_1_unical_skyapprox = []
        real_avg_g_minus_1_skycal = []
        calc_avg_abs_model_errors = []
        calc_avg_abs_thermal_noise = []
        for sigma_t in sigma_t_scales:
            for sigma_m in sigma_m_scales:
                print(f"\n\n***Testing***\n\t{sigma_t=:.2f} {sigma_m=:.2f}\n\n")
                caldata_obj.gains[:, 0, 0] = 1
                caldata_obj.data_visibilities[0, :, 0, 0] = original_data
                caldata_obj.model_visibilities[0, :, 0, 0] = original_model
                e_real, e_imag, _, _ = sim.simulate_model_error(
                    n_bls=caldata_obj.Nbls,
                    n_times=caldata_obj.Ntimes,
                    sigma_e_0=max(np.abs(sigma_m), 0.1),
                    uv_norm_array=caldata_obj.uv_norm,
                    threshold_length=0,
                    weighting_function="constant_weights",
                    scaling_factor=1,
                    # seed=int(time.time()),
                    seed=1,
                )
                if sigma_m < 0:
                    caldata_obj.model_visibilities[0, :, 0, 0] += e_real + 1j * e_imag
                else:
                    caldata_obj.data_visibilities[0, :, 0, 0] += e_real + 1j * e_imag
                n_real, n_imag = sim.simulate_thermal_noise(
                    sigma_t_0=max(sigma_t, 0.1),
                    n_bls=caldata_obj.Nbls,
                    n_times=caldata_obj.Ntimes,
                    seed=1,
                )
                data_vT = caldata_obj.data_visibilities[0, :, 0, 0].copy()
                caldata_obj.data_visibilities[0, :, 0, 0] += n_real + 1j * n_imag
                gains = calopt.run_skycal_optimization_per_pol_single_freq(
                    caldata_obj=caldata_obj, xtol=1e-5, maxiter=200
                )
                real_avg_g_minus_1_skycal.append(np.mean(gains).real - 1)
                vwa = variable_weights.VariableWeightsArray()
                scaling_factors = [1, 100000]
                for scaling_factor in scaling_factors:
                    vwa.set_algorithm_weights(
                        caldata_obj,
                        sigma_t_0=max(sigma_t, 0.1),
                        sigma_m_0=sigma_m,
                        threshold_length=0,
                        weighting_function="constant_weights",
                        scaling_factor=scaling_factor,
                    )
                    gains, _ = calopt.run_unical_optimization(
                        caldata_obj=caldata_obj, xtol=1e-5, maxiter=200
                    )
                    if scaling_factor == 1:
                        real_avg_g_minus_1_unical_truth.append(np.mean(gains).real - 1)
                        vT_minus_m = (
                            data_vT - caldata_obj.model_visibilities[0, :, 0, 0]
                        )
                        thermal_noise = (
                            caldata_obj.data_visibilities[0, :, 0, 0] - data_vT
                        )
                        avg_abs_model_error = np.mean(np.abs(vT_minus_m))
                        if sigma_m < 0:
                            avg_abs_model_error *= -1
                        calc_avg_abs_model_errors.append(avg_abs_model_error)
                        calc_avg_abs_thermal_noise.append(np.std(thermal_noise.real))
                    else:
                        real_avg_g_minus_1_unical_skyapprox.append(
                            np.mean(gains).real - 1
                        )
        start_time = time.time()
        start_time_suffix = str(start_time)
        git_hash = str(
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("ascii")
            .strip()
        )
        git_time_suffix = f"g{git_hash}_t{start_time_suffix}"
        file_suffix = git_time_suffix
        start_time_dt = datetime.fromtimestamp(start_time, tz=timezone.utc)
        metadata = {
            "Date": f"{start_time_dt:%B %d, %Y}",
            "Time": f"{start_time_dt:%H:%M:%S}",
            "Sigma_t Vals": ",".join(
                f"{noise:.3f}" for noise in sigma_t_scales.tolist()
            ),
            "Sigma_m Vals": ",".join(
                f"{error:.3f}" for error in sigma_m_scales.tolist()
            ),
            "Scaling Factors (Cost)": "None",
            "Scaling Factor (Sim)": "1",
            "Git ID": git_hash,
            "Time ID": start_time_suffix,
            "Weighting Function": "constant_weights",
            "Optimization Function": "None",
        }
        print(
            "Unical gains arrays equal for both scaling factors?"
            f"{
                np.array_equal(
                    real_avg_g_minus_1_unical_truth, real_avg_g_minus_1_unical_skyapprox
                )
            }"
        )
        dev_tools.plot_3d_data_as_2d_hist(
            x_array=np.asarray(calc_avg_abs_model_errors),
            y_array=np.asarray(calc_avg_abs_thermal_noise),
            z_array=np.asarray(real_avg_g_minus_1_unical_truth),
            plot_title="<Re(g)>-1 vs $\\sigma_t$ & $\\sigma_m$ GMM: "
            f"{caldata_obj.gains_multiply_model}"
            "\n(Calculated, Unical - Scalefactor: 1)",
            plot_xlabel="$<|m - v_T|>$",
            plot_ylabel="$<|v - v_T|>$",
            plot_vmax=0.3,
            plot_vmin=-0.3,
            # plot_xlim_h = 1,
            # plot_ylim_h = 1,
            # plot_ylim_l = -1,
            # plot_xlim_l = -1,
            filename=f"{THIS_DIR}/images/grid_search_{file_suffix}_unical_truth.png",
            plot_cmap="PuOr",
            cmap_label="$Re(g)-1$",
            suffix=file_suffix,
            metadata=metadata,
        )
        dev_tools.plot_3d_data_as_2d_hist(
            x_array=np.asarray(calc_avg_abs_model_errors),
            y_array=np.asarray(calc_avg_abs_thermal_noise),
            z_array=np.asarray(real_avg_g_minus_1_unical_skyapprox),
            plot_title="<Re(g)>-1 vs $\\sigma_t$ & $\\sigma_m$ GMM:"
            f"{caldata_obj.gains_multiply_model}"
            "\n(Calculated, Unical - Scalefactor: 100000)",
            plot_xlabel="$<|m - v_T|>$",
            plot_ylabel="$<|v - v_T|>$",
            plot_vmax=0.3,
            plot_vmin=-0.3,
            # plot_xlim_h = 1,
            # plot_ylim_h = 1,
            # plot_ylim_l = -1,
            # plot_xlim_l = -1,
            filename=f"{THIS_DIR}/images/grid_search_{file_suffix}_unical_skyapprox.png",
            plot_cmap="PuOr",
            cmap_label="$Re(g)-1$",
            suffix=file_suffix,
            metadata=metadata,
        )
        dev_tools.plot_3d_data_as_2d_hist(
            x_array=np.asarray(calc_avg_abs_model_errors),
            y_array=np.asarray(calc_avg_abs_thermal_noise),
            z_array=np.asarray(real_avg_g_minus_1_skycal),
            plot_title="<Re(g)>-1 vs $\\sigma_t$ & $\\sigma_m$ GMM:"
            f"{caldata_obj.gains_multiply_model}\n(Calculated, Skycal)",
            plot_xlabel="$<|m - v_T|>$",
            plot_ylabel="$<|v - v_T|>$",
            plot_vmax=0.3,
            plot_vmin=-0.3,
            # plot_xlim_h = 1,
            # plot_ylim_h = 1,
            # plot_ylim_l = -1,
            # plot_xlim_l = -1,
            filename=f"{THIS_DIR}/images/grid_search_{file_suffix}_skycal.png",
            plot_cmap="PuOr",
            cmap_label="$Re(g)-1$",
            suffix=file_suffix,
            metadata=metadata,
        )

    def plot_skycal_unical_diff_per_scaling_factor():
        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/tutorial_full_onetime_unflagged.uvfits")
        data = model.copy()
        caldata_obj = caldata.CalData()
        caldata_obj.load_data(data, model)
        n_real, n_imag = sim.simulate_thermal_noise(
            sigma_t_0=3,
            n_bls=caldata_obj.Nbls,
            n_times=caldata_obj.Ntimes,
            seed=int(time.time()),
        )
        e_real, e_imag, _, _ = sim.simulate_model_error(
            n_bls=caldata_obj.Nbls,
            n_times=caldata_obj.Ntimes,
            sigma_e_0=8,
            uv_norm_array=caldata_obj.uv_norm,
            threshold_length=0,
            weighting_function="constant_weights",
            scaling_factor=1,
            seed=int(time.time()),
        )
        caldata_obj.data_visibilities[0, :, 0, 0] += n_real + 1j * n_imag
        caldata_obj.model_visibilities[0, :, 0, 0] += e_real + 1j * e_imag
        # caldata_obj.data_visibilities[0,:,0,0] += e_real + 1j*e_imag
        caldata_obj.gains_multiply_model = True
        gains_skycal = calopt.run_skycal_optimization_per_pol_single_freq(
            caldata_obj=caldata_obj, xtol=1e-5, maxiter=200
        )
        # avg_mag_skycal = np.mean(np.abs(gains_skycal))
        avg_real_skycal = np.mean(gains_skycal).real
        gains_diff = []
        # iterate through algorithm weights for unical
        vwa = variable_weights.VariableWeightsArray()
        scaling_factors_lo = [scaling_factor for scaling_factor in range(1, 100, 5)]
        scaling_factors_hi = [
            scaling_factor for scaling_factor in range(100, 100000, 19980)
        ]
        scaling_factors = scaling_factors_lo + scaling_factors_hi
        sigma_m = 1
        for scaling_factor in scaling_factors:
            vwa.set_algorithm_weights(
                caldata_obj,
                sigma_t_0=1,
                sigma_m_0=sigma_m,
                threshold_length=0,
                weighting_function="constant_weights",
                scaling_factor=scaling_factor,
            )
            gains_unical, _ = calopt.run_unical_optimization(
                caldata_obj=caldata_obj, xtol=1e-5, maxiter=200
            )
            print(f"***Avg Mag Gains - Unical: {np.mean(gains_unical).real:.3f}***")
            gains_diff.append(np.mean(gains_unical.real) - avg_real_skycal)
        print(f"{caldata_obj.gains_multiply_model=}")
        plt.plot(scaling_factors, gains_diff)
        plt.title(
            "Unical - Skycal Avg Mag Gains vs Scaling Factor "
            f"(~$1/\\sigma_m^2$)\n$\\sigma_m = {sigma_m}$, $m < v_T$"
        )
        plt.xlabel("Scaling Factor (~$1/\\sigma_m^2$)")
        plt.ylabel("$Re<g_u> - Re<g_s>$")
        plt.show()

    def compare_optimizers(self, sigma_m=0.1, sigma_t=0.1, maxiter=20):
        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/tutorial_full_onetime_unflagged.uvfits")
        data = model.copy()
        caldata_obj = caldata.CalData()
        caldata_obj.load_data(data, model)
        org_gains = copy.deepcopy(caldata_obj.gains[:, 0, 0])
        org_fit_vis = copy.deepcopy(caldata_obj.fit_vis[0, :, 0, 0])
        n_real, n_imag = sim.simulate_thermal_noise(
            sigma_t_0=3,
            n_bls=caldata_obj.Nbls,
            n_times=caldata_obj.Ntimes,
            seed=int(time.time()),
        )
        e_real, e_imag, _, _ = sim.simulate_model_error(
            n_bls=caldata_obj.Nbls,
            n_times=caldata_obj.Ntimes,
            sigma_e_0=3,
            uv_norm_array=caldata_obj.uv_norm,
            threshold_length=0,
            weighting_function="constant_weights",
            scaling_factor=1,
            seed=int(time.time()),
        )
        caldata_obj.data_visibilities[:, :, 0, 0] += n_real + 1j * n_imag
        # caldata_obj.model_visibilities[:,:,0,0] += e_real + 1j*e_imag  # m > v_T
        caldata_obj.data_visibilities[:, :, 0, 0] += e_real + 1j * e_imag  # m < v_T
        caldata_obj.gains_multiply_model = True
        vwa = variable_weights.VariableWeightsArray()
        vwa.set_algorithm_weights(
            caldata_obj,
            sigma_t_0=0.45,
            sigma_m_0=0.6,
            threshold_length=0,
            weighting_function="constant_weights",
            scaling_factor=1e4,
        )
        optimizers = ["pytorch", "powell"]
        opt_gains = []
        for optimizer in optimizers:
            caldata_obj.gains[:, 0, 0] = org_gains
            caldata_obj.fit_vis[0, :, 0, 0] = org_fit_vis
            if optimizer == "powell":
                this_maxiter = 200
            else:
                this_maxiter = maxiter
            gains, _ = calopt.run_unical_optimization(
                caldata_obj=caldata_obj,
                xtol=1e-5,
                maxiter=this_maxiter,  # is this actually showing up in PyTorch?
                optimization_scheme=optimizer,
            )
            print(f"{optimizer=}\n\n{gains=}")
            opt_gains.append(gains)
        lbfgs_gains = opt_gains[0]
        powell_gains = opt_gains[
            1
        ]  # make sure these track the correct optimizer order...
        plot_time = int(time.time())
        abs_powell_minus_lbfgs = np.abs(powell_gains) - np.abs(lbfgs_gains)
        print("Plotting abs diff plot")
        x_arr = [x for x in range(abs_powell_minus_lbfgs.size)]
        plt.plot(x_arr, abs_powell_minus_lbfgs)
        plt.title("$|g_P| - |g_L|$ per antenna")
        plt.ylabel("$|g_P| - |g_L|$")
        plt.xlabel("Antennas")
        plt.savefig(
            f"calico/images/powell_lbfgs_diff_abs_{plot_time}_maxiter{maxiter}.png"
        )
        plt.close()
        real_powell_minus_lbfgs = powell_gains.real - lbfgs_gains.real
        print("Plotting real diff plot")
        x_arr = [x for x in range(real_powell_minus_lbfgs.size)]
        plt.plot(x_arr, real_powell_minus_lbfgs)
        plt.title("$Re(g_P) - Re(g_L)$ per antenna")
        plt.ylabel("$Re(g_P) - Re(g_L)$")
        plt.xlabel("Antennas")
        plt.savefig(f"calico/images/powell_lbfgs_diff_real_{plot_time}.png")
        plt.close()
        imag_powell_minus_lbfgs = powell_gains.imag - lbfgs_gains.imag
        print("Plotting imag diff plot")
        x_arr = [x for x in range(imag_powell_minus_lbfgs.size)]
        plt.plot(x_arr, imag_powell_minus_lbfgs)
        plt.title("$Im(g_P) - Im(g_L)$ per antenna")
        plt.ylabel("$Im(g_P) - Im(g_L)$")
        plt.xlabel("Antennas")
        plt.savefig(f"calico/images/powell_lbfgs_diff_imag_{plot_time}.png")
        plt.close()
        print("Plotting scatter plot in nsew-plane (abs diff)")
        en_plane = caldata_obj.antenna_positions[:, :-1]
        print(f"en-plane shape - {en_plane.shape}")
        max_diff = np.max(
            [np.max(abs_powell_minus_lbfgs), np.abs(np.min(abs_powell_minus_lbfgs))]
        )
        scatter_plot = plt.scatter(
            x=en_plane[:, 0],
            y=en_plane[:, 1],
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
        return abs_powell_minus_lbfgs

    def elbow_plot(self):
        diffs = []
        maxiter_arr = []
        for i in range(0, 100, 10):
            diffs.append(
                self.compare_optimizers(
                    TestStringMethods, maxiter=i, sigma_m=1, sigma_t=0.5
                ).mean()
            )
            maxiter_arr.append(i)
        plt.plot(maxiter_arr, diffs)
        plt.title("$|g_P| - |g_L|$ as function of max iterations")
        plt.ylabel("$|g_P| - |g_L|$")
        plt.xlabel("Maxiter")
        plt.savefig("calico/images/powell_lbfgs_diff_elbow_plot.png")

    def plot_aggregate_montecarlos():
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
        plt.xlim(0.99, 1.05)
        plt.savefig("calico/images/hist_ne5means")
        plt.close()

        plt.hist(ne10_arr, bins=7, color="orange")
        plt.title("Averages of Monte Carlos with 1e7 samples (10 Jy error/noise)")
        plt.xlim(0.95, 1.3)
        plt.savefig("calico/images/hist_ne10means")
        plt.close()

        plt.hist(ne20_arr, bins=10, color="green")
        plt.title("Averages of Monte Carlos with 1e7 samples (20 Jy error/noise)")
        plt.xlim(0.9, 3.5)
        plt.savefig("calico/images/hist_ne20means")
        plt.close()

    def test_pack_reshape_multiple_times(self):
        model = pyuvdata.UVData()
        model.read_uvfits("./calico/data/tutorial_medium.uvfits")
        data = model.copy()
        caldata_obj = caldata.CalData()
        caldata_obj.load_data(
            data,
            model,
            gains_multiply_model=True,
            weighting_function="constant_weights",
            sigma_t_0=1,
            sigma_m_0=1,
            scaling_factor_cost=1,
            threshold_length=0,
            lambda_val=100,
            simulate_visibilities=True,
        )
        caldata_obj.set_ant_inds(0, 0)
        caldata_obj.set_bl_inds(0, 0)
        known = caldata_obj.model_visibilities[:, :, 0, 0]
        packed = caldata_obj.pack(0, 0, unical=True)
        u_part = packed[2 * len(caldata_obj.ant_inds) :]

        # how the cost wrapper reads it
        try:
            fit_vis = np.reshape(
                u_part, (caldata_obj.Ntimes * len(caldata_obj.bl_inds), 2)
            )
            fit_vis = fit_vis[:, 0] + 1j * fit_vis[:, 1]
            print(f"\nCost-wrapper unpack OK, shape {fit_vis.shape}")
        except ValueError as e:
            print(f"\nCost-wrapper unpack FAILS\n  {type(e).__name__}: {e}\n")

        # how the result reshape reads it; should equal `known`
        fit_vis2 = np.reshape(u_part, (caldata_obj.Ntimes, len(caldata_obj.bl_inds), 2))
        fit_vis2 = fit_vis2[:, :, 0] + 1j * fit_vis2[:, :, 1]
        print(
            f"Pack\n  result-reshape consistent? "
            f"{np.allclose(fit_vis2, known[:, caldata_obj.bl_inds])}\n"
        )

    def test_calibration_completes_multiple_times(self):
        data = pyuvdata.UVData()
        data.read_uvfits("./calico/data/tutorial_medium.uvfits")
        model = data.copy()
        caldata_obj = caldata.CalData()
        caldata_obj.load_data(
            data,
            model,
            gains_multiply_model=True,
            weighting_function="constant_weights",
            sigma_t_0=1,
            sigma_m_0=1,
            scaling_factor_cost=1,
            threshold_length=0,
            lambda_val=100,
            simulate_visibilities=True,
        )
        caldata_obj.set_ant_inds(0, 0)
        caldata_obj.set_bl_inds(0, 0)
        optimizers = ["powell", "pytorch"]
        for optimizer in optimizers:
            caldata_obj.unified_calibration(verbose=True, optimization_scheme=optimizer)
            print(f"\nCalibration for {optimizer} SUCCEEDS")

    def test_basic_stability(self, optimizer):
        model = pyuvdata.UVData()
        model.read_uvfits("./calico/data/tutorial_medium.uvfits")
        data = model.copy()
        caldata_obj = caldata.CalData()
        caldata_obj.load_data(
            data,
            model,
            gains_multiply_model=True,
            weighting_function="constant_weights",
            sigma_t_0=1,
            sigma_m_0=1,
            scaling_factor_cost=1,
            threshold_length=0,
            lambda_val=100,
        )
        starting_gains = caldata_obj.gains.copy()
        ending_gains, _ = calopt.run_unical_optimization(
            caldata_obj=caldata_obj,
            xtol=1e-5,
            maxiter=200,
            optimization_scheme=optimizer,
        )
        dev_tools.complex_trajectory_plot(
            starting_complex_point=starting_gains,
            complex_step=ending_gains,
            n_trajectories=caldata_obj.Nants,
            filename_prefix=f"basic_stability_test_{optimizer}",
            title=f"Stability Test\nOptimizer = {optimizer}",
            xlabel="Real",
            ylabel="Imag",
            xlims=(0.9, 1.1),
            ylims=(-0.1, 0.1),
        )

    def compare_unical_skycal_gains_and_us(self):
        seed = 421
        same_sky_all_times = True
        scaling_factor_skycal = 0.001
        scaling_factor_unical = 1
        sigma_m = 10
        sigma_t = 0.1
        filename = "tutorial_medium"
        # filename = "fhd_data_one_freq_015"
        gaussian_simulation = True
        caldata_obj = caldata.CalData()
        skycal_model_error = dev_tools.prepare_standard_unical_test_run(
            filename=filename,
            caldata_obj=caldata_obj,
            gaussian_simulation=gaussian_simulation,
            seed=seed,
            sigma_m=sigma_m,
            sigma_t=sigma_t,
            scaling_factor=scaling_factor_skycal,
            same_sky_all_times=same_sky_all_times,
            return_model_error=True,
        )
        orig_data = caldata_obj.data_visibilities.copy()
        orig_model = caldata_obj.model_visibilities.copy()
        caldata_obj.unified_calibration(
            verbose=True, xtol=1e-5, maxiter=200, optimization_scheme="pytorch"
        )
        skycal_gains = caldata_obj.gains[..., 0, 0]
        # flatten data arrays from (Ntimes, Nbls) to (Nblts,)
        skycal_u = np.ravel(caldata_obj.fit_vis[..., 0, 0])
        skycal_m = np.ravel(caldata_obj.model_visibilities[..., 0, 0])
        skycal_e = np.ravel(skycal_model_error[..., 0])

        unical_model_error = dev_tools.prepare_standard_unical_test_run(
            filename=filename,
            caldata_obj=caldata_obj,
            gaussian_simulation=gaussian_simulation,
            seed=seed,
            sigma_m=sigma_m,
            sigma_t=sigma_t,
            scaling_factor=scaling_factor_unical,
            same_sky_all_times=same_sky_all_times,
            return_model_error=True,
        )
        caldata_obj.data_visibilities = orig_data
        caldata_obj.model_visibilities = orig_model
        caldata_obj.unified_calibration(
            verbose=True, xtol=1e-5, maxiter=200, optimization_scheme="pytorch"
        )
        unical_gains = caldata_obj.gains[..., 0, 0]
        # flatten data arrays from (Ntimes, Nbls) to (Nblts,)
        unical_u = np.ravel(caldata_obj.fit_vis[..., 0, 0])
        unical_m = np.ravel(caldata_obj.model_visibilities[..., 0, 0])
        unical_e = np.ravel(unical_model_error[..., 0])

        # plotting quantities
        ants = np.arange(unical_gains.size)
        skycal_vT_u_diff = np.abs(skycal_m + skycal_e - skycal_u)
        unical_vT_u_diff = np.abs(unical_m + unical_e - unical_u)
        skycal_e = np.abs(skycal_e)
        unical_e = np.abs(unical_e)

        # plotting gains over ants
        plt.scatter(
            ants, skycal_gains.real, color="tab:blue", s=25, zorder=3, label="skycal"
        )
        plt.scatter(
            ants, unical_gains.real, color="darkorange", s=25, zorder=3, label="unical"
        )
        plt.title(
            f"\nunical $<Re(g)>$ {np.mean(unical_gains.real):.3f}"
            f" - $Std(Re(g))$ {np.std(unical_gains.real):.3f}"
            f"\nskycal $<Re(g)>$ {np.mean(skycal_gains.real):.3f}"
            f" - $Std(Re(g))$ {np.std(skycal_gains.real):.3f}"
        )
        plt.xlabel("Ant #")
        plt.ylabel("$Re(g)$")
        # plt.ylim(0.7, 1.7)
        plt.legend()
        plt.savefig("calico/images/gains_avg_unical_vs_skycal.png")
        plt.close()
        # plotting u's hist for skycal
        plt.hist(skycal_vT_u_diff, bins=50, color="red", label="$|vT-u|$")
        plt.hist(skycal_e, bins=50, color="blue", alpha=0.5, label="$|e|$")
        plt.xlabel("Jy")
        plt.title(
            "Fit model parameter $u$s convergence (skycal)"
            f"\n$\\sigma_m=${sigma_m:.2f} $\\sigma_t=${sigma_t:.2f}"
        )
        plt.xlim(0, 0.35)
        plt.legend()
        plt.savefig("calico/images/fit_us_convergence_skycal.png")
        plt.close()

        # plotting u's hist for unical
        plt.hist(unical_vT_u_diff, bins=50, color="red", label="$|vT-u|$")
        plt.hist(unical_e, bins=50, color="blue", alpha=0.5, label="$|e|$")
        # plt.hist(
        #     unical_n,
        #     bins=50,
        #     color="pink",
        #     label="$|n|$",
        # )
        plt.xlabel("Jy")
        plt.title(
            "Fit model parameter $u$s convergence (unical)"
            f"\n$\\sigma_m=${sigma_m:.2f} $\\sigma_t=${sigma_t:.2f}"
        )
        plt.xlim(0, 0.35)
        plt.legend()
        plt.savefig("calico/images/fit_us_convergence_unical.png")
        plt.close()

    def examine_gains_fit_time_by_time(self):
        seed = 421
        same_sky_all_times = True
        scaling_factor_skycal = 0.0001
        # scaling_factor_unical = 1
        sigma_m = 0.1
        sigma_t = 5
        caldata_obj = caldata.CalData()
        filename = "tutorial_medium"
        # filename = "fhd_data_one_freq_015"
        gaussian_simulation = False
        dev_tools.prepare_standard_unical_test_run(
            filename=filename,
            caldata_obj=caldata_obj,
            gaussian_simulation=gaussian_simulation,
            seed=seed,
            sigma_m=sigma_m,
            sigma_t=sigma_t,
            scaling_factor=scaling_factor_skycal,
            same_sky_all_times=same_sky_all_times,
        )
        real_gains_arr = []
        data_copy = caldata_obj.data_visibilities.copy()
        model_copy = caldata_obj.model_visibilities.copy()
        fit_vis_copy = caldata_obj.fit_vis.copy()
        model_weights_copy = caldata_obj.model_weights.copy()
        vis_weights_copy = caldata_obj.visibility_weights.copy()
        for time_ind in range(caldata_obj.Ntimes):
            print(f"\n\n***time {time_ind}***\n\n")
            caldata_obj.data_visibilities = data_copy[time_ind : time_ind + 1, ...]
            caldata_obj.model_visibilities = model_copy[time_ind : time_ind + 1, ...]
            caldata_obj.model_weights = model_weights_copy[time_ind : time_ind + 1, ...]
            caldata_obj.visibility_weights = vis_weights_copy[
                time_ind : time_ind + 1, ...
            ]
            caldata_obj.gains = np.ones((caldata_obj.Nants, 1, 1))
            caldata_obj.fit_vis = model_copy[time_ind : time_ind + 1, ...]
            caldata_obj.unified_calibration(
                verbose=True, xtol=1e-5, maxiter=200, optimization_scheme="pytorch"
            )
            real_gains_arr.append(caldata_obj.gains[..., 0, 0].real)
        real_gains_arr = np.asarray(real_gains_arr)
        avg_gains = np.mean(np.asarray(real_gains_arr), axis=0)
        ants = np.arange(real_gains_arr.shape[1])
        x_cloud = np.broadcast_to(ants, real_gains_arr.shape).ravel()
        y_cloud = real_gains_arr.ravel()
        caldata_obj.data_visibilities = data_copy
        caldata_obj.model_visibilities = model_copy
        caldata_obj.fit_vis = fit_vis_copy
        caldata_obj.model_weights = model_weights_copy
        caldata_obj.visibility_weights = vis_weights_copy
        caldata_obj.gains = np.ones((caldata_obj.Nants, 1, 1))
        caldata_obj.fit_vis = model_copy
        caldata_obj.unified_calibration(
            verbose=True, xtol=1e-5, maxiter=200, optimization_scheme="pytorch"
        )
        gains = caldata_obj.gains[..., 0, 0]
        plt.scatter(
            x_cloud,
            y_cloud,
            color="orange",
            alpha=0.15,
            s=12,
            edgecolors="none",
            zorder=1,
            label="Individual Times",
        )
        plt.scatter(
            ants, gains.real, color="tab:blue", s=25, zorder=3, label="Full 56 Times"
        )
        plt.scatter(
            ants, avg_gains, color="darkorange", s=25, zorder=3, label="Avg Over Times"
        )
        plt.title(
            "Compare time avg of Re(g) vs fit over all times"
            f"\nAnt avg - Individuals: {np.mean(avg_gains):.5f}"
            f"\nAll times: {np.mean(gains.real):.5f}"
        )
        plt.xlabel("Ant #")
        plt.ylabel("$Re(g)$")
        plt.ylim(0.7, 1.7)
        plt.legend()
        plt.savefig("calico/images/gains_avg_over_time_comp.png")
        plt.close()
        print(
            f"\nAny negatives? {(gains < 0).any()}"
            f"\nHow many?      {gains[gains < 0].size}"
            f"\nWhat are they? {gains[gains < 0]}\n"
        )

    def gain_offset_with_more_times(self):
        times = [t for t in range(1, 57)]
        data_files = [f"data_{t}" for t in range(1, 56)]
        data_files.append("tutorial_medium")
        gain_offsets = []
        seed = 100
        same_sky_all_times = True
        # scaling_factor_skycal = 0.0001
        scaling_factor_unical = 1
        sigma_m = 1.5
        sigma_t = 5
        for file in data_files:
            caldata_obj = caldata.CalData()
            dev_tools.prepare_standard_unical_test_run(
                filename=file,
                caldata_obj=caldata_obj,
                seed=seed,
                sigma_m=sigma_m,
                sigma_t=sigma_t,
                scaling_factor=scaling_factor_unical,
                same_sky_all_times=same_sky_all_times,
            )
            caldata_obj.unified_calibration(
                verbose=True, xtol=1e-5, maxiter=200, optimization_scheme="pytorch"
            )
            gain_offsets.append(np.mean(caldata_obj.gains[..., 0, 0].real, axis=0) - 1)
        plt.scatter(times, gain_offsets)
        plt.title("Gain offsets over time")
        plt.xlabel("Ntimes")
        plt.ylabel("Gain offset")
        plt.savefig("calico/images/gain_offset_with_more_times.png")
        plt.close()


if __name__ == "__main__":
    # TestStringMethods.examine_gains_fit_time_by_time(TestStringMethods)
    # TestStringMethods.gain_offset_with_more_times(TestStringMethods)
    TestStringMethods.compare_unical_skycal_gains_and_us(TestStringMethods)
    # TestStringMethods.test_basic_stability(TestStringMethods, "bfgs")
    # TestStringMethods.test_basic_stability(TestStringMethods, "newton-cg")
