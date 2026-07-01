import subprocess
import time
import pickle
import json
import os
import copy

from scipy.differentiate import jacobian
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, PngImagePlugin
from matplotlib.backends.backend_pdf import PdfPages

from calico import cost_function_calculations, calibration_optimization as cal_opt, calibration_wrappers as calwrap, caldata
from pyuvdata import UVData, UVCal, Telescope
import noise_and_error_simulation as sim
import variable_weights

class DevTools:

    def __init__(self):
        params_init_flattened=None
        caldata_obj=None
        Nants_unflagged=None
        freq_ind=None
        vis_pol_ind=None

    def format_var_name(self, 
                        input_string: str
    ) -> str:
        output_string = input_string.replace("_", " ")
        return output_string.title()

    def compare_analytic_and_numeric_jacobians(self) -> None:
        jac_numeric_result = jacobian(self.cost_vectorized, 
                                      self.params_init_flattened).df
        jac_analytic_result = cal_opt.jacobian_unical_wrapper(
            self.params_init_flattened,
            self.caldata_obj,
            self.caldata_obj.ant_inds,
            self.Nants_unflagged,
            self.caldata_obj.bl_inds + self.Nants_unflagged,
            self.freq_ind,
            self.vis_pol_ind,
        )
        self.display_where_large_real_or_imag(
            jac_analytic_result,
            jac_numeric_result,
        )
        jac, jac_numeric = self.assemble_jac_into_complex_array(
            jac_analytic_result,
            jac_numeric_result,
        )
        jac_error, jac_frac, where_large = self.calc_error_vals(jac_numeric, jac)
        n_vals = len(where_large[0])
        # find large errors
        analytic_vals = jac[where_large]
        numeric_vals = jac_numeric[where_large]
        # reshape u params and get m-u
        fit_vis_flat = np.reshape(self.params_init_flattened[2*self.Nants_unflagged:], (self.caldata_obj.Nbls, 2))
        fit_vis_reshaped = fit_vis_flat[:,0] + 1.0j * fit_vis_flat[:,1]
        u_m_diff = fit_vis_reshaped - self.caldata_obj.model_vis_reshaped[0,:]
        # jac error plot
        self.complex_trajectory_plot(
            analytic_vals,
            numeric_vals,
            n_vals,
            "Jacobian Error Plot (from analytics to numeric)",
            (-1,1),
            (-1,1),
            "jac_error_",
        )
        # analytic step direction
        analytic_step = u_m_diff - jac[self.Nants_unflagged:]
        self.complex_trajectory_plot(
            u_m_diff,
            analytic_step,
            n_vals,
            "Step direction for analytic gradients",
            (-3,3),
            (-2,2),
            "step_dir_analytic_",
        )
        # numeric step direction
        numeric_step = u_m_diff - jac_numeric[self.Nants_unflagged:]
        self.complex_trajectory_plot(
            u_m_diff,
            numeric_step,
            n_vals,
            "Step direction for numeric gradients",
            (-3,3),
            (-2,2),
            "step_dir_numeric_",
        )
    
    def display_where_large_real_or_imag(
        self,
        jac_analytic: np.ndarray[float],
        jac_numeric: np.ndarray[float],
        verbose : bool = False,
    ) -> None:
        jac_error, jac_frac, where_large = self.calc_error_vals(jac_numeric, jac_analytic)
        n_vals = len(where_large[0])
        # find large errors
        analytic_vals = jac_analytic[where_large]
        numeric_vals = jac_numeric[where_large]
        param_vals = self.params_init_flattened[where_large]
        # print display
        np.set_printoptions(precision=4)
        if verbose:
           print("***WHERE ERROR IS LARGE***")
        part = lambda x : "Real" if x == 0 else "Imag"
        if verbose:
            for val in range(n_vals):
                print("Value",val+1)
                print(f"aj: {analytic_vals[val]} nj: {numeric_vals[val]} val: {param_vals[val]} bl_ind: {part((where_large[0][val] - 2*self.Nants_unflagged) % 2)}")

    def calc_error_vals(
        self,
        numeric_jac : np.ndarray,
        analytic_jac : np.ndarray,
    ) -> tuple[float, float, np.ndarray[int]]:
        jac_error = (np.abs(analytic_jac - numeric_jac)
                     / (np.abs(analytic_jac) + np.abs(numeric_jac)))
        jac_frac = (2 * np.abs(numeric_jac)
                     / (np.abs(analytic_jac) + np.abs(numeric_jac)))
        where_large = np.nonzero(jac_error > 0.9)
        return jac_error, jac_frac, where_large

    # take array of real and imaginary parts and convert into
    # normalized complex array
    def assemble_jac_into_complex_array(
        self,
        analytic_jac_result : np.ndarray[float], 
        numeric_jac_result : np.ndarray[float],
    ) -> tuple[np.ndarray[complex], np.ndarray[complex]]:
        jac = np.zeros(shape=(self.Nants_unflagged + self.caldata_obj.Nbls), dtype=complex)
        jac_numeric = np.zeros(shape=(self.Nants_unflagged + self.caldata_obj.Nbls), dtype=complex)
        # even indices are real, odd indices are imaginary
        jac.real += analytic_jac_result[0::2]
        jac.imag += analytic_jac_result[1::2]
        jac_numeric.real += numeric_jac_result[0::2]
        jac_numeric.imag += numeric_jac_result[1::2]
        # normalize
        jac /= np.abs(jac)
        jac_numeric /= np.abs(jac_numeric)
        return jac, jac_numeric

    def get_starting_cost_func_val(self) -> None:
        print("***STARTING FUNCTION VALUE***", 
        cal_opt.cost_unical_wrapper(self.params_init_flattened,
                                    self.caldata_obj,
                                    self.caldata_obj.ant_inds,
                                    self.Nants_unflagged,
                                    self.caldata_obj.bl_inds + self.Nants_unflagged,
                                    self.freq_ind,
                                    self.vis_pol_ind,))
        
    def cost_for_numeric_jac(self, 
                             params_array : np.ndarray[float]
    ) -> float:
        self.Nants_unflagged = len(self.caldata_obj.ant_inds)
        # reshape gain params
        gains_reshaped = np.reshape(params_array[:2*self.Nants_unflagged], (self.Nants_unflagged, 2))
        gains_reshaped = gains_reshaped[:, 0] + 1.0j * gains_reshaped[:, 1]
        gains_array = np.ones((self.Nants_unflagged), dtype=complex)
        gains_array[self.caldata_obj.ant_inds] = gains_reshaped
        # reshape u params
        fit_vis_flat = np.reshape(params_array[2*self.Nants_unflagged:], (len(self.caldata_obj.bl_inds), 2))
        fit_vis_reshaped = fit_vis_flat[:,0] + 1.0j * fit_vis_flat[:,1]
        gains_expanded = (gains_array[self.caldata_obj.ant1_inds]
                           * np.conj(gains_array[self.caldata_obj.ant2_inds]))[np.newaxis, :]
        res_vec_1 = self.caldata_obj.data_vis_reshaped - gains_expanded * fit_vis_reshaped
        res_vec_2 = fit_vis_reshaped - self.caldata_obj.model_vis_reshaped
        cost = np.sum(self.caldata_obj.vis_weights_reshaped * np.abs(res_vec_1) ** 2
                       + self.caldata_obj.model_weights_reshaped * np.abs(res_vec_2)**2)
        return cost
    
    def cost_vectorized(self, 
                        params_array : np.ndarray[float]
    ) -> np.ndarray:
        return np.apply_along_axis(self.cost_for_numeric_jac, axis=0, arr=params_array)
    
    def complex_trajectory_plot(
        self,
        starting_complex_point : list[complex] | np.ndarray[complex],
        complex_step : list[complex] | np.ndarray[complex],
        n_trajectories : int,
        filename_prefix : str,
        xlims : tuple[int | float] = None,
        ylims : tuple[int | float] = None,
        title : str = "",
        xlabel : str = "",
        ylabel : str = "",
    ) -> None:
        fig, ax = plt.subplots()
        for val in range(n_trajectories):
            ax.annotate(
                "",
                xytext=(
                        starting_complex_point[val].real,
                        starting_complex_point[val].imag,
                ),
                xy=(
                    complex_step[val].real,
                    complex_step[val].imag,
                ),
                arrowprops=dict(arrowstyle="->"),
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if xlims is not None:
            ax.set_xlim(xlims[0],xlims[1])
        if ylims is not None:
            ax.set_ylim(ylims[0],ylims[1])
        fig.savefig('images/' + filename_prefix
                 + subprocess.check_output(['git','rev-parse','--short','HEAD']).decode('ascii').strip()
                 + '.png',
                 bbox_inches=0,)
        plt.close()

    def plot_change_in_gain_and_model_params(
        self, 
        gains_array : np.ndarray, 
        models_array : np.ndarray,
        type : str = "trajectory",
        glim : tuple[int | float, int | float] = (-0.013, 0.013),
        ulim : tuple[int | float, int | float] = (-1.5,1.5),
        error_type : str = "thermal",
        stddev_thermal : str = "1",
        stddev_model : str = "1",
        verbose : bool = False,
    ) -> None:
        if verbose:
            print("***FIT TESTS***")
            print("\tGains Error, Min -", np.min(np.abs(self.caldata_obj.gains - 1)))
            print("\tGains Error, Max -", np.max(np.abs(self.caldata_obj.gains - 1)))
            print("\t|u-m|, Min -", np.min(np.abs(self.caldata_obj.fit_vis - self.caldata_obj.model_visibilities)))
            print("\t|u-m|, Max -", np.max(np.abs(self.caldata_obj.fit_vis - self.caldata_obj.model_visibilities)))
        if type == "trajectory" or type == "both":
            # plot gains parameters trajectory
            self.complex_trajectory_plot(
                gains_array,
                self.caldata_obj.gains,
                len(self.caldata_obj.ant_inds),
                "Gains Trajectory Plot",
                (0.75, 1.25),
                (-0.25, 0.25),
                # (-2,2),
                # (-2,2),
                "change_gains_",
            )
            # plot fitted visibility parameters trajectory
            u_minus_m_before = models_array[:1,:,0] - self.caldata_obj.model_visibilities[:1,:,0]
            u_minus_m_after = self.caldata_obj.fit_vis[:1,:,0] - self.caldata_obj.model_visibilities[:1,:,0]
            self.complex_trajectory_plot(
                u_minus_m_before[0],
                u_minus_m_after[0],
                len(self.caldata_obj.bl_inds),
                "|u-m| Trajectory Plot",
                (-5,5),
                (-5,5),
                "change_fit-vis_",
            )
        if type == "scatter":
            # gains
            plt.scatter(gains_array.real - 1, gains_array.imag)
            plt.xlim(-1,1)
            plt.ylim(-1,1)
            plt.title("Final Gains")
            plt.xlabel("Real - 1")
            plt.ylabel("Imag")
            plt.savefig('images/' + "final_gains_"
                 + subprocess.check_output(['git','rev-parse','--short','HEAD']).decode('ascii').strip()
                 + '.png',
                 bbox_inches=0,)
            plt.close()
            # models
            u_minus_m_after = self.caldata_obj.fit_vis[:1,:,0] - self.caldata_obj.model_visibilities[:1,:,0]
            plt.scatter(u_minus_m_after.real, u_minus_m_after.imag)
            # plt.xlim(-15,15)
            # plt.ylim(-15,15)
            plt.title("Final u-m")
            plt.xlabel("Real")
            plt.ylabel("Imag")
            plt.savefig('images/' + "final_models_"
                 + subprocess.check_output(['git','rev-parse','--short','HEAD']).decode('ascii').strip()
                 + '.png',
                 bbox_inches=0,)
            plt.close()

        # add in 2D probability density later, starting with scatter plot
        if type == "histogram" or type == "both":
            print("***HISTOGRAM***")
            gains_hist, gains_imag, gains_real = np.histogram2d(gains_array.real, gains_array.imag, bins=50, density=True)
            models_hist, models_imag, models_real = np.histogram2d(models_array.real, models_array.imag, bins=50, density=True)
            fig1, ax1 = plt.subplots()
            # gains plot
            hh1 = ax1.pcolormesh(gains_real, gains_imag, gains_hist, cmap="inferno")
            ax1.add_patch(plt.Circle((0,0), radius=np.std(gains_array), fill=False, color="white"))
            if glim:
                print("***G LIM***")
                ax1.set_xlim(glim[0], glim[1])
                ax1.set_ylim(glim[0], glim[1])
            ax1.set_title(f"Final Gains (Error: {error_type}, sigma_T = {stddev_thermal}, sigma_M = {stddev_model})")
            ax1.set_xlabel("Real - 1")
            ax1.set_ylabel("Imag")
            fig1.colorbar(hh1, ax=ax1)
            plt.savefig('images/' + "final_gains_"
                 + subprocess.check_output(['git','rev-parse','--short','HEAD']).decode('ascii').strip()
                 + '.png',
                 bbox_inches=0,)
            plt.close()
            # models plot
            fig2, ax2 = plt.subplots()
            hh2 = ax2.pcolormesh(models_real, models_imag, models_hist, cmap="inferno")
            ax2.add_patch(plt.Circle((0,0), radius=np.std(models_array), fill=False, color="white"))
            if ulim:
                ax2.set_xlim(ulim[0], ulim[1])
                ax2.set_ylim(ulim[0], ulim[1])
            ax2.set_title(f"Final u-m (Error: {error_type}, sigma_T = {stddev_thermal}, sigma_M = {stddev_model})")
            ax2.set_xlabel("Real")
            ax2.set_ylabel("Imag")
            fig2.colorbar(hh2, ax=ax2)
            plt.savefig('images/' + "final_fit-vis_"
                 + subprocess.check_output(['git','rev-parse','--short','HEAD']).decode('ascii').strip()
                 + '.png',
                 bbox_inches=0,)
            plt.close()

    def old_calculate_many_realizations(self, 
                                    data_file : str,
                                    run_params_filename : str,
                                    out_file : str,
                                    model_file : str = "",
                                    variation : str = "stddev", 
                                    max_realizations : int = 100,
                                    verbose : bool = False,
                                    simulate_visibilities : bool = False,
    ) -> tuple[np.ndarray[complex] | np.ndarray[complex] | np.ndarray[complex]]:

        with open(f'calico/data/{run_params_filename}.pkl', 'rb') as file:
            run_params_list = pickle.load(file)
        print(data_file, run_params_filename, out_file)
        g_runs = []
        u_runs = []
        m_runs = []
        v_runs = []
        vT_runs = []
        n_runs = []
        e_runs_short = []
        e_runs_long = []
        uv_runs = []
        run_times = []
        i = 0
        for n, run_params in enumerate(run_params_list):
            print("Run", i := i + 1)
            calibration_start_time = time.perf_counter()
            # TODO: Handle paths more programmatically with os package
            data_path = f'calico/data/{data_file}.uvfits'
            if model_file:
                model_path = f'calico/data/{model_file}.uvfits'
            else:
                model_path = f'calico/data/{data_file}.uvfits'
            print("***TEST - sigma_e_0***", run_params["sigma_e"])
            uvc,g,u,m,v,vT,n,el,es,uv = calwrap.unified_calibration_wrapper(
                data=data_path,
                model=model_path,
                parallel=False,
                verbose=verbose,
                glim=None,
                ulim=None,
                antenna_gain_weights=None,
                model_baseline_weights=None,
                threshold_length=100,
                weighting_function=run_params["weighting_function"],
                power=2,
                sigma_t_0=run_params["sigma_t"],
                sigma_m_0=run_params["sigma_m"],
                sigma_n_0=run_params["sigma_n"],
                sigma_e_0=run_params["sigma_e"],
                gain_realizations=run_params["gain_realizations"],
                model_realizations=run_params["model_realizations"],
                scaling_factor_sim=run_params["scaling_factor_sim"],
                scaling_factor_cost=run_params["scaling_factor_cost"],
                simulate_visibilties=simulate_visibilities,
            )
            total_time = (time.perf_counter() - calibration_start_time) / 60
            g_runs.append(g)
            u_runs.append(u)
            m_runs.append(m)
            v_runs.append(v)
            vT_runs.append(vT)
            n_runs.append(n)
            e_runs_short.append(es)
            e_runs_long.append(el)
            uv_runs.append(uv)
            run_times.append(total_time)

        g_runs = np.asarray(g_runs)
        u_runs = np.asarray(u_runs)
        m_runs = np.asarray(m_runs)
        v_runs = np.asarray(v_runs)
        vT_runs = np.asarray(vT_runs)
        n_runs = np.asarray(n_runs)
        e_runs_short = np.asarray(e_runs_short)
        e_runs_long = np.asarray(e_runs_long)
        uv_runs = np.asarray(uv_runs)
        run_times = np.asarray(run_times)
        np.save(f'calico/data/{out_file}_g_runs', g_runs)
        np.save(f'calico/data/{out_file}_u_runs', u_runs)
        np.save(f'calico/data/{out_file}_m_runs', m_runs)
        np.save(f'calico/data/{out_file}_v_runs', v_runs)
        np.save(f'calico/data/{out_file}_vT_runs', vT_runs)
        np.save(f'calico/data/{out_file}_n_runs', n_runs)
        np.save(f'calico/data/{out_file}_e_runs_short', e_runs_short)
        np.save(f'calico/data/{out_file}_e_runs_long', e_runs_long)
        np.save(f'calico/data/{out_file}_uv_runs', uv_runs)
        np.save(f'calico/data/{out_file}_run_times', run_times)
        return g_runs, u_runs, v_runs

    def calculate_many_realizations(
        self,
        run_params_filename          : str    = 'baseline_dependence_runs_large_noise',
        vis_data_writeout_filename   : str    = 'tutorial_full_onetime_unflagged',
        model_data_writeout_filename : str    = 'tutorial_full_onetime_unflagged',
        verbose                      : bool   = True,
        caldata_obj                           = None,
        freq_ind                     : int    = 0,
        vis_pol_ind                  : int    = 0,
        feed_pol_ind                 : int    = 0,
        suffix                       : str    = "",
        metadata                     : dict   = None,
        example_data                 : UVData = None,
        optimization_scheme          : str    = "powell",
        calibration_type             : str    = "unical",
        xtol                         : float  = 1e-5,
        maxiter                      : int    = 200,
        force_fit_to_true_vis        : bool   = False,
        gains_real_guess                      = None,
    ) -> None:
        # TODO: Currently freq ind will work for 0 and we're not worried about multiple
        #       freqs so there's no immediate issue. However we will want to get there
        #       eventually, and right now the used freq ind is set inside the unical
        #       function on the caldata object in a for loop through the different freqs.
        #       So that's not accessible right now to this many realizations study. It's
        #       worth think about whether or not we do in fact care about doing many
        #       freqs for the many realizations study and if we do if it's possible to
        #       move the freq ind loop out of the unical function somehow. Ditto for
        #       the pol inds which are being done in the optimization wrappers.
        # TODO: Also, should the filenames be passed through the unical wrapper? I don't
        #       know of another to get it here but I don't like the idea of exposing
        #       that to the unical wrapper (which the user is interacting with). Perhaps
        #       instead the filenames could be programmatically determined from the data
        #       and model paths which are naturally exposed in the wrapper?
        start_many_real_time = time.time()
        data_path = os.getcwd() + f'/calico/data/{vis_data_writeout_filename}'
        model_path = os.getcwd() + f'/calico/data/{model_data_writeout_filename}'
        run_params_path = os.getcwd() + f'/calico/data/{run_params_filename}'
        if verbose:
            print("settings path", run_params_path)
        with open(f'{run_params_path}.json', 'r') as file:
            run_params_list = json.load(file)

        # preserve deep copies of original data and model from uvfits file
        original_data_vis = copy.deepcopy(caldata_obj.data_visibilities[0,:,freq_ind,vis_pol_ind])
        original_model_vis = copy.deepcopy(caldata_obj.model_visibilities[0,:,freq_ind,vis_pol_ind])
        print(f"\n\n\n***ORIGINAL DATA VIS AVG***\n\t{np.mean(np.abs(original_data_vis))}\n\n\n")
        print(f"\n\n\n***ORIGINAL MODEL VIS AVG***\n\t{np.mean(np.abs(original_model_vis))}\n\n\n")

        if verbose:
            print("len settings list", len(run_params_list))
        for run, run_params in enumerate(run_params_list):
            data_vis_realizations = []
            model_vis_realizations = []
            noise_realizations = []
            model_err_realizations = []
            model_err_realizations_long = []
            model_err_realizations_short = []
            num_thermal_realizations = run_params['thermal_noise_realizations']
            num_model_realizations = run_params['model_error_realizations']
            threshold_length = run_params['threshold_length']

            if verbose:
                print("Number of runs", run)

            # do one if not set
            if verbose:
                print("num thermal realizations (before)", num_thermal_realizations)
                print("num model realizations (before)", num_model_realizations)
            if num_thermal_realizations is None: num_thermal_realizations = 1
            if num_model_realizations is None: num_model_realizations = 1
            if verbose:
                print("num thermal realizations (after)", num_thermal_realizations)
                print("num model realizations (after)", num_model_realizations)

            # reset caldata obj data and model arrays to originals and initialize new
            # data and model arrays for this run (NOTE: true_vis=model is set as an arg
            # in simulate_visibilities() in noise_and_error_simulation.py)
            caldata_obj.data_visibilities[0,:,freq_ind,vis_pol_ind] = copy.deepcopy(original_data_vis)
            initial_data_vis = copy.deepcopy(caldata_obj.data_visibilities[0,:,freq_ind,vis_pol_ind])
            caldata_obj.model_visibilities[0,:,freq_ind,vis_pol_ind] = copy.deepcopy(original_model_vis)
            initial_model_vis = copy.deepcopy(caldata_obj.model_visibilities[0,:,freq_ind,vis_pol_ind])
            initial_gains = copy.deepcopy(caldata_obj.gains[:,0,0])

            # no sim, just use vis as-is
            if num_thermal_realizations == 0: 
                if verbose: 
                    print("No thermal realizations, just using original data vis")
                data_vis_realizations.append(initial_data_vis)
            if num_model_realizations == 0:
                if verbose: 
                    print("No model realizations, just using original model vis")
                model_vis_realizations.append(initial_model_vis)
            # create realizations of thermal noise and model error
            for i in range(num_model_realizations):
                if verbose: 
                    print(f"Creating model error realization {i+1}")
                model_error_real, model_error_imag, me_real_long, me_real_short = sim.simulate_model_error(
                                                                                      Nbls=caldata_obj.Nbls,
                                                                                      sigma_e_0=np.abs(run_params['sigma_e']),
                                                                                      uv_norm_array=caldata_obj.uv_norm,
                                                                                      threshold_length=threshold_length,
                                                                                      weighting_function=run_params['weighting_function'],
                                                                                      scaling_factor=run_params['scaling_factor_sim'],
                                                                                      seed=i+1,)
                if model_error_real is None:
                    if verbose: 
                        print("Did not simulate model error")
                    model_error_real = 0
                    model_error_imag = 0
                this_model_error = model_error_real + 1.0j*model_error_imag 
                model_err_realizations.append(this_model_error)
                # vT < m
                if run_params['sigma_e'] < 0:
                    model_vis_realizations.append(initial_model_vis + this_model_error)
                # vT > m
                elif run_params['sigma_e'] >= 0:
                    model_vis_realizations.append(initial_model_vis)
                    initial_data_vis += this_model_error
                model_err_realizations_long.append(me_real_long)
                model_err_realizations_short.append(me_real_short)
            for i in range(num_thermal_realizations):
                if verbose: 
                    print(f"Creating thermal noise realization {i+1}")
                thermal_noise_real, thermal_noise_imag = sim.simulate_thermal_noise(
                                                             sigma_t_0=run_params['sigma_t'],
                                                             Nbls=caldata_obj.Nbls,
                                                             seed=i+2,)
                if thermal_noise_real is None:
                    if verbose: 
                        print("Did not simulate thermal noise")
                    thermal_noise_real = 0
                    thermal_noise_imag = 0
                this_thermal_noise = thermal_noise_real + 1.0j*thermal_noise_imag
                data_vis_realizations.append(initial_data_vis + this_thermal_noise)
                noise_realizations.append(thermal_noise_real)

            full_data_realizations = np.array([])
            full_model_realizations = np.array([])
            gain_params_realizations = np.array([])
            model_params_realizations = np.array([])
            true_sky_realizations = np.array([])
            full_noise_realizations = np.array([])
            full_error_realizations = np.array([])
            cost_function_realizations = []

            sum_data_realizations     = np.zeros_like(initial_data_vis)
            sum_model_realizations    = np.zeros_like(initial_model_vis)
            sum_u_params_realizations = np.zeros_like(initial_model_vis)
            sum_gains_realizations    = np.zeros_like(initial_gains)
            counter                   = 0

            for j, data in enumerate(data_vis_realizations):
                if verbose: print(f"Optimization - Data thermal noise realization {j+1}")
                for k, model in enumerate(model_vis_realizations):
                    if verbose: print(f"Optimization - Model error realization {k+1}")
                    caldata_obj.data_visibilities[0,:,freq_ind,vis_pol_ind] = data
                    caldata_obj.model_visibilities[0,:,freq_ind,vis_pol_ind] = model
                    if force_fit_to_true_vis:
                        caldata_obj.fit_vis[0,:,freq_ind,vis_pol_ind] = original_data_vis
                    if gains_real_guess is not None:
                        caldata_obj.gains[:,freq_ind,feed_pol_ind].real = gains_real_guess
                    vwa = variable_weights.VariableWeightsArray()
                    vwa.set_algorithm_weights(
                        caldata_obj,
                        weighting_function=run_params['weighting_function'],
                        scaling_factor=run_params['scaling_factor_cost'],
                        sigma_t_0=run_params['sigma_t'],
                        sigma_m_0=run_params['sigma_m'],
                        threshold_length=caldata_obj.threshold_length
                    )
                    if calibration_type == "unical":
                        caldata_obj.unified_calibration(
                            verbose=verbose,
                            maxiter=maxiter,
                            xtol=xtol,
                            optimization_scheme=optimization_scheme,
                        )
                    elif calibration_type == "skycal":
                        caldata_obj.sky_based_calibration(
                            verbose=verbose,
                            maxiter=maxiter,
                            xtol=xtol,
                        )
                    else:
                        raise ValueError("Unknown calibration type -- possibilities are 'unical' and 'skycal'")
                    
                    # store data
                    full_data_realizations = np.concatenate((full_data_realizations, data))
                    full_model_realizations = np.concatenate((full_model_realizations, model))
                    gains = copy.deepcopy(caldata_obj.gains[:,freq_ind,feed_pol_ind])
                    gain_params_realizations = np.concatenate((gain_params_realizations, gains))
                    u_params = copy.deepcopy(caldata_obj.fit_vis[0,:,freq_ind,vis_pol_ind])
                    # print(f"***U PARAMS***\n{u_params}\n\n")
                    model_params_realizations = np.concatenate((model_params_realizations, u_params))
                    true_sky_realizations = np.concatenate((true_sky_realizations, initial_data_vis))
                    full_noise_realizations = np.concatenate((
                        full_noise_realizations, 
                        noise_realizations[j]
                    ))
                    full_error_realizations = np.concatenate((
                        full_error_realizations,
                        model_err_realizations[j]
                    ))
                    # full_m_err_long_realizations = np.concatenate((
                    #     full_m_err_long_realizations,
                    #     model_err_long_read_realizations[k]
                    # ))
                    # full_m_err_long_realizations = np.concatenate((
                    #     full_m_err_short_realizations,
                    #     model_err_short_read_realizations[k]
                    # ))
                    uv_array = caldata_obj.uv_array
                    # get value of first cost function term g*gv-u
                    gains_expanded = (gains[caldata_obj.ant1_inds] * np.conj(gains[caldata_obj.ant2_inds]))[np.newaxis, :]
                    residual_vector = data - gains_expanded * u_params
                    cost = np.sum(caldata_obj.visibility_weights[0,:,freq_ind,vis_pol_ind] * np.abs(residual_vector) ** 2)
                    cost_function_realizations.append(cost)

                    sum_data_realizations += data
                    sum_model_realizations += model
                    sum_u_params_realizations += u_params
                    sum_gains_realizations += gains
                    counter += 1

            if verbose:
                print("***ARRAY SHAPE***")
                print("\tfull data realizations\t\t", full_data_realizations.shape)
                print("\tfull model realizations\t\t", full_model_realizations.shape)
                print("\tgain params realizations\t", gain_params_realizations.shape)
                print("\tmodel params realizations\t", model_params_realizations.shape)
                print("\ttrue sky realizations\t\t", true_sky_realizations.shape)
                print("\tfull noise realizations\t\t", full_noise_realizations.shape)
                print("\tfull model error realizations\t\t", full_error_realizations.shape)
                try:
                    print("\tmodel err long realizations\t", model_err_realizations_long[0].shape)
                except:
                    print("No model error simulated on long baselines.")
                try:
                    print("\tmodel err short realizations\t", model_err_realizations_short[0].shape)
                except:
                    print("No model error simulated on short baselines.")
                print("\tuv array\t\t\t", uv_array.shape)
                print("\tcost function realizations\t\t", len(cost_function_realizations))
            output_arrays = {
                'v runs'    : full_data_realizations,
                'm runs'    : full_model_realizations,
                'g runs'    : gain_params_realizations,
                'u runs'    : model_params_realizations,
                'vT runs'   : true_sky_realizations,
                'n runs'    : full_noise_realizations,
                'e runs'    : full_error_realizations,
                'uv array'  : uv_array,
                'cost runs' : np.asarray(cost_function_realizations),
            }
            try:
                output_arrays['e runs long']  = model_err_realizations_long[0]
                output_arrays['e runs short'] = model_err_realizations_short[0] 
            except:
                if verbose:
                    print(f"No baseline dependent model error realizations to write")
            with open(
                f'{model_path}_many_reals_output_data_{run_params_filename}_{run}.pkl', 
                mode='wb'
            ) as file:
                print(f"data path {model_path}")
                print(f"file\n\t{file}")
                pickle.dump(output_arrays, file)

            # uvd = copy.deepcopy(example_data)
            # uvm = copy.deepcopy(example_data)
            # uvu = copy.deepcopy(example_data)

            # uvd.data_array = sum_data_realizations[
            #     ..., np.newaxis, np.newaxis
            # ] / counter
            # uvm.data_array = sum_model_realizations[
            #     ..., np.newaxis, np.newaxis
            # ] / counter
            # uvu.data_array = sum_model_realizations[
            #     ..., np.newaxis, np.newaxis
            # ] / counter

            # uvd.extra_keywords = metadata
            # uvm.extra_keywords = metadata
            # uvu.extra_keywords = metadata

            # uvg = caldata_obj.convert_to_uvcal()
            # uvg.telescope.name = "MWA"
            # # uvg.set_telescope_params(overwrite=True)
            # uvg.telescope.Nants = sum_gains_realizations.size
            # uvg.telescope.antenna_names = uvg.antenna_names
            # uvg.telescope.antenna_numbers = uvg.antenna_numbers
            # uvg.telescope.antenna_positions = uvg.antenna_positions
            # uvg.gain_array = sum_gains_realizations[
            #     ..., np.newaxis, np.newaxis, np.newaxis
            # ] / counter
            # uvg.extra_keywords = metadata

            # uvfits_writeout_filename = f"calico/data/many_realizations_out_{run}_avg"
            # uvd.write_uvfits(f"calico/data/{uvfits_writeout_filename}_v_{suffix}.uvfits")
            # uvm.write_uvfits(f"calico/data/{uvfits_writeout_filename}_m_{suffix}.uvfits")
            # uvu.write_uvfits(f"calico/data/{uvfits_writeout_filename}_u_{suffix}.uvfits")
            # # uvg.write_calfits(f"{uvfits_writeout_filename}_g_{suffix}.calfits")

            # np.save(f"calico/data/{uvfits_writeout_filename}_g_{suffix}.npy", 
            #         sum_gains_realizations / counter)

            if verbose:
                print(f"***FINISHED RUN {run}***")
                finish_time = (time.time() - start_many_real_time)/3600
                print(f"\n\t{finish_time=:.4f} hours\n")

    def plot_many_realizations(
        self, 
        variation           : str  = "stddev", 
        data_filepath       : str  = "", 
        run_params_filename : str  = "",
        threshold_length    : int  = 100,
        simulation_type     : str  = "gaussian",
        verbose             : bool = False,
        suffix              : str  = "",
        metadata            : dict = None,
        save_plot           : bool = True,
    ) -> None:

        with open(
            f'calico/data/{run_params_filename}.json', 
            mode='r',
        ) as file:
            run_params_list = json.load(file)

        output_dicts = []

        fig, ax = plt.subplots(len(run_params_list), 
                               9, 
                               figsize=(43,3*len(run_params_list)), 
                               squeeze=False, 
                               sharex=False, 
                               sharey=False)
        
        for run, run_params in enumerate(run_params_list):
            with open(
                f'{data_filepath}_many_reals_output_data_{run_params_filename}_{run}.pkl', 
                mode='rb',
            ) as file:
                output_arrays = pickle.load(file)

            g_arr    = output_arrays['g runs'] - 1
            u_arr    = output_arrays['u runs']
            m_arr    = output_arrays['m runs']
            v_arr    = output_arrays['v runs']
            vT_arr   = output_arrays['vT runs']
            n_arr    = output_arrays['n runs']
            e_arr    = output_arrays['e runs']
            uv_arr   = output_arrays['uv array']
            cost_arr = output_arrays['cost runs']

            split_model_error_arrays = True
            try:
                e_short_arr = output_arrays['e runs short']
                e_long_arr = output_arrays['e runs long']
            except:
                e_short_arr = None
                e_long_arr = None
            if not (np.any(e_short_arr) and np.any(e_long_arr)):
                split_model_error_arrays = False

            max_realizations = max([run_params["thermal_noise_realizations"], 
                                    run_params["model_error_realizations"]])

            u_minus_m  = u_arr - m_arr
            u_minus_vT = u_arr - vT_arr
            v_minus_vT = v_arr - vT_arr

            # set constants
            if variation == "stddev":
                g_var = np.std(g_arr)
                um_var = np.std(u_minus_m)
                uvT_var = np.std(u_minus_vT)
                v_var = np.std(v_arr)
                vT_var = np.std(vT_arr)
            elif variation == "iqr":
                g_var_real = np.percentile(g_arr.real, 75) - np.percentile(g_arr.real, 25)
                um_var_real = np.percentile(u_minus_m.real, 75) - np.percentile(u_minus_m.real, 25)
                uvT_var_real = np.percentile(u_minus_vT.real, 75) - np.percentile(u_minus_vT.real, 25)
                g_var_imag = np.percentile(g_arr.imag, 75) - np.percentile(g_arr.imag, 25)
                um_var_imag = np.percentile(u_minus_m.imag, 75) - np.percentile(u_minus_m.imag, 25)
                uvT_var_imag = np.percentile(u_minus_vT.imag, 75) - np.percentile(u_minus_vT.imag, 25)
                g_var = np.sqrt(g_var_real**2 + g_var_imag**2)
                um_var = np.sqrt(um_var_real**2 + um_var_imag**2)
                uvT_var = np.sqrt(uvT_var_real**2 + uvT_var_imag**2)

            sigma_re_vT = np.std(vT_arr.real)
            if simulation_type == "fhd":
                title=f"FHD simulation with source cutoff at 0.15 Jy\n$\\sigma_v$ = {sigma_re_vT:.2f} for True Visibilities\n\n"
            elif simulation_type == "gaussian":
                title=f"Gaussian simulation of data, model, noise, and error (additive)\n$\\sigma_v$ = {sigma_re_vT:.2f} for True Visibilities\n\n"
            else:
                if verbose: print("Warning: Invalid simulation type given")
                title=f"<<Unrecognized simulation type>>\n$\\sigma_v$ = {sigma_re_vT:.2f} for True Visibilities\n\n"
            fig.suptitle(title, fontsize="25", fontweight="bold")

            g_boundary = np.max([np.abs(np.min(g_arr.real)),
                                 np.abs(np.max(g_arr.real))])
            if np.isnan(g_boundary) or np.isinf(g_boundary):
                print("Plot Many Realizations - g_boundary is inf or nan, setting to 1")
                g_boundary = 1
            # um_boundary = np.max([np.abs(np.min(u_minus_m.real)),
            #                       np.abs(np.max(u_minus_m.real))])
            # if np.isnan(um_boundary) or np.isinf(um_boundary):
            #     print("Plot Many Realizations - um_boundary is inf or nan, setting to 1")
            #     um_boundary = 1
            # uvT_boundary = np.max([np.abs(np.min(u_minus_vT.real)),
            #                        np.abs(np.max(u_minus_vT.real))])
            # if np.isnan(uvT_boundary) or np.isinf(uvT_boundary):
            #     print("Plot Many Realizations - uvT_boundary is inf or nan, setting to 1")
            #     uvT_boundary = 1
            # vT_boundary = np.max([np.abs(np.min(vT_arr)),
            #                       np.abs(np.max(vT_arr))])
            # if np.isnan(vT_boundary) or np.isinf(vT_boundary):
            #     print("Plot Many Realizations - vT_boundary is inf or nan, setting to 1")
            #     vT_boundary = 1
            v_boundary = np.max([np.abs(np.min(v_arr.real)),
                                 np.abs(np.max(v_arr.real))])
            if np.isnan(v_boundary) or np.isinf(v_boundary):
                print("Plot Many Realizations - v_boundary is inf or nan, setting to 1")
                v_boundary = 1
            m_boundary = np.max([np.abs(np.min(m_arr.real)),
                                 np.abs(np.max(m_arr.real))])
            if np.isnan(m_boundary) or np.isinf(m_boundary):
                print("Plot Many Realizations - m_boundary is inf or nan, setting to 1")
                m_boundary = 1
            n_boundary = np.max([np.abs(np.min(n_arr.real)),
                                 np.abs(np.max(n_arr.real))])
            if np.isnan(n_boundary) or np.isinf(n_boundary):
                print("Plot Many Realizations - n_boundary is inf or nan, setting to 1")
                n_boundary = 1
            el_boundary = None
            es_boundary = None
            if split_model_error_arrays:
                el_boundary = np.max([np.min(np.abs(e_long_arr.real)), 
                                      np.max(np.abs(e_long_arr.real))])
                es_boundary = np.max([np.min(np.abs(e_short_arr.real)), 
                                      np.max(np.abs(e_short_arr.real))])

            # set bin sizes
            # g_step = g_var / 7.5
            # g_step = 0.1
            # um_step = 0.05
            # uvT_step = 0.05
            g_step = g_boundary / 3
            # um_step = um_boundary / 10
            # uvT_step = uvT_boundary / 10
            v_step = vT_var / 7.5  # change to appropriate fixed size
            e_step = 0.2

            g_bins = np.arange(-g_boundary, g_boundary, g_step)
            # um_bins = np.arange(-um_boundary, um_boundary, um_step)
            # uvT_bins = np.arange(-uvT_boundary, uvT_boundary, uvT_step)
            # vT_bins = np.arange(-vT_boundary, vT_boundary, uvT_step)
            v_bins = np.arange(-v_boundary, v_boundary, e_step)
            m_bins = np.arange(-m_boundary, m_boundary, e_step)
            n_bins = np.arange(-n_boundary, n_boundary, e_step)
            if el_boundary is not None: 
                el_bins = np.arange(-el_boundary, el_boundary, e_step)
            if es_boundary is not None:
                es_bins = np.arange(-es_boundary, es_boundary, e_step)

            # calculate centers
            if variation == "stddev":
                g_center_real = np.mean(g_arr.real)
                g_center_imag = np.mean(g_arr.imag)
                um_center_real = np.mean(u_minus_m.real)
                um_center_imag = np.mean(u_minus_m.imag)
                # uvT_center_real = np.mean(u_minus_vT.real)
                # uvT_center_imag = np.mean(u_minus_vT.imag)
                vT_center_real = np.mean(v_arr.real)
                vT_center_imag = np.mean(v_arr.imag)
            elif variation == "iqr":
                g_center = np.median(g_arr)
                um_center = np.median(u_minus_m)
                uvT_center = np.median(u_minus_vT)

            if verbose:
                print(f"g arr mean {np.mean(g_arr)}\ng arr real/imag of mean {np.mean(g_arr).real} {np.mean(g_arr).imag}")
                print(f"g arr mean of real {np.mean(g_arr.real)} mean of imag {np.mean(g_arr.imag)}\n")

            # get histograms
            # vT_real_hist, vT_real_bins = np.histogram(
            #     vT_arr.real,
            #     bins=vT_bins,
            #     density=True
            # )
            data_hist, data_bins = np.histogram(
                v_arr.real, 
                bins=v_bins, 
                density=True
            )
            model_hist, model_bins = np.histogram(
                m_arr.real, 
                bins=m_bins, 
                density=True
            )
            noise_hist, noise_bins = np.histogram(
                n_arr.real, 
                bins=n_bins, 
                density=True
            )
            if el_boundary is not None:
                error_long_hist, error_long_bins = np.histogram(
                    e_long_arr, 
                    bins=el_bins, 
                    density=True
                )
            if es_boundary is not None:
                error_short_hist, error_short_bins = np.histogram(
                    e_short_arr, 
                    bins=es_bins, 
                    density=True
                )
            g_real_hist, g_real_bins = np.histogram(
                g_arr.real, 
                bins=g_bins, 
                density=True
            )
            g_imag_hist, g_imag_bins = np.histogram(
                g_arr.imag,
                bins=g_bins, 
                density=True
            )
            gains_hist2d, gains_real2d, gains_imag2d = np.histogram2d(
                g_arr.real, 
                g_arr.imag, 
                bins=g_bins, 
                density=True
            )
            # um_hist2d, um_real2d, um_imag2d = np.histogram2d(
            #     u_minus_m.real, 
            #     u_minus_m.imag, 
            #     bins=um_bins, 
            #     density=True
            # )
            # uvT_hist2d, uvT_real2d, uvT_imag2d = np.histogram2d(
            #     u_minus_vT.real, 
            #     u_minus_vT.imag, 
            #     bins=uvT_bins, 
            #     density=True
            # )

            glim    = 1.0*g_boundary
            # uvT_lim = 0.5*uvT_boundary

            if np.isnan(glim) or np.isinf(glim):
                glim = 1
            # if np.isnan(uvT_lim) or np.isinf(uvT_lim):
            #     uvT_lim = 1

            uv_norm   = np.linalg.norm(uv_arr, axis=1)
            uv_extend = np.array([])
            for _ in range(max_realizations):
                uv_extend = np.concatenate((uv_extend, uv_norm))

            ax[run,0].set_axis_off()
            if run == 0:
                ax[run,0].text(0.2,1.03,"Cost Function Value\n(Gains Term)", fontsize="22")
            ax[run,0].text(0.4,0.6,f"{np.mean(cost_arr):.2f}", fontsize="17")

            # plot g real and imaginary histograms
            ax[run,1].plot(
                g_real_bins[:-1], 
                g_real_hist, 
                label="Real",
            )
            ax[run,1].plot(
                g_imag_bins[:-1], 
                g_imag_hist, 
                label="Imaginary",
            )
            # glim = 0.4
            ax[run,1].set_xlim(-glim, glim)
            g_1d_max = np.max([np.max(g_real_hist), np.max(g_imag_hist)])
            if np.isnan(g_1d_max) or np.isinf(g_1d_max):
                g_1d_max = 1

            # g_1d_max = 8
            ax[run,1].set_ylim(0,g_1d_max)
            if run == 0:
                ax[run,1].set_title(
                    "1D Gains Error\n(Real=Blue, Imag=Orange)", 
                    fontsize="22",
                )
            ax[run,1].tick_params(labelbottom=True, labelleft=True)

            # initial gains
            g_vmax = np.max(gains_hist2d)
            if np.isnan(g_vmax) or np.isinf(g_vmax):
                if verbose:
                    print("Ploting - g_vmax is inf or nan, setting to 1")
                g_vmax = 1
            # g_vmax = 15000
            im = ax[run,2].pcolormesh(
                gains_real2d, 
                gains_imag2d, 
                gains_hist2d.T, 
                cmap="viridis", 
                vmin=0, 
                vmax=g_vmax, 
                rasterized=True
            )
            ax[run,2].add_patch(plt.Circle(
                (g_center_real, 
                g_center_imag), 
                radius=g_var, 
                fill=False, 
                color="white"
            ))
            ax[run,2].plot(0, 0, 'w^')
            ax[run,2].set_ylabel("Imag")
            ax[run,2].set_xlabel("Real - 1")
            # glim = 0.021
            ax[run,2].set_xlim(-glim, glim)
            ax[run,2].set_ylim(-glim, glim)
            if run == 0:
                ax[run,2].set_title(
                    f"2D Gains Error\n(Complex Plane)", 
                    fontsize="22",
                )
            ax[run,2].tick_params(labelbottom=True, labelleft=True)

            # |u-m| hist for short and long baselines
            if el_boundary is not None:
                long_um = np.abs(u_minus_m)[uv_extend >= threshold_length]
            if es_boundary is not None:
                short_um = np.abs(u_minus_m)[uv_extend < threshold_length]
            if es_boundary is not None:
                ax[run,3].hist(short_um, bins=50, label="Short Baselines", histtype="step")
            if el_boundary is not None:
                ax[run,3].hist(long_um, bins=50, label="Long Baselines", histtype="step")
            # else:
            #     ax[run,3].hist(np.abs(u_minus_m), bins=um_bins, histtype="step")
            # ax[run,3].set_xlim(0,um_boundary)
            # ax[run,3].set_xlim(0,5)
            ax[run,3].set_xlabel("(Jy)")
            if run == 0:
                ax[run,3].set_title("1D False Model Error", fontsize="22")
            ax[run,3].tick_params(labelbottom=True, labelleft=True)
            ax[run,3].legend()

            # |u-v_T| hist for short and long baselines
            if el_boundary is not None:
                long_uvT = np.abs(u_minus_vT)[uv_extend >= threshold_length]
            if es_boundary is not None:
                short_uvT = np.abs(u_minus_vT)[uv_extend < threshold_length]
            if es_boundary is not None:
                ax[run,4].hist(
                    short_uvT, 
                    bins=50, 
                    label="Short Baselines", 
                    histtype="step",
                )
            if el_boundary is not None:
                ax[run,4].hist(
                    long_uvT, 
                    bins=50, 
                    label="Long Baselines",
                    histtype="step",
                )
            # else:
            #     ax[run,4].hist(
            #         np.abs(u_minus_vT), 
            #         bins=uvT_bins, 
            #         histtype="step",
            #     )
            # ax[run,4].set_xlim(0,uvT_boundary)
            # ax[run,4].set_xlim(0,4)
            ax[run,4].set_xlabel("(Jy)")
            if run == 0:
                ax[run,4].set_title(
                    "1D True Model Error", 
                    fontsize="22",
                )
            ax[run,4].tick_params(labelbottom=True, labelleft=True)
            ax[run,4].legend()

            # initial models
            # uvT_vmax = np.max(uvT_hist2d)
            # if np.isnan(uvT_vmax) or np.isinf(uvT_vmax):
            #     print("Plot Many Realizations - uvT_vmax is inf or nan, setting to 1")
            #     uvT_vmax = 1
            # uvT_vmax = 2
            # im2 = ax[run,5].pcolormesh(
            #     uvT_real2d, 
            #     uvT_imag2d, 
            #     uvT_hist2d, 
            #     cmap="inferno", 
            #     vmin=0, 
            #     vmax=uvT_vmax, 
            #     rasterized=True,
            # )
            # ax[run,5].add_patch(plt.Circle(
            #     (uvT_center_real, 
            #      uvT_center_imag), 
            #      radius=uvT_var, 
            #      fill=False, 
            #      color="white",
            # ))
            # ax[run,5].set_ylabel("Imag")
            # ax[run,5].set_xlabel("Real")
            # uvT_lim = 1.5
            # ax[run,5].set_xlim(-uvT_lim, uvT_lim)
            # ax[run,5].set_ylim(-uvT_lim, uvT_lim)
            # if run == 0:
            #     ax[run,5].set_title(
            #         f"2D True Model Error\n(Complex Plane)", 
            #         fontsize="22",
            #     )

            # plot distributions for vis data, thermal noise, and long/short model errors
            # ax[run,6].stairs(vT_real_hist, vT_real_bins, label="vT")
            ax[run,6].stairs(model_hist, model_bins, label="m")
            ax[run,6].set_xlabel("Real")
            if run == 0:
                ax[run,6].set_title(
                    "Data Distributions", 
                    fontsize="22",
                )
            ax[run,6].tick_params(labelbottom=True, labelleft=True)
            ax[run,6].legend()

            # main calculated quantities
            sigma_re_m = np.std(m_arr.real)
            avg_mag_model = np.mean(np.abs(m_arr))
            avg_mag_vT = np.mean(np.abs(vT_arr))
            sigma_re_vTm = np.std(vT_arr.real - m_arr.real)
            avg_mag_vTm = np.mean(np.abs(vT_arr - m_arr))
            sigma_re_n = np.std(v_arr.real - vT_arr.real)
            avg_mag_v = np.mean(np.abs(v_arr))
            ax[run,7].set_axis_off()
            if run == 0:
                ax[run,7].text(0.0,1.03,"Output Calcs", fontsize="22")
            ax[run,7].text(
                0.0, 0.85,
                f"$\\sigma$ Re($m$) = {sigma_re_m:.2f}", 
                fontsize="13",
            )
            ax[run,7].text(
                0.0, 0.71,
                f"$<|m|>$ = {avg_mag_model:.2f}", 
                fontsize="13",
            )
            ax[run,7].text(
                0.0, 0.57,
                f"$<|v_T|>$ = {avg_mag_vT:.2f}", 
                fontsize="13",
            )
            ax[run,7].text(
                0.0, 0.43,
                f"$\\sigma$ Re($v_T-m$) = {sigma_re_vTm:.2f}", 
                fontsize="13", 
                fontweight="bold",
            )
            ax[run,7].text(
                0.0, 0.29,
                f"$<|v_T-m|>$ = {avg_mag_vTm:.2f}",
                fontsize="13",
            )
            ax[run,7].text(
                0.0, 0.15,
                f"$\\sigma$ Re($n$) (out) = {sigma_re_n:.2f}", 
                fontsize="13", 
                fontweight="bold",
            )
            ax[run,7].text(
                0.0, 0.0,
                f"$<|v|>$ = {avg_mag_v:.2f}", 
                fontsize="13",
            )

            if run == 0:
                ax[run,7].text(
                    1.02, 1.03,
                    "Given to Algo", 
                    fontsize="22",
                )
            ax[run,7].text(
                1.05, 0.85,
                f"$\\sigma_t$ = {run_params['sigma_t']:.2f}",
                fontsize="13",
            )
            ax[run,7].text(
                1.05, 0.71,
                f"$\\sigma_n$ (in) = {run_params['sigma_n']:.2f}",
                fontsize="13",
            )
            if run == 2 or run == len(run_params_list) - 1:
                ax[run,7].text(
                    1.05, 0.57,
                    f"$\\sigma_e$ = {run_params['sigma_m'] \
                        / (run_params['scaling_factor_cost'])**2:.2f}", 
                    fontsize="15",
                    fontweight="bold",
                )
            else:
                ax[run,7].text(
                    1.05, 0.57,
                    f"$\\sigma_e$ = {run_params['sigma_m'] \
                        / (run_params['scaling_factor_cost'])**2:.2f}", 
                    fontsize="13",
                )

            # additional calculated quantities
            avg_re_g_offset = np.mean(g_arr.real)
            avg_im_g_offset = np.mean(g_arr.imag)
            avg_mag_um = np.mean(np.abs(u_minus_m))
            avg_mag_uvT = np.mean(np.abs(u_minus_vT))
            sigma_re_g = np.std(g_arr.real)
            sigma_im_g = np.std(g_arr.imag)
            sigma_re_um = np.std(u_minus_m.real)
            sigma_re_uvT = np.std(u_minus_vT.real)
            avg_mag_u = np.mean(np.abs(u_arr))
            sigma_re_u = np.std(u_arr.real)
            ax[run,8].set_axis_off()
            if run == 0:
                ax[run,8].text(
                    0.2, 1.03,
                    "More Calcs", 
                    fontsize="22",
                )
            ax[run,8].text(
                0.2, 0.85,
                f"$<Re(g-1)>$ = {avg_re_g_offset:.6f}", 
                fontsize="13", 
                fontweight="bold",
            )
            ax[run,8].text(
                0.2, 0.71,
                f"$<Im(g)>$ = {avg_im_g_offset:.6f}", 
                fontsize="13",
            )
            ax[run,8].text(
                0.2, 0.57,
                f"$<|u-m|>$ = {avg_mag_um:.2f}   $<|u-v_T|> = {avg_mag_uvT:.2f}$", 
                fontsize="13",
            )
            if variation == "stddev":
                ax[run,8].text(
                    0.2, 0.43,
                    f"$\\sigma$ Re($g$) = {sigma_re_g:.2f}\t\t  $\\sigma$ Im($g$): {sigma_im_g:.2f}", 
                    fontsize="13",
                )
                ax[run,8].text(
                    0.2, 0.29,
                    f"$\\sigma$ Re($u-m$) = {sigma_re_um:.2f}    $\\sigma$ Re($u-v_T$): {sigma_re_uvT:.2f}", 
                    fontsize="13",
                )
            elif variation == "iqr":
                ax[run,8].text(
                    0.2, 0.43,
                    f"IQR($g$): {g_var:.2f}", 
                    fontsize="13",
                )
                ax[run,8].text(
                    0.2, 0.29,
                    f"IQR($u-m$): {um_var:.2f}\tIQR($u-v_T$): {uvT_var:.2f}",
                    fontsize="13",
                )
            ax[run,8].text(
                0.2, 0.15,
                f"$<|u|>$ = {avg_mag_u:.3f}", 
                fontsize="13",
            )
            ax[run,8].text(
                0.2, 0.00,
                f"$\\sigma$ Re($u$) = {sigma_re_u:.3f}", 
                fontsize="13",
            )

            # predict gains based on whether model error
            # is "additive" or "subtractive"
            # if np.mean(vT_arr) > np.mean(m_arr):
            #     e_arr_mag = np.sqrt(np.abs(vT_arr)**2 - np.abs(m_arr)**2)
            # else:
            #     e_arr_mag = np.sqrt(np.abs(m_arr)**2 - np.abs(vT_arr)**2)
            # e_arr_mag = np.abs(e_arr)
            g_squared_left = np.sqrt(
                np.sqrt(
                    np.abs(vT_arr)**2 + np.abs(n_arr)**2
                ) /
                np.sqrt(
                    np.abs(vT_arr)**2 + np.abs(e_arr)**2
                )
            )
            g_squared_right = np.sqrt(
                np.sqrt(
                    np.abs(m_arr)**2 + np.abs(e_arr)**2
                    # np.abs(vT_arr**2) + e_arr_mag**2 + 
                    # np.abs(n_arr)**2
                ) /
                # np.abs(m_arr)
                np.sqrt(
                    np.abs(m_arr)**2 + np.abs(n_arr)**2
                )
            )
            # using calculated thermal noise and model error
            # instead of passed because that's available
            alpha = 2.6e-4
            angle = np.radians(26.57)
            avg_g_offset_predict_right = alpha * (
                np.cos(angle) * avg_mag_vTm - np.sin(angle) * sigma_re_n
            )

            re_g_minus_one_left      = g_squared_left - 1
            re_g_minus_one_right     = g_squared_right - 1
            avg_re_g_minus_one_left  = re_g_minus_one_left.mean()
            avg_re_g_minus_one_right = re_g_minus_one_right.mean()
            # correlations
            model_error_thermal_noise_corr_abs = np.corrcoef(
                np.abs(e_arr), np.abs(n_arr)
            )[0,1]
            thermal_noise_model_vis_corr_abs = np.corrcoef(
                np.abs(n_arr), np.abs(m_arr)
            )[0,1]
            model_error_model_vis_corr_abs = np.corrcoef(
                np.abs(e_arr), np.abs(m_arr)
            )[0,1]
            model_error_thermal_noise_corr_phase = np.corrcoef(
                np.angle(e_arr), np.angle(n_arr)
            )[0,1]
            thermal_noise_model_vis_corr_phase = np.corrcoef(
                np.angle(n_arr), np.angle(m_arr)
            )[0,1]
            model_error_model_vis_corr_phase = np.corrcoef(
                np.angle(e_arr), np.angle(m_arr)
            )[0,1]
            print(f"m-e corr abs {model_error_model_vis_corr_abs}")
            print(f"m-e corr phase {model_error_model_vis_corr_phase}")
            print(f"n-m corr abs {thermal_noise_model_vis_corr_abs}")
            print(f"n-m corr phase {thermal_noise_model_vis_corr_phase}")
            print(f"e-m corr abs {model_error_model_vis_corr_abs}")
            print(f"e-m corr phase {model_error_model_vis_corr_phase}") 

            std_gain_phase = np.std(np.angle(g_arr))

            this_output_dict = {
                "sigma_re_m"                : sigma_re_m,
                "avg_mag_model"             : avg_mag_model,
                "avg_mag_vT"                : avg_mag_vT,
                "sigma_re_vTm"              : sigma_re_vTm,
                "avg_mag_vTm"               : avg_mag_vTm,
                "sigma_re_n"                : sigma_re_n,
                "avg_mag_v"                 : avg_mag_v,
                "avg_re_g_offset"           : avg_re_g_offset,
                "avg_im_g_offset"           : avg_im_g_offset,
                "avg_mag_um"                : avg_mag_um,
                "avg_mag_uvT"               : avg_mag_uvT,
                "sigma_re_g"                : sigma_re_g,
                "sigma_im_g"                : sigma_im_g,
                "sigma_re_um"               : sigma_re_um,
                "sigma_re_uvT"              : sigma_re_uvT,
                "avg_mag_u"                 : avg_mag_u,
                "sigma_re_u"                : sigma_re_u,
                "sigma_re_vT"               : sigma_re_vT,
                "avg_re_g_minus_one_left"   : avg_re_g_minus_one_left,
                "avg_re_g_minus_one_right"  : avg_g_offset_predict_right,
                "std_gain_phase"            : std_gain_phase,
                "scaling_factor_cost"       : run_params["scaling_factor_cost"],
                "e_n_corr_coeff"            : model_error_thermal_noise_corr_abs,
                "n_m_corr_coeff"            : thermal_noise_model_vis_corr_abs,
                "e_m_corr_coeff"            : model_error_model_vis_corr_abs,
                "e_n_corr_coeff_phase"      : model_error_thermal_noise_corr_phase,
                "n_m_corr_coeff_phase"      : thermal_noise_model_vis_corr_phase,
                "e_m_corr_coeff_phase"      : model_error_model_vis_corr_phase,
                "avg_cost_func_val"         : np.mean(cost_arr),
                "g_arr_real"                : (g_arr.real).tolist(),
            }
            output_dicts.append(this_output_dict)
            
            fig.tight_layout()

        which_sigma_t = ""
        sigma_t = run_params['sigma_t']
        while sigma_t < 1:
            which_sigma_t += "0"
            sigma_t *= 10
        which_sigma_t += str(int(sigma_t*100))

        # filename = f'calico/images/sigma_t_{which_sigma_t}_{max_realizations}-realizations_{variation}_{suffix}.png'
        # plt.savefig(
        #     filename,
        #     bbox_inches=0,
        # )
        plt.close()

        fix, ax = plt.subplots()
        plt.scatter(g_arr.real, g_arr.imag)
        ax.add_patch(plt.Circle(
            (g_center_real, 
            g_center_imag), 
            radius=g_var, 
            fill=False,
        ))
        which_model_error_type = ""
        if avg_mag_model < avg_mag_vT:
            which_model_error_type += "m < v_T"
        else:
            which_model_error_type += "m > v_T"
        plt.title(f"Gain Fits ${which_model_error_type}$\nsigma_re_vTm {sigma_re_vTm:.2f} sigma_re_n {sigma_re_n:.2f}")
        plt.ylabel("Imag")
        plt.xlabel("Real - 1")
        plt.xlim(-glim, glim)
        plt.ylim(-glim, glim)
        ax.set_aspect('equal', adjustable='datalim')
        ax.autoscale_view()
        filename = f'calico/images/sigma_t_{which_sigma_t}_gains2d_{variation}_{suffix}.png'
        # plt.savefig(
        #     filename,
        #     bbox_inches=0,
        # )
        plt.close()

        # print(f"\n\n***METADATA***\n\nLength: {len(metadata)}\n\n{metadata}\n\n")
        # metadata_str = "\n".join([f"{key}: {val}" for key, val in metadata.items()])
        # img = Image.open(filename)
        # img_metadata = PngImagePlugin.PngInfo()
        # img_metadata.add_text("Description", f"Project Settings and Info:\n{metadata_str}")
        # img.save(filename, pnginfo=img_metadata)        

        with open(
            f'calico/data/output_calcs_{suffix}.json',
            mode='w'
        ) as file:
            if verbose:
                print(f"***calculated values***")
                print(f"data path {data_filepath}")
                print(f"file\n\t{file}")
            json.dump(output_dicts, file)

    def test_function(self) -> None:
        return 'Returning a new and beautiful string, some say the best string, from within dev tools test function'

    # plot gain errors across realizations for one antenna at a time at two
    # scales: one set to "var" (stddev/IQR), the other to "max" (outliers)
    def plot_gains_one_ant_same_noise_and_error(self, 
                                                num_realizations : int = 20, 
                                                sigma : int | float = 1.0,
                                                variation : str = "stddev", 
                                                plot_type : str = "variation",
                                                data_path : str = 'data/tutorial_medium_onetime.uvfits',
                                                weights_threshold : int | float = 50,
                                                cutoff_function : str = "constant_weights"
        ) -> float:

        uvc, g_arr, u_arr = calwrap.unified_calibration_wrapper(data_path,
                                                                data_path,
                                                                parallel=False,
                                                                gain_init_stddev=0.2,
                                                                fit_vis_init_stddev=0.4,
                                                                verbose=False,
                                                                glim=None,
                                                                ulim=None,
                                                                antenna_gain_weights=None,
                                                                model_baseline_weights=None,
                                                                weights_threshold=weights_threshold,
                                                                cutoff_function=cutoff_function,
                                                                power=2,
                                                                sigma_t=sigma,
                                                                sigma_m=sigma,
                                                                sigma_n=None,
                                                                sigma_e=None,
                                                                gain_realizations=num_realizations,
                                                                model_realizations=num_realizations)
        Nants = np.size(g_arr) // num_realizations
        fig, ax = plt.subplots(Nants, 2, figsize=(13, 5*Nants), squeeze=False)

        # track average of centers
        antenna_gain_error_centers = np.zeros(Nants, dtype=complex)

        ant_array = np.zeros(num_realizations, dtype=complex)

        # plot realizations per antenna
        for ant in range(Nants):
            
            for i in range(num_realizations):
                index = i * Nants + ant
                ant_array[i] = g_arr[index]

            # set constants
            if variation == "stddev":
                g_var = np.std(ant_array)
            elif variation == "iqr":
                g_var_real = np.percentile(ant_array.real, 75) - np.percentile(ant_array.real, 25)
                g_var_imag = np.percentile(ant_array.imag, 75) - np.percentile(ant_array.imag, 25)
                g_var = np.sqrt(g_var_real**2 + g_var_imag**2)

            # calculate centers
            if variation == "stddev":
                g_center = np.mean(ant_array)
            elif variation == "iqr":
                g_center = np.median(ant_array)
            antenna_gain_error_centers[ant] = g_center

            g_offset_lo = np.min([np.min(g_center.real - g_var.real), np.min(g_center.imag - g_var.imag)])
            g_offset_hi = np.max([np.max(g_center.real + g_var.real), np.max(g_center.imag + g_var.imag)])
            g_boundary = np.max([np.abs(g_offset_lo), np.abs(g_offset_hi)])

            # plot scatter of realizations for this antenna scaled to "var" (stddev/IQR)
            ax[ant,0].scatter(ant_array.real - 1, ant_array.imag)
            ax[ant,0].set_title(f"Gain Error across {num_realizations} Realizations for Antenna {ant+1} (Lim: Var) Func: ({cutoff_function})", fontsize="7.5")
            ax[ant,0].set_xlabel("Real - 1", fontsize = "7.5")
            ax[ant,0].set_ylabel("Imag", fontsize = "7.5")
            ax[ant,0].plot(g_center.real - 1, g_center.imag, 'bx')
            ax[ant,0].plot(0,0,'rx')
            ax[ant,0].add_patch(plt.Circle((g_center.real-1, g_center.imag), radius=g_var, fill=False, color="black"))
            ax[ant,0].set_xlim(g_center.real-1-g_boundary, g_center.real-1+g_boundary)
            ax[ant,0].set_ylim(g_center.imag-g_boundary, g_center.imag+g_boundary)
            
        for ant in range(Nants):

            for i in range(num_realizations):
                index = i * Nants + ant
                ant_array[i] = g_arr[index]

            # set constants
            if variation == "stddev":
                g_var = np.std(ant_array)
            elif variation == "iqr":
                g_var_real = np.percentile(ant_array.real, 75) - np.percentile(ant_array.real, 25)
                g_var_imag = np.percentile(ant_array.imag, 75) - np.percentile(ant_array.imag, 25)
                g_var = np.sqrt(g_var_real**2 + g_var_imag**2)

            # calculate centers
            if variation == "stddev":
                g_center = np.mean(ant_array)
            elif variation == "iqr":
                g_center = np.median(ant_array)
            antenna_gain_error_centers[ant] = g_center

            # for plot limits for outlier plot (we want a square plot so same for real and imag)
            g_max_lo = np.max([np.min(g_center.real - ant_array.real), np.min(g_center.imag - ant_array.imag)])
            g_max_hi = np.max([np.max(g_center.real + ant_array.real), np.max(g_center.imag + ant_array.imag)])
            g_boundary = np.max([np.abs(g_max_lo), np.abs(g_max_hi)])

            # plot scatter of realizations for this antenna scaled to "max" (outliers)
            ax[ant,1].scatter(ant_array.real - 1, ant_array.imag)
            ax[ant,1].set_title(f"Gain Error across {num_realizations} Realizations for Antenna {ant+1} (Lim: Max) Func: ({cutoff_function})", fontsize="7.5")
            ax[ant,1].set_xlabel("Real - 1", fontsize = "7.5")
            ax[ant,1].set_ylabel("Imag", fontsize = "7.5")
            ax[ant,1].plot(g_center.real - 1, g_center.imag, 'bx')
            ax[ant,1].plot(0,0,'rx')
            ax[ant,1].add_patch(plt.Circle((g_center.real-1, g_center.imag), radius=g_var, fill=False, color="black"))
            ax[ant,1].set_xlim(g_center.real-1-1.2*g_boundary, g_center.real-1+1.2*g_boundary)
            ax[ant,1].set_ylim(g_center.imag-1.2*g_boundary, g_center.imag+1.2*g_boundary)
            
        plt.tight_layout()
        plt.savefig('images/realizations_per_antenna_limit_' + str(Nants) + '_'
            # + subprocess.check_output(['git','rev-parse','--short','HEAD']).decode('ascii').strip()
            + cutoff_function + '.png',
            bbox_inches=0,)
        plt.close(fig)
            
        # print("***AVERAGE OF CENTERS ACROSS ANTENNAS***", np.mean(antenna_gain_error_centers))
        return np.mean(np.abs(antenna_gain_error_centers - 1)), np.abs(np.mean(antenna_gain_error_centers) - 1)

    def plot_gain_error_per_realization(self, 
                                        gain_error_array : np.ndarray, 
                                        variation : str = "stddev", 
                                        plot_type : str = "variation"
        ) -> None:
        num_realizations = len(gain_error_array)

        # print("***NUM REALIZATIONS***", num_realizations)
        fig, ax = plt.subplots(num_realizations, 2, figsize=(13, 5*num_realizations), squeeze=False)

        # render LaTeX math
        plt.rcParams['text.usetex'] = True

        # track centers
        realization_centers = np.zeros(num_realizations, dtype=complex)

        for realization in range(num_realizations):

            gain_errors = gain_error_array[realization]
            Nants = np.size(gain_errors)

            # set constants
            if variation == "stddev":
                g_var = np.std(gain_errors-1)
            elif variation == "iqr":
                g_var_real = np.percentile(gain_errors.real-1, 75) - np.percentile(gain_errors.real-1, 25)
                g_var_imag = np.percentile(gain_errors.imag, 75) - np.percentile(gain_errors.imag, 25)
                g_var = np.sqrt(g_var_real**2 + g_var_imag**2)

            # calculate centers
            if variation == "stddev":
                g_center = np.mean(gain_errors)
            elif variation == "iqr":
                g_center = np.median(gain_errors)
            realization_centers[realization] = g_center

            g_offset_lo = np.min([np.min(g_center.real - g_var.real), np.min(g_center.imag - g_var.imag)])
            g_offset_hi = np.max([np.max(g_center.real + g_var.real), np.max(g_center.imag + g_var.imag)])
            g_boundary = np.max([np.abs(g_offset_lo), np.abs(g_offset_hi)])

            # plot scatter of realizations for this antenna scaled to "var" (stddev/IQR)
            ax[realization,0].scatter(gain_errors.real-1, gain_errors.imag)
            ax[realization,0].set_title(f"Gain Scatter for Realization {realization} (Lim: Var) Powell", fontsize="7.5")
            ax[realization,0].set_xlabel("Real - 1", fontsize = "7.5")
            ax[realization,0].set_ylabel("Imag", fontsize = "7.5")
            ax[realization,0].plot(g_center.real-1, g_center.imag, 'bx')
            ax[realization,0].plot(0,0,'rx')
            ax[realization,0].add_patch(plt.Circle((g_center.real-1, g_center.imag), radius=g_var, fill=False, color="black"))
            ax[realization,0].set_xlim(g_center.real-1-g_boundary, g_center.real-1+g_boundary)
            ax[realization,0].set_ylim(g_center.imag-g_boundary, g_center.imag+g_boundary)

        for realization in range(num_realizations):

            gain_errors = gain_error_array[realization]

            # set constants
            if variation == "stddev":
                g_var = np.std(gain_errors-1)
            elif variation == "iqr":
                g_var_real = np.percentile(gain_errors.real-1, 75) - np.percentile(gain_errors.real-1, 25)
                g_var_imag = np.percentile(gain_errors.imag, 75) - np.percentile(gain_errors.imag, 25)
                g_var = np.sqrt(g_var_real**2 + g_var_imag**2)

            # calculate centers
            if variation == "stddev":
                g_center = np.mean(gain_errors)
            elif variation == "iqr":
                g_center = np.median(gain_errors)
            realization_centers[realization] = g_center

            # for plot limits for outlier plot (we want a square plot so same for real and imag)
            g_max_lo = np.max([np.min(g_center.real - gain_errors.real), np.min(g_center.imag - gain_errors.imag)])
            g_max_hi = np.max([np.max(g_center.real + gain_errors.real), np.max(g_center.imag + gain_errors.imag)])
            g_boundary = np.max([np.abs(g_max_lo), np.abs(g_max_hi)])

            # plot scatter of realizations for this antenna scaled to "max" (outliers)
            ax[realization,1].scatter(gain_errors.real-1, gain_errors.imag)
            ax[realization,1].set_title(f"Gain Scatter for Realization {realization} (Lim: Max) Powell", fontsize="7.5")
            ax[realization,1].set_xlabel("Real - 1", fontsize = "7.5")
            ax[realization,1].set_ylabel("Imag", fontsize = "7.5")
            ax[realization,1].plot(g_center.real-1, g_center.imag, 'bx')
            ax[realization,1].plot(0,0,'rx')
            ax[realization,1].add_patch(plt.Circle((g_center.real-1, g_center.imag), radius=g_var, fill=False, color="black"))
            ax[realization,1].set_xlim(g_center.real-1-1.2*g_boundary, g_center.real-1+1.2*g_boundary-1)
            ax[realization,1].set_ylim(g_center.imag-1.2*g_boundary, g_center.imag+1.2*g_boundary)
            
        plt.tight_layout()
        plt.savefig('images/gain_error_per_realization_' + str(Nants) + '_'
            + subprocess.check_output(['git','rev-parse','--short','HEAD']).decode('ascii').strip()
            + '.png',
            bbox_inches=0,)
        plt.close(fig)

    """
        Plots a scatter plot spatial array with errors denoted by colors
        (assumes spatial array of shape (N,2) with N being e.g. Nbls or Nants
        and 2 corresponding to x/y)
    """
    def plot_spatial_array_with_colored_errors(self,
                                               spatial_array : np.ndarray, 
                                               error_array : np.ndarray, 
                                               title : str, 
                                               xlabel : str, 
                                               ylabel : str, 
                                               filename : str,
                                               scaling_factor : int = 1,
                                               threshold_length : int = 50,
                                               upper_limit : int = None,
                                               lower_limit : int = None
        ) -> None:
        colors = error_array / np.max(np.abs(error_array))
        plt.scatter(spatial_array[:,0], spatial_array[:,1], c=colors, cmap='viridis')
        plt.title(f"{title}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if not upper_limit and not lower_limit:
            upper_limit = np.max([np.max(np.abs(spatial_array[:,0])), np.max(np.abs(spatial_array[:,1]))]) + 25
            lower_limit = -upper_limit
        plt.xlim(lower_limit, upper_limit)
        plt.ylim(lower_limit, upper_limit)
        plt.colorbar()
        params = (
            f'1/$\\sigma_e^2$ Scaling Factor {scaling_factor}\n'
            f'Threshold Length {threshold_length}\n'
        )
        bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.85)
        plt.text(150, 100, params, fontsize=9, bbox=bbox, horizontalalignment='right')
        plt.savefig('images/' + filename + '.png')
        plt.close()

    # plot model visibilities in uv plane
    def plot_visibilities_in_uv_plane(self, 
                                      threshold_length : int, 
                                      u_minus_m : np.ndarray[complex], 
                                      uv_arr : np.ndarray[complex], 
                                      scaling_factor : int = 1,
        ) -> None:
        self.plot_spatial_array_with_colored_errors(
            uv_arr,
            np.abs(u_minus_m),
            "|u-v_T| in uv plane",
            "u",
            "v",
            "u_minus_m_in_uv_plane_redfac" + str(scaling_factor),
            threshold_length=threshold_length,
            scaling_factor=scaling_factor,
        )
    
    # plot gain errors in position spacex
    def plot_gains_in_position_space(self, 
                                     threshold_length : int, 
                                     g_errors : np.ndarray[float], 
                                     ant_pos_arr : np.ndarray[float]
        ) -> None:
        self.plot_spatial_array_with_colored_errors(
            ant_pos_arr,
            g_errors,
            "g-(1,0) in north-east plane",
            "N",
            "E",
            "gain_error_in_spatial_plane",
            threshold_length=threshold_length,
            upper_limit=300,
            lower_limit=-300,
        )
    
    def variable_sigmas_plot_individual_and_diff(self, 
                                                 scaling_factor_sim : int = 1,
                                                 scaling_factor_cost : int = 1, 
                                                 threshold_length : int = 50,
                                                 weighting_function : str = "constant_weights",
                                                 sigma_m_0 : int = 1,
                                                 sigma_e_0 : int = None,
                                                 datafile : str = "data/tutorial_medium_onetime.uvfits"
        ) -> None:

        _,g1,u1,m,uv = calwrap.unified_calibration_wrapper(data=datafile,
                                                           model=datafile,
                                                           parallel=False,
                                                           verbose=False,
                                                           glim=None,
                                                           ulim=None,
                                                           antenna_gain_weights=None,
                                                           model_baseline_weights=None,
                                                           threshold_length=threshold_length,
                                                           weighting_function="step_down_weights",
                                                           power=2,
                                                           sigma_t_0=1,
                                                           sigma_m_0=sigma_m_0,
                                                           sigma_n_0=None,
                                                           sigma_e_0=sigma_e_0,
                                                           gain_realizations=1,
                                                           model_realizations=1,
                                                           scaling_factor_sim=0.5,
                                                           scaling_factor_cost=0.5)
        
        _,g2,u2,m,uv = calwrap.unified_calibration_wrapper(data=datafile,
                                                           model=datafile,
                                                           parallel=False,
                                                           verbose=False,
                                                           glim=None,
                                                           ulim=None,
                                                           antenna_gain_weights=None,
                                                           model_baseline_weights=None,
                                                           threshold_length=threshold_length,
                                                           weighting_function=weighting_function,
                                                           power=2,
                                                           sigma_t_0=1,
                                                           sigma_m_0=sigma_m_0,
                                                           sigma_n_0=None,
                                                           sigma_e_0=None,
                                                           gain_realizations=1,
                                                           model_realizations=1,
                                                           scaling_factor_sim=scaling_factor_sim,
                                                           scaling_factor_cost=scaling_factor_cost)

        title = "|u1-u2| in uv plane" + \
                f"\nWeighting Function: {self.format_var_name(weighting_function)}" 
        self.plot_spatial_array_with_colored_errors(
            uv,
            # np.abs(u1 - u2),
            np.abs(u1),
            title,
            "u (m)",
            "v (m)",
            f"mag_reduced_minus_unreduced_in_uv_plane_cof_{weighting_function}",
            threshold_length=threshold_length,
            scaling_factor=scaling_factor_sim,
        )
        
        uv_norm = np.linalg.norm(uv, axis=1)
        # inner_baselines = np.abs(u1-u2)[uv_norm < threshold_length]
        # outer_baselines = np.abs(u1-u2)[uv_norm >= threshold_length]
        inner_baselines = np.abs(u1)[uv_norm < threshold_length]
        outer_baselines = np.abs(u1)[uv_norm >= threshold_length]
        self.plot_histogram(main_array=outer_baselines,
                            extra_array=inner_baselines,
                            main_label="Outer Baselines",
                            extra_label="Inner Baselines",
                            title=f"Histogram |u1|\nWeighting Function: {weighting_function}",
                            # xlabel="|u1-u2|",
                            xlabel="|u1|",
                            ylabel="count",
                            filename="u1u2_hist",
                            main_num_bins=50,
                            extra_num_bins=50,
                            params = (
                                f'1/$\\sigma_m^2$ Scaling Factor {scaling_factor_cost}\n'
                                f'1/$\\sigma_e^2$ Scaling Factor {scaling_factor_sim}\n'
                                f'Threshold Length {threshold_length}\n'
                            ),
                            # xlim_hi=3.5,
                            # xlim_lo=0,
                            )
        
        self.plot_histogram(main_array=np.abs(g1),  #=np.abs(g1-g2)
                            title=f"Histogram |g1|\nWeighting Function: {weighting_function}",
                            # xlabel="|g1-g2|",
                            xlabel="|g1|",
                            ylabel="count",
                            filename="g1g2_hist",
                            main_num_bins=50,
                            extra_num_bins=50,
                            params = (
                                f'1/$\\sigma_m^2$ Scaling Factor {scaling_factor_cost}\n'
                                f'1/$\\sigma_e^2$ Scaling Factor {scaling_factor_sim}\n'
                                f'Threshold Length {threshold_length}\n'
                            ),
                            # xlim_hi=0.008,
                            # xlim_lo=0,
                            )
        
        self.plot_histogram2d(main_array=u2-m,
                              title=f"Histogram u-m\nWeighting Function: {weighting_function}",
                              xlabel="Real",
                              ylabel="Imag",
                              filename="u_err_hist2d",
                              params = (
                                f'1/$\\sigma_m^2$ Scaling Factor {scaling_factor_cost}\n'
                                f'1/$\\sigma_e^2$ Scaling Factor {scaling_factor_sim}\n'
                                f'Threshold Length {threshold_length}\n'
                            ),
                            )

        self.plot_histogram2d(main_array=g2-1,
                              title=f"Histogram g-1\nWeighting Function: {weighting_function}",
                              xlabel="Real - 1",
                              ylabel="Imag",
                              filename="g_err_hist2d",
                              params = (
                                f'1/$\\sigma_m^2$ Scaling Factor {scaling_factor_cost}\n'
                                f'1/$\\sigma_e^2$ Scaling Factor {scaling_factor_sim}\n'
                                f'Threshold Length {threshold_length}\n'
                            ),
                            )

    def plot_weights_per_baseline(self,
                                  uv_norm_array,
                                  weight_array,
                                  weighting_function,
                                  scaling_factor=1,
                                  threshold_length=50,
                                  sigma : str = "sigma_m",
                                  ylim : float = None
    ) -> None:
        plt.scatter(uv_norm_array, weight_array, marker="_")
        plt.title(f"Weight per baseline length\nWeighting Function: {self.format_var_name(weighting_function)}")
        plt.xlabel("Baseline length (m)")
        plt.ylabel(f"1 / $\\{sigma}^2$")
        if not ylim:
            ylim = np.max(weight_array) + 0.1
        plt.ylim(0, ylim)
        params = (
            f'1/$\\{sigma}^2$ Scaling Factor {scaling_factor}\n'
            f'Threshold Length {threshold_length}\n'
        )
        bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.7)
        plt.text(np.max(uv_norm_array), 0.1, params, fontsize=9, bbox=bbox, horizontalalignment='right')
        plt.savefig(f'calico/images/weights_per_baseline_{sigma}.png')
        plt.close()

    def plot_histogram(self,
                       main_array : np.ndarray[float],
                       title : str,
                       xlabel : str,
                       ylabel : str,
                       filename : str,
                       params : str,
                       extra_array : np.ndarray[float] = None,
                       extra_label="",
                       main_label="",
                       main_num_bins=50,
                       extra_num_bins=50,
                       xlim_hi: int = None,
                       xlim_lo: int = -1,
    ) -> None:
        if not xlim_hi:
            if extra_array is not None:
                xlim_hi = np.max([np.max(main_array), np.max(extra_array)]) * 1.1
            else:
                xlim_hi = np.max(main_array) * 1.1
        else:
            xlim_hi = xlim_hi * 1.1
        if xlim_lo == -1:
            if extra_array is not None:
                xlim_lo = np.minimum(np.min(main_array), np.min(extra_array)) * 1.1
            else:
                xlim_lo = np.min(main_array) * 1.1
        else:
            xlim_lo = xlim_lo*1.1
        main_step_size = (np.max(main_array) - np.min(main_array)) / main_num_bins
        main_hist_edges = np.arange(np.min(main_array) - main_step_size/2, np.max(main_array) + main_step_size/2, main_step_size)
        main_hist, _ = np.histogram(main_array, bins=main_hist_edges)
        plt.hist(main_hist_edges[:-1], weights=main_hist, bins=main_num_bins, label=main_label, alpha=0.9)
        if extra_array is not None:
            extra_step_size = (np.max(extra_array) - np.min(extra_array)) / extra_num_bins
            extra_hist_edges = np.arange(np.min(extra_array) - extra_step_size/2, np.max(extra_array) + extra_step_size/2, extra_step_size)
            extra_hist, _ = np.histogram(extra_array, bins=extra_hist_edges)
            plt.hist(extra_hist_edges[:-1], weights=extra_hist, bins=extra_num_bins, label=extra_label, alpha=0.8)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        print(f"xlim hi {xlim_hi} xlim lo {xlim_lo}")
        # plt.xlim(xlim_lo, xlim_hi)
        if extra_array is not None:
            plt.legend()
            text_x = max([np.max(main_array), max(extra_array)])
            text_y = max([max(main_hist), max(extra_hist)]) / 2
        else:
            text_x = np.max(main_array)
            text_y = max(main_hist) / 2
        bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.7)
        plt.text(text_x, text_y, params, fontsize=9, bbox=bbox, horizontalalignment='right')
        plt.savefig(f"images/{filename}.png")
        plt.close()

    def plot_histogram2d(self,
                         main_array : np.ndarray[float],
                         title : str,
                         xlabel : str,
                         ylabel : str,
                         filename : str,
                         params : str,
                         main_label : str = "",
                         main_num_bins : int = 50,
                         xlim_hi: int | float = None,
                         xlim_lo: int | float = None,
                         ylim_lo: int | float = None,
                         ylim_hi: int | float = None,
                         variation : str = "stddev",
                         radius : float = None,
                         xlim : int | float = None,
                         ylim : int | float = None,
                         ax : plt.axes = None
    ) -> None:
        
        if variation == "stddev":
            main_var = np.std(main_array)
        elif variation == "iqr":
            main_var_real = np.percentile(main_array.real, 75) - np.percentile(main_array.real, 25)
            main_var_imag = np.percentile(main_array.imag, 75) - np.percentile(main_array.imag, 25)
            main_var = np.sqrt(main_var_real**2 + main_var_imag**2)
        main_boundary = 1.5 * main_var

        if radius is None:
            radius = np.std(main_array)

        # set bin sizes
        main_step = main_var / 7.5
        main_bins = np.arange(-main_boundary, main_boundary + main_step, main_step)

        # calculate centers
        if variation == "stddev":
            main_center = np.mean(main_array)
        elif variation == "iqr":
            main_center = np.median(main_array)

        main_hist, main_imag, main_real = np.histogram2d(main_array.real, main_array.imag, bins=main_bins, density=True)

        if ax is None:
            ax = plt.axes()
            # NOTE: How to do subplots with this?

        ax.set_axis_on()
        im = ax.pcolormesh(main_real, main_imag, main_hist, cmap="inferno")
        ax.add_patch(plt.Circle((main_center.real, main_center.imag), radius=main_var, fill=False, color="white"))  # std dev of gain errors
        ax.add_patch(plt.Circle((0,0), radius=radius, fill=False, color="white", linestyle="dashed"))  # expected variation
        ax.plot(main_center.real, main_center.imag, 'wx')
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        if xlim_lo is not None and xlim_hi is not None:
            ax.set_xlim(xlim_lo, xlim_hi)
        if ylim is not None and ylim_hi is not None:
            ax.set_ylim(ylim_lo, ylim_hi)
        ax.set_title(title, fontsize="15")
        ax.tick_params(labelbottom=True, labelleft=True)
        bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.7)
        text_x = np.max(main_real)
        text_y = np.max(main_imag) / 2
        plt.text(text_x, text_y, params, fontsize=9, bbox=bbox, horizontalalignment='right')
        plt.colorbar(im)
        plt.savefig(f"images/{filename}.png")
        plt.close()
    """
        Getters and Setters
    """
    # params_init_flattened
    def get_params_init_flattened(
        self
    ) -> np.ndarray[float]:
        return self.params_init_flattened
    
    def set_params_init_flattened(
        self, 
        val : np.ndarray[float]
    ) -> None:
        self.params_init_flattened = val

    # caldata_obj
    def get_caldata_obj(
        self
    ) -> object:
        return self.caldata_obj
    
    def set_caldata_obj(
        self,
        val : object,
    ) -> None:
        self.caldata_obj = val

    # Nants_unflagged
    def get_Nants_unflagged(
        self
    ) -> int:
        return self.Nants_unflagged
    
    def set_Nants_unflagged(
        self, 
        val : int,
    ) -> None:
        self.Nants_unflagged = val

    # freq_ind
    def get_freq_ind(
        self
    ) -> int:
        return self.freq_ind
    
    def set_freq_ind(self,
                     val : int,
    ) -> None:
        self.freq_ind = val

    # vis_pol_ind
    def get_vis_pol_ind(
        self
    ) -> int:
        return self.vis_pol_ind
    
    def set_vis_pol_ind(
        self, 
        val : int,
    ) -> None:
        self.vis_pol_ind = val

def build_3d_scatter_plot(
    x_array           : np.ndarray,
    y_array           : np.ndarray,
    z_array           : np.ndarray,
    z_array_2         : np.ndarray = None, 
    second_plot       : bool = False,
    show_plot         : bool = False,
    plot_title        : str = "",
    plot_xlabel       : str = "",
    plot_ylabel       : str = "",
    plot_zlabel       : str = "",
    xlim_hi           : int | float = 1,
    xlim_lo           : int | float = 0,
    ylim_hi           : int | float = 1,
    ylim_lo           : int | float = 0,
    zlim_hi           : int | float = 1,
    zlim_lo           : int | float = 0,
    first_plot_label  : str = "",
    second_plot_label : str = "",
    filename          : str = "",
    suffix            : str = "",
    metadata          : dict = None,
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x_array, y_array, z_array, label=first_plot_label)
    if second_plot:
        ax.scatter(x_array, y_array, z_array_2, label=second_plot_label)
    xx, yy = np.meshgrid(range(int(np.min(x_array) - 1), int(np.max(x_array) + 1)), 
                        range(int(np.min(y_array) - 1), int(np.max(y_array) + 1)))
    zz = 0 * np.ones_like(xx)
    ax.plot_surface(xx,yy,zz, alpha=0.2)
    ax.set_title(plot_title)
    ax.set_xlabel(plot_xlabel)
    ax.set_ylabel(plot_ylabel)
    ax.set_zlabel(plot_zlabel)
    ax.set_xlim(xlim_lo, xlim_hi)
    ax.set_ylim(ylim_lo, ylim_hi)
    ax.set_zlim(zlim_lo, zlim_hi)
    if len(first_plot_label) > 0:
        plt.legend()
    if show_plot:
        plt.show()
    plt.savefig(filename, bbox_inches=0, metadata=metadata)
    plt.close()

    metadata_str = "\n".join([f"{key}: {val}" for key, val in metadata.items()])
    img = Image.open(filename)
    img_metadata = PngImagePlugin.PngInfo()
    img_metadata.add_text("Description", f"Project Settings and Info:\n{metadata_str}")
    img.save(filename, pnginfo=img_metadata)

def plot_3d_data_as_2d_hist(
    x_array       : np.ndarray,
    y_array       : np.ndarray,
    z_array       : np.ndarray,
    num_x_vals    : int,
    num_y_vals    : int,
    # top arrays
    x_array_2     : np.ndarray = None,
    x_array_3     : np.ndarray = None,
    # left side array
    z_array_2     : np.ndarray = None,
    plot_title    : str = "",
    plot_xlabel   : str = "",
    plot_xlabel_2 : str = "",
    plot_xlabel_3 : str = "",
    plot_ylabel   : str = "",
    plot_xlim_h   : int | float = None,
    plot_xlim_l   : int | float = None,
    plot_ylim_h   : int | float = None,
    plot_ylim_l   : int | float = None,
    plot_vmax     : int | float = 1, 
    plot_vmin     : int | float = 0,     
    filename      : str = "",   
    plot_cmap     : str = "viridis",  
    log_cmap      : bool = False, 
    cmap_label    : str = "",
    suffix        : str = "",
    metadata      : dict = None,
    angle         : int | float = 0,
) -> None:
    from scipy.interpolate import griddata
    from scipy import ndimage
    from matplotlib import colors
    z_grid = np.asarray(z_array).reshape((num_x_vals, num_y_vals))
    np.set_printoptions(precision=4, suppress=True, linewidth=500)
    print(f"Original z grid\n\n{z_grid}\n\n")
    # rotate grid
    z_grid_rot = ndimage.rotate(z_grid, angle=angle)
    print(f"Rotated z grid\n\n{z_grid_rot}\n\n")
    if z_array_2 is None:
        # NOTE: Move above code here?
        ...
    else:
        # merging left and right arrays
        ...
    if log_cmap:
        max_abs = np.max(np.abs(z_grid_rot))
        im = plt.imshow(
            z_grid_rot,
            cmap=plot_cmap,
            extent=[np.min(x_array),
                    np.max(x_array),
                    np.min(y_array),
                    np.max(y_array)],
            aspect='equal',
            origin='lower',
            norm=colors.SymLogNorm(10**-4, 
                                   vmin=-1*max_abs,
                                   vmax=max_abs,),
        )
    else:
        im = plt.imshow(
            z_grid_rot,
            cmap=plot_cmap,
            vmax=plot_vmax,
            vmin=plot_vmin,
            extent=[np.min(x_array),
                    np.max(x_array),
                    np.min(y_array),
                    np.max(y_array)],
            aspect='equal',
            origin='lower',
        )
    plt.colorbar(im, label=cmap_label)
    plt.title(plot_title)
    plt.xlabel(plot_xlabel)
    if x_array_2 is not None:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(x_array)
        x2_labels = [f"{str(int(val))}" for val in x_array_2]
        ax2.set_xticklabels(x2_labels)
        ax2.set_xlabel(plot_xlabel_2)
        if x_array_3 is not None:
            ax3 = ax.twiny()
            ax3.set_xlim(ax.get_xlim())
            ax3.set_xticks(x_array)
            x3_labels = [f"{str(int(val))}" for val in x_array_3]
            ax3.set_xticklabels(x3_labels)
            ax2.set_xlabel(plot_xlabel_3)  # should have been ax3 but this works now
            ax3.spines["top"].set_position(("axes", 1.15))
    ax.set_ylabel(plot_ylabel)
    plt.xlim(plot_xlim_l, plot_xlim_h)
    plt.ylim(plot_ylim_l, plot_ylim_h)
    # plt.grid(visible=True, axis='both', which='major', color='black', linewidth=1)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', metadata=metadata,)
    plt.close()

    metadata_str = "\n".join([f"{key}: {val}" for key, val in metadata.items()])
    img = Image.open(filename)
    img_metadata = PngImagePlugin.PngInfo()
    img_metadata.add_text("Description", f"Project Settings and Info:\n{metadata_str}")
    img.save(filename, pnginfo=img_metadata)