from scipy.differentiate import jacobian
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from calico import cost_function_calculations, calibration_optimization as cal_opt, calibration_wrappers as calwrap
import time

class DevTools:

    def __init__(self):
        params_init_flattened=None
        caldata_obj=None
        Nants_unflagged=None
        freq_ind=None
        vis_pol_ind=None


    def compare_analytic_and_numeric_jacobians(self):
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
        # scatter plot of numeric step vs analytic step
        # plt.scatter(analytic_step.real, analytic_step.imag, c="blue")
        # plt.scatter(numeric_step.real, numeric_step.imag, c="orange")
        # plt.title("Analytic Step (Blue) and Numeric Step (Orange)")
        # plt.xlabel("Real")
        # plt.ylabel("Imag")
        # plt.savefig('images/' + 'analytic-vs-numeric_'
        #          + subprocess.check_output(['git','rev-parse','--short','HEAD']).decode('ascii').strip()
        #          + '.png',
        #          bbox_inches=0,)
    
    def display_where_large_real_or_imag(
        self,
        jac_analytic,
        jac_numeric,
    ):
        jac_error, jac_frac, where_large = self.calc_error_vals(jac_numeric, jac_analytic)
        n_vals = len(where_large[0])
        # find large errors
        analytic_vals = jac_analytic[where_large]
        numeric_vals = jac_numeric[where_large]
        param_vals = self.params_init_flattened[where_large]
        # print display
        np.set_printoptions(precision=4)
        print("***WHERE ERROR IS LARGE***")
        part = lambda x : "Real" if x == 0 else "Imag"
        for val in range(n_vals):
            print("Value",val+1)
            print(f"aj: {analytic_vals[val]} nj: {numeric_vals[val]} val: {param_vals[val]} bl_ind: {part((where_large[0][val] - 2*self.Nants_unflagged) % 2)}")
        # plot display
        # params = [x for x in range(2 * self.Nants_unflagged + 2 * len(self.caldata_obj.bl_inds))]
        # plt.scatter(params, jac_error)
        # plt.scatter(params, jac_frac, alpha=0.3)
        # plt.title("Fractional error vs Numerical fraction")
        # plt.xlabel("Params (alternating real/imag)")
        # plt.savefig('images/' + 'where_large_'
        #          + subprocess.check_output(['git','rev-parse','--short','HEAD']).decode('ascii').strip()
        #          + '.png',
        #          bbox_inches=0,)

    def calc_error_vals(
        self,
        numeric_jac,
        analytic_jac,
    ):
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
        analytic_jac_result, 
        numeric_jac_result,
    ):
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

    def get_starting_cost_func_val(self):
        print("***STARTING FUNCTION VALUE***", 
        cal_opt.cost_unical_wrapper(self.params_init_flattened,
                                    self.caldata_obj,
                                    self.caldata_obj.ant_inds,
                                    self.Nants_unflagged,
                                    self.caldata_obj.bl_inds + self.Nants_unflagged,
                                    self.freq_ind,
                                    self.vis_pol_ind,))
        
    def cost_for_numeric_jac(self, params_array):
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
    
    def cost_vectorized(self, params_array):
        return np.apply_along_axis(self.cost_for_numeric_jac, axis=0, arr=params_array)
    
    def complex_trajectory_plot(
        self,
        starting_complex_point,
        complex_step,
        n_trajectories,
        title,
        xlims,
        ylims,
        filename_prefix,
    ):
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
        ax.set_xlabel("Real")
        ax.set_ylabel("Imag")
        ax.set_title(title)
        ax.set_xlim(xlims[0],xlims[1])
        ax.set_ylim(ylims[0],ylims[1])
        fig.savefig('images/' + filename_prefix
                 + subprocess.check_output(['git','rev-parse','--short','HEAD']).decode('ascii').strip()
                 + '.png',
                 bbox_inches=0,)
        plt.close()

    def plot_change_in_gain_and_model_params(
        self, 
        gains_array, 
        models_array,
        type="trajectory",
        glim=(-0.013, 0.013),
        ulim=(-1.5,1.5),
        error_type="thermal",
        stddev_thermal="1",
        stddev_model="1",
    ):
        # print("***FIT TESTS***")
        # print("\tGains Error, Min -", np.min(np.abs(self.caldata_obj.gains - 1)))
        # print("\tGains Error, Max -", np.max(np.abs(self.caldata_obj.gains - 1)))
        # print("\t|u-m|, Min -", np.min(np.abs(self.caldata_obj.fit_vis - self.caldata_obj.model_visibilities)))
        # print("\t|u-m|, Max -", np.max(np.abs(self.caldata_obj.fit_vis - self.caldata_obj.model_visibilities)))
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
            # print("***GAINS STD DEV***", np.std(gains_array))
            # print("***MODELS STD DEV***", np.std(models_array))
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

    def plot_many_realizations(self, variation="stddev", max_realizations=100):
        # NOTE: May be good to read this in as a data file
        sigma_combinations = [
            # only thermal noise (100 realizations)
            {
                "sigma_t": 1, 
                "sigma_n": None, 
                "sigma_m": 1, 
                "sigma_e": 0, 
                "gain_realizations": max_realizations, 
                "model_realizations": 1
            },
            {
                "sigma_t": 10,
                "sigma_n": None, 
                "sigma_m": 1, 
                "sigma_e": 0, 
                "gain_realizations": max_realizations, 
                "model_realizations": 1
            },
            {
                "sigma_t": 0.1, 
                "sigma_n": None, 
                "sigma_m": 1, 
                "sigma_e": 0, 
                "gain_realizations": max_realizations, 
                "model_realizations": 1
            },
            {
                "sigma_t": 1, 
                "sigma_n": None, 
                "sigma_m": 10, 
                "sigma_e": 0, 
                "gain_realizations": max_realizations, 
                "model_realizations": 1
            },
            {
                "sigma_t": 10, 
                "sigma_n": None, 
                "sigma_m": 10, 
                "sigma_e": 0, 
                "gain_realizations": max_realizations, 
                "model_realizations": 1
            },
            {
                "sigma_t": 0.1, 
                "sigma_n": None, 
                "sigma_m": 10, 
                "sigma_e": 0, 
                "gain_realizations": max_realizations, 
                "model_realizations": 1
            },
            {
                "sigma_t": 1, 
                "sigma_n": None, 
                "sigma_m": 0.1, 
                "sigma_e": 0, 
                "gain_realizations": max_realizations, 
                "model_realizations": 1
            },
            {
                "sigma_t": 10,
                "sigma_n": None, 
                "sigma_m": 0.1, 
                "sigma_e": 0, 
                "gain_realizations": max_realizations, 
                "model_realizations": 1
            },
            {
                "sigma_t": 0.1,
                "sigma_n": None,
                "sigma_m": 0.1, 
                "sigma_e": 0, 
                "gain_realizations": max_realizations,
                "model_realizations": 1
            },
            # thermal noise (100 realizations) and model error (1 realization)
            {
                "sigma_t": 1, 
                "sigma_n": None, 
                "sigma_m": 1, 
                "sigma_e": None, 
                "gain_realizations": max_realizations, 
                "model_realizations": 1
            },
            {
                "sigma_t": 1, 
                "sigma_n": None, 
                "sigma_m": 10, 
                "sigma_e": None, 
                "gain_realizations": max_realizations, 
                "model_realizations": 1
            },
            {
                "sigma_t": 1, 
                "sigma_n": None, 
                "sigma_m": 0.1, 
                "sigma_e": None, 
                "gain_realizations": max_realizations, 
                "model_realizations": 1
            },
            {
                "sigma_t": 0.1, 
                "sigma_n": None, 
                "sigma_m": 1, 
                "sigma_e": None, 
                "gain_realizations": max_realizations, 
                "model_realizations": 1
            },
            {
                "sigma_t": 0.1, 
                "sigma_n": None, 
                "sigma_m": 10, 
                "sigma_e": None, 
                "gain_realizations": max_realizations, 
                "model_realizations": 1
            },
            {
                "sigma_t": 0.1,
                "sigma_n": None, 
                "sigma_m": 0.1, 
                "sigma_e": None, 
                "gain_realizations": max_realizations, 
                "model_realizations": 1
            },
            {
                "sigma_t": 10, 
                "sigma_n": None, 
                "sigma_m": 1, 
                "sigma_e": None, 
                "gain_realizations": max_realizations, 
                "model_realizations": 1
            },
            {
                "sigma_t": 10, 
                "sigma_n": None, 
                "sigma_m": 10, 
                "sigma_e": None, 
                "gain_realizations": max_realizations, 
                "model_realizations": 1
            },
            {
                "sigma_t": 10, 
                "sigma_n": None, 
                "sigma_m": 0.1, 
                "sigma_e": None, 
                "gain_realizations": max_realizations, 
                "model_realizations": 1
            },
            # thermal noise (100 realizations) and model error (100 realizations)
            {
                "sigma_t": 1, 
                "sigma_n": None, 
                "sigma_m": 1, 
                "sigma_e": None, 
                "gain_realizations": max_realizations, 
                "model_realizations": max_realizations
            },
            {
                "sigma_t": 1, 
                "sigma_n": None, 
                "sigma_m": 10, 
                "sigma_e": None, 
                "gain_realizations": max_realizations, 
                "model_realizations": max_realizations
            },
            {
                "sigma_t": 1, 
                "sigma_n": None, 
                "sigma_m": 0.1, 
                "sigma_e": None, 
                "gain_realizations": max_realizations, 
                "model_realizations": max_realizations
            },
            {
                "sigma_t": 0.1, 
                "sigma_n": None, 
                "sigma_m": 1, 
                "sigma_e": None, 
                "gain_realizations": max_realizations, 
                "model_realizations": max_realizations
            },
            {
                "sigma_t": 0.1, 
                "sigma_n": None, 
                "sigma_m": 10, 
                "sigma_e": None, 
                "gain_realizations": max_realizations, 
                "model_realizations": max_realizations
            },
            {
                "sigma_t": 0.1, 
                "sigma_n": None, 
                "sigma_m": 0.1, 
                "sigma_e": None, 
                "gain_realizations": max_realizations, 
                "model_realizations": max_realizations
            },
            {
                "sigma_t": 10, 
                "sigma_n": None, 
                "sigma_m": 1, 
                "sigma_e": None, 
                "gain_realizations": max_realizations, 
                "model_realizations": max_realizations
            },
            {
                "sigma_t": 10, 
                "sigma_n": None, 
                "sigma_m": 10, 
                "sigma_e": None, 
                "gain_realizations": max_realizations, 
                "model_realizations": max_realizations
            },
            {
                "sigma_t": 10, 
                "sigma_n": None, 
                "sigma_m": 0.1, 
                "sigma_e": None, 
                "gain_realizations": max_realizations, 
                "model_realizations": max_realizations
            },
            # thermal noise (1 realization) and model error (100 realizations)
            {
                "sigma_t": 1, 
                "sigma_n": None, 
                "sigma_m": 1, 
                "sigma_e": None, 
                "gain_realizations": 1, 
                "model_realizations": max_realizations
            },
            {
                "sigma_t": 1, 
                "sigma_n": None, 
                "sigma_m": 10, 
                "sigma_e": None, 
                "gain_realizations": 1, 
                "model_realizations": max_realizations
            },
            {
                "sigma_t": 1, 
                "sigma_n": None, 
                "sigma_m": 0.1, 
                "sigma_e": None, 
                "gain_realizations": 1, 
                "model_realizations": max_realizations
            },
            {
                "sigma_t": 0.1, 
                "sigma_n": None, 
                "sigma_m": 1, 
                "sigma_e": None, 
                "gain_realizations": 1, 
                "model_realizations": max_realizations
            },
            {
                "sigma_t": 0.1, 
                "sigma_n": None, 
                "sigma_m": 10, 
                "sigma_e": None, 
                "gain_realizations": 1, 
                "model_realizations": max_realizations
            },
            {
                "sigma_t": 0.1,
                "sigma_n": None, 
                "sigma_m": 0.1, 
                "sigma_e": None, 
                "gain_realizations": 1, 
                "model_realizations": max_realizations
            },
            {
                "sigma_t": 10, 
                "sigma_n": None, 
                "sigma_m": 1, 
                "sigma_e": None, 
                "gain_realizations": 1, 
                "model_realizations": max_realizations
            },
            {
                "sigma_t": 10,
                "sigma_n": None, 
                "sigma_m": 10, 
                "sigma_e": None, 
                "gain_realizations": 1, 
                "model_realizations": max_realizations
            },
            {
                "sigma_t": 10, 
                "sigma_n": None, 
                "sigma_m": 0.1, 
                "sigma_e": None, 
                "gain_realizations": 1, 
                "model_realizations": max_realizations
            },
            # only model noise (100 realizations)
            {
                "sigma_t": 1, 
                "sigma_n": 0, 
                "sigma_m": 1, 
                "sigma_e": None, 
                "gain_realizations": 1, 
                "model_realizations": max_realizations
            },
            {
                "sigma_t": 1, 
                "sigma_n": 0, 
                "sigma_m": 10, 
                "sigma_e": None, 
                "gain_realizations": 1, 
                "model_realizations": max_realizations
            },
            {
                "sigma_t": 1, 
                "sigma_n": 0, 
                "sigma_m": 0.1, 
                "sigma_e": None, 
                "gain_realizations": 1, 
                "model_realizations": max_realizations
            },
            {
                "sigma_t": 10, 
                "sigma_n": 0, 
                "sigma_m": 1, 
                "sigma_e": None,
                "gain_realizations": 1,
                "model_realizations": max_realizations
            },
            {
                "sigma_t": 10, 
                "sigma_n": 0, 
                "sigma_m": 10, 
                "sigma_e": None, 
                "gain_realizations": 1, 
                "model_realizations": max_realizations
            },
            {
                "sigma_t": 10, 
                "sigma_n": 0, 
                "sigma_m": 0.1, 
                "sigma_e": None, 
                "gain_realizations": 1, 
                "model_realizations": max_realizations
            },
            {
                "sigma_t": 0.1, 
                "sigma_n": 0, 
                "sigma_m": 1, 
                "sigma_e": None, 
                "gain_realizations": 1, 
                "model_realizations": max_realizations
            },
            {
                "sigma_t": 0.1, 
                "sigma_n": 0, 
                "sigma_m": 10, 
                "sigma_e": None, 
                "gain_realizations": 1, 
                "model_realizations": max_realizations
            },
            {
                "sigma_t": 0.1,
                "sigma_n": 0, 
                "sigma_m": 0.1, 
                "sigma_e": None, 
                "gain_realizations": 1, 
                "model_realizations": max_realizations
            },
        ]

        number_sigma_combos = len(sigma_combinations)
        i = 0
        fig, ax = plt.subplots(number_sigma_combos, 8, figsize=(27,3*number_sigma_combos), squeeze=False, sharex=False, sharey=False)
        for sigma_dict in sigma_combinations:
            calibration_start_time = time.perf_counter()
            uvc, g_arr, u_arr = calwrap.unified_calibration_wrapper(
                'data/tutorial_medium_onetime.uvfits',
                'data/tutorial_medium_onetime.uvfits',
                parallel=False,
                verbose=False,
                sigma_t=sigma_dict["sigma_t"],
                sigma_m=sigma_dict["sigma_m"],
                sigma_n=sigma_dict["sigma_n"],
                sigma_e=sigma_dict["sigma_e"],
                gain_realizations=sigma_dict["gain_realizations"],
                model_realizations=sigma_dict["model_realizations"],
            )
            total_time = (time.perf_counter() - calibration_start_time) / 60

            # set constants
            if variation == "stddev":
                g_var = np.std(g_arr)
                u_var = np.std(u_arr)
            elif variation == "iqr":
                g_var_real = np.percentile(g_arr.real, 75) - np.percentile(g_arr.real, 25)
                u_var_real = np.percentile(u_arr.real, 75) - np.percentile(u_arr.real, 25)
                g_var_imag = np.percentile(g_arr.imag, 75) - np.percentile(g_arr.imag, 25)
                u_var_imag = np.percentile(u_arr.imag, 75) - np.percentile(u_arr.imag, 25)
                g_var = np.sqrt(g_var_real**2 + g_var_imag**2)
                u_var = np.sqrt(u_var_real**2 + u_var_imag**2)
            g_boundary = 1.5 * g_var
            u_boundary = 1.5 * u_var

            # set bin sizes
            g_step = g_var / 7.5
            u_step = u_var / 7.5
            g_bins = np.arange(-g_boundary, g_boundary + g_step, g_step)
            u_bins = np.arange(-u_boundary, u_boundary + u_step, u_step)

            # calculate centers
            if variation == "stddev":
                g_center = np.mean(g_arr)
                u_center = np.mean(u_arr)
            elif variation == "iqr":
                g_center = np.median(g_arr)
                u_center = np.median(u_arr)

            # for plot limits for outlier plot (we want a square plot so same for real and imag)
            g_max_lo = np.max([np.min(g_center.real - u_arr.real), np.min(g_center.imag - g_arr.imag)])
            g_max_hi = np.max([np.max(g_center.real + u_arr.real), np.max(g_center.imag + g_arr.imag)])
            # g_boundary = np.max([np.abs(g_max_lo), np.abs(g_max_hi)])
            u_max_lo = np.max([np.min(u_center.real - u_arr.real), np.min(u_center.imag - u_arr.imag)])
            u_max_hi = np.max([np.max(u_center.real + u_arr.real), np.max(u_center.imag + u_arr.imag)])
            # u_boundary = np.max([np.abs(u_max_lo), np.abs(u_max_hi)])

            # get histograms
            gains_hist, gains_imag, gains_real = np.histogram2d(g_arr.real-1, g_arr.imag, bins=g_bins, density=True)
            models_hist, models_imag, models_real = np.histogram2d(u_arr.real, u_arr.imag, bins=u_bins, density=True)

            # render LaTeX math
            plt.rcParams['text.usetex'] = True

            # text output run types
            ax[i,0].set_axis_off()
            ax[i,0].text(0.4,0.7,rf"$\sigma_t$: {sigma_dict['sigma_t']}", fontsize="20")
            ax[i,0].set_axis_off()
            if sigma_dict["sigma_n"] == None:
                ax[i,0].text(0.4,0.3,rf"$\sigma_n$: {sigma_dict['sigma_t']}", fontsize="20")
            else:
                ax[i,0].text(0.4,0.3,r"$\sigma_n$: 0", fontsize="20")
            ax[i,1].set_axis_off()
            ax[i,1].text(0.4,0.7,rf"$\sigma_m$: {sigma_dict['sigma_m']}", fontsize="20")
            ax[i,1].set_axis_off()
            if sigma_dict["sigma_e"] == None:
                ax[i,1].text(0.4,0.3,rf"$\sigma_e$: {sigma_dict['sigma_m']}",fontsize="20")
            else:
                ax[i,1].text(0.4,0.3,r"$\sigma_e$: 0", fontsize="20")

            # text output how many realizations
            ax[i,2].set_axis_off()
            if sigma_dict["sigma_n"] == 0:
                ax[i,2].text(0.0,0.7,"Thermal Rolls: 0", fontsize="20")
            else:
                ax[i,2].text(0.0,0.7,f"Thermal Rolls: {sigma_dict['gain_realizations']}", fontsize="20")
            ax[i,3].set_axis_off()
            if sigma_dict["sigma_e"] == 0:
                ax[i,2].text(0.0,0.3,"Model Rolls: 0", fontsize="20")
            else:
                ax[i,2].text(0.0,0.3,f"Model Rolls: {sigma_dict['model_realizations']}", fontsize="20")

            # initial gains
            ax[i,3].set_axis_on()
            ax[i,3].pcolormesh(gains_real, gains_imag, gains_hist, cmap="inferno")
            ax[i,3].add_patch(plt.Circle((g_center.real-1,g_center.imag), radius=g_var, fill=False, color="white"))  # std dev of gain errors
            ax[i,3].add_patch(plt.Circle((0,0), radius=sigma_dict['sigma_t'], fill=False, color="white", linestyle="dashed"))  # expected variation
            ax[i,3].plot(g_center.real - 1, g_center.imag, 'wx')
            ax[i,3].set_ylabel("Imag")
            ax[i,3].set_xlabel("Real - 1")
            ax[i,3].set_xlim(g_center.real-1-g_boundary, g_center.real-1+g_boundary)
            ax[i,3].set_ylim(g_center.imag-g_boundary, g_center.imag+g_boundary)
            ax[i,3].set_title(f"Final Gains Error", fontsize="15")
            ax[i,3].tick_params(labelbottom=True, labelleft=True)

            # standard deviation for gains
            ax[i,4].set_axis_off()
            if variation == "stddev":
                ax[i,4].text(0.2,0.8,rf"$\sigma_g$: {(g_var / sigma_dict['sigma_t']):.2f} $\sigma_t$", fontsize="15")
                ax[i,4].text(0.2,0.6,rf"$\sigma^2_g$: {(g_var**2 / sigma_dict['sigma_t']**2):.2f} $\sigma_t^2$", fontsize="15")
                ax[i,4].text(0.2,0.4,f"Max g err: {np.max(np.abs(g_arr)):.2f}", fontsize="15")
                ax[i,4].text(0.2,0.2,f"Min g err: {np.min(np.abs(g_arr)):.2f}", fontsize="15")
            elif variation == "iqr":
                ax[i,4].text(0.2,0.7,rf"IQR: {(g_var / sigma_dict['sigma_t']):.2f} $\sigma_t$", fontsize="17")
                ax[i,4].text(0.2,0.5,f"Max g err: {np.max(np.abs(g_arr)):.2f}", fontsize="15")
                ax[i,4].text(0.2,0.3,f"Min g err: {np.min(np.abs(g_arr)):.2f}", fontsize="15")

            # initial models
            ax[i,5].pcolormesh(models_real, models_imag, models_hist, cmap="inferno")
            ax[i,5].add_patch(plt.Circle((u_center.real,u_center.imag), radius=u_var, fill=False, color="white"))  # std dev of u-m errors
            ax[i,5].add_patch(plt.Circle((0,0), radius=sigma_dict['sigma_m'], fill=False, color="white", linestyle="dashed"))
            ax[i,5].plot(u_center.real, u_center.imag, 'wx')
            ax[i,5].set_ylabel("Imag")
            ax[i,5].set_xlabel("Real")
            ax[i,5].set_xlim(u_center.real-u_boundary, u_center.real+u_boundary)
            ax[i,5].set_ylim(u_center.imag-u_boundary, u_center.imag+u_boundary)
            ax[i,5].set_title(f"Final u-m Error", fontsize="15")

            # standard deviation for models
            ax[i,6].set_axis_off()
            if variation == "stddev":
                ax[i,6].text(0.2,0.8,rf"$\sigma_u$: {(u_var / sigma_dict['sigma_m']):.2f} $\sigma_m$", fontsize="15")
                ax[i,6].text(0.2,0.6,rf"$\sigma_u^2$: {(u_var**2 / sigma_dict['sigma_m']**2):.2f} $\sigma_m^2$", fontsize="15")
                ax[i,6].text(0.2,0.4,f"Max u-m: {np.max(np.abs(u_arr)):.2f}", fontsize="15")
                ax[i,6].text(0.2,0.2,f"Min u-m: {np.min(np.abs(u_arr)):.2f}", fontsize="15")
            elif variation == "iqr":
                ax[i,6].text(0.2,0.7,rf"IQR: {(u_var / sigma_dict['sigma_m']):.2f} $\sigma_m$", fontsize="17")
                ax[i,6].text(0.2,0.5,f"Max u-m: {np.max(np.abs(u_arr)):.2f}", fontsize="17")
                ax[i,6].text(0.2,0.3,f"Min u-m: {np.min(np.abs(u_arr)):.2f}", fontsize="17")

            # time to do realizations
            ax[i,7].set_axis_off()
            ax[i,7].text(0.0,0.6,f"Time to Complete", fontsize="15")
            ax[i,7].text(0.0,0.4,rf"{total_time:.2f} s", fontsize="15")
            
            fig.tight_layout()
            i += 1
        plt.savefig('images/' + str(max_realizations) + '-realizations_' + variation + '_'
                + subprocess.check_output(['git','rev-parse','--short','HEAD']).decode('ascii').strip()
                + '.png',
                bbox_inches=0,)

    # plot gain errors across realizations for one antenna at a time at two
    # scales: one set to "var" (stddev/IQR), the other to "max" (outliers)
    def plot_gains_one_ant_same_noise_and_error(self, 
                                                num_realizations=20, 
                                                sigma=1,
                                                variation="stddev", 
                                                plot_type="variation",
                                                data_path='data/tutorial_medium_onetime.uvfits',
                                                weights_threshold=50,
                                                cutoff_function="constant_weights"):

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

    def plot_gain_error_per_realization(self, gain_error_array, variation="stddev", plot_type="variation"):
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

    # scatter plot spatial array with errors denoted by colors
    # (assumes spatial array of shape (N,2) with N being e.g. Nbls or Nants
    #  and 2 corresponding to x/y)
    def plot_spatial_array_with_colored_errors(self, 
                                               spatial_array, 
                                               error_array, 
                                               title, 
                                               xlabel, 
                                               ylabel, 
                                               filename,
                                               upper_limit=None,
                                               lower_limit=None):
        colors = error_array * 100 / np.abs(error_array)
        plt.scatter(spatial_array[:,0], spatial_array[:,1], c=colors, cmap='viridis')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if not upper_limit and not lower_limit:
            upper_limit = np.max([np.max(np.abs(spatial_array[:,0])), np.max(np.abs(spatial_array[:,1]))]) + 25
            lower_limit = -upper_limit
        plt.xlim(lower_limit, upper_limit)
        plt.ylim(lower_limit, upper_limit)
        plt.colorbar()
        plt.savefig('images/' + filename + '.png')
        plt.close()

    # plot model visibilities in uv plane
    def plot_visibilities_in_uv_plane(self, u_minus_m, uv_arr, variation="stddev"):
        self.plot_spatial_array_with_colored_errors(
            uv_arr,
            u_minus_m,
            "u-v_T in uv plane",
            "u",
            "v",
            "u_minus_m_in_uv_plane"
        )
    
    # plot gain errors in position space
    def plot_gains_in_position_space(self, g_errors, ant_pos_arr):
        self.plot_spatial_array_with_colored_errors(
            ant_pos_arr,
            g_errors,
            "g-(1,0) in north-east plane",
            "N",
            "E",
            "gain_error_in_spatial_plane",
            upper_limit=300,
            lower_limit=-300,
        )

    """getters and setters"""
    # params_init_flattened
    def get_params_init_flattened(self):
        return self.params_init_flattened
    def set_params_init_flattened(self, val):
        self.params_init_flattened = val
    # caldata_obj
    def get_caldata_obj(self):
        return self.caldata_obj
    def set_caldata_obj(self, val):
        self.caldata_obj = val
    # Nants_unflagged
    def get_Nants_unflagged(self):
        return self.Nants_unflagged
    def set_Nants_unflagged(self, val):
        self.Nants_unflagged = val
    # freq_ind
    def get_freq_ind(self):
        return self.freq_ind
    def set_freq_ind(self, val):
        self.freq_ind = val
    # vis_pol_ind
    def get_vis_pol_ind(self):
        return self.vis_pol_ind
    def set_vis_pol_ind(self, val):
        self.vis_pol_ind = val