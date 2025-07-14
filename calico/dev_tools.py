from scipy.differentiate import jacobian
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from calico import cost_function_calculations, calibration_optimization as cal_opt

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
    ):
        print("***FIT TESTS***")
        print("\tGains Error, Min -", np.min(np.abs(self.caldata_obj.gains - 1)))
        print("\tGains Error, Max -", np.max(np.abs(self.caldata_obj.gains - 1)))
        print("\t|u-m|, Min -", np.min(np.abs(self.caldata_obj.fit_vis - self.caldata_obj.model_visibilities)))
        print("\t|u-m|, Max -", np.max(np.abs(self.caldata_obj.fit_vis - self.caldata_obj.model_visibilities)))
        if type is "trajectory" or type is "both":
            # plot gains parameters trajectory
            self.complex_trajectory_plot(
                gains_array,
                self.caldata_obj.gains,
                len(self.caldata_obj.ant_inds),
                "Gains Trajectory Plot",
                (0.75, 1.25),
                (-0.25, 0.25),
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
        if type is "scatter":
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
        if type is "histogram" or type is "both":
            fig1, ax1 = plt.subplots()
            # gains plot
            # hh1 = ax1.hist2d(gains_array.real - 1, gains_array.imag, bins=50, cmap="inferno")
            plt.scatter(self.caldata_obj.gains.real - 1, self.caldata_obj.gains.imag)
            # ax1.set_xlim(-0.1, 0.1)
            # ax1.set_ylim(-0.05, 0.05)
            ax1.set_title("Final Gains")
            ax1.set_xlabel("Real - 1")
            ax1.set_ylabel("Imag")
            # fig1.colorbar(hh1[3], ax=ax1)
            plt.savefig('images/' + "final_gains_"
                 + subprocess.check_output(['git','rev-parse','--short','HEAD']).decode('ascii').strip()
                 + '.png',
                 bbox_inches=0,)
            plt.close()
            # models plot
            u_minus_m_after = models_array - self.caldata_obj.model_visibilities[:1,:,0]
            fig2, ax2 = plt.subplots()
            # hh2 = ax2.hist2d(models_array.real, models_array.imag, bins=50, cmap="inferno")
            plt.scatter(u_minus_m_after.real, u_minus_m_after.imag)
            # plt.xlim(-15, 15)
            # plt.ylim(-15, 15)
            ax2.set_title("Final u-m")
            ax2.set_xlabel("Real")
            ax2.set_ylabel("Imag")
            # fig2.colorbar(hh2[3], ax=ax2)
            plt.savefig('images/' + "final_fit-vis_"
                 + subprocess.check_output(['git','rev-parse','--short','HEAD']).decode('ascii').strip()
                 + '.png',
                 bbox_inches=0,)
            plt.close()
    
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