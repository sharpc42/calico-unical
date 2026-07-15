import numpy as np
import hickle as hkl

import dev_tools as dev

import subprocess
import pickle
import json
import os
import argparse
import time
from datetime import datetime

"""
    Test grid of model error and thermal noise values
    in calibration to find gain offset from Re(g)=1
    and sigma(u) - sigma(v_T)
"""
def main(calibrate            : bool = True, 
         verbose              : bool = False,
         show_plot            : bool = False,
         git_id               : str  = "",
         time_id              : str  = "",
         optim_type           : str  = "powell",
         cal_type             : str  = "unical",
         gains_multiply_model : bool = False,
         test_torch           : bool = False,
         give_gains_guess     : bool = False,
) -> None:
    data_path   = 'calico/data'
    image_path  = 'calico/images'
    file_suffix = ""

    top_start_time = time.time()

    if calibrate:
        if give_gains_guess:
            if git_id is None or time_id is None:
                raise ValueError("Need values passed for git and time IDs")
            guess_git_time_suffix = f"g{git_id}_t{time_id}"
        scaling_factors = [0.001, 1]  # skycal and truth
        sigma_t_scales  = np.arange(  0, 1, 0.05, dtype=float)
        sigma_m_scales  = np.arange(0, 1, 0.05, dtype=float)
        model_error_realizations = 1
        thermal_noise_realizations = 1
        scaling_factor_sim = 1
        output_calcs_list = []
        start_time = time.time()
        start_time_suffix = str(start_time)
        git_hash = str(subprocess.check_output(['git','rev-parse','--short','HEAD']).decode('ascii').strip())
        git_time_suffix = f"g{git_hash}_t{start_time_suffix}"
        file_suffix = git_time_suffix
        weighting_function = "constant_weights"
        if optim_type is None:
            optim_type = "powell"

        start_time_dt = datetime.fromtimestamp(start_time)
        metadata = {
            "Date"                   : f"{start_time_dt:%B %d, %Y}",
            "Time"                   : f"{start_time_dt:%H:%M:%S}",
            "Sigma_t Vals"           : ",".join(f"{noise:.3f}" for noise in sigma_t_scales.tolist()),
            "Sigma_m Vals"           : ",".join(f"{error:.3f}" for error in sigma_m_scales.tolist()),
            "Scaling Factors (Cost)" : ",".join(f"{factor:.3f}" for factor in scaling_factors),
            "Scaling Factor (Sim)"   : str(scaling_factor_sim),
            "Git ID"                 : git_hash,
            "Time ID"                : start_time_suffix,
            "Weighting Function"     : weighting_function,
            "Optimization Function"  : optim_type,
        }

        if verbose:
            print("Beginning sigma loops")
        for i, sigma_t in enumerate(sigma_t_scales):
            if np.abs(sigma_t) - 0.0 < 1e-5:
                sigma_t = 0.1
            for j, sigma_m in enumerate(sigma_m_scales):
                if np.abs(sigma_m) - 0.0 < 1e-5:
                    sigma_m = 0.1
                if verbose:
                    print(f"Creating settings files\n\tsigma_m {sigma_m}\tsigma_t {sigma_t}")
                sigma_suffix = f"{int(sigma_t*100):d}_{int(sigma_m*10):d}"
                suffix = sigma_suffix + git_time_suffix
                # if give_gains_guess: 
                #     guess_suffix = sigma_suffix + guess_git_time_suffix
                filename = f"gain_error_offset_analysis_{suffix}",
                if give_gains_guess: 
                    guess_filename = f"output_calcs_list_{guess_git_time_suffix}"
                custom_file = []
                for k, scaling_factor in enumerate(scaling_factors):
                    scaling_factor_cost = 1 / scaling_factor**2
                    custom_file.append(
                        {
                            "sigma_t"                    : float(sigma_t),
                            "sigma_n"                    : float(sigma_t),
                            "sigma_m"                    : float(sigma_m),
                            "sigma_e"                    : float(sigma_m),
                            "thermal_noise_realizations" : thermal_noise_realizations,
                            "model_error_realizations"   : model_error_realizations,
                            "weighting_function"         : weighting_function,
                            "scaling_factor_cost"        : scaling_factor_cost,
                            "scaling_factor_sim"         : scaling_factor_sim,
                            "optimization_scheme"        : optim_type,
                            "threshold_length"           : 0,
                        },
                    )
                cwd = os.getcwd()
                # with open(
                #     f'{cwd}/calico/data/{filename}_settings.hkl',
                #     mode='wb',
                # ) as file:
                #     hkl.dump(
                #         custom_file, 
                #         file, 
                #         compression='gzip',
                #     )
                hkl.dump(
                    custom_file, 
                    f'{cwd}/calico/data/{filename}_settings.hkl', 
                    compression='gzip',
                )

                if verbose:
                    print("Beginning realizations")
                this_metadata = {
                    "Date"                : f"{start_time_dt:%B %d, %Y}",
                    "Time"                : f"{start_time_dt:%H:%M:%S}",
                    "Sigma_t"             : sigma_t,
                    "Sigma_m"             : sigma_m,
                    "Git Hash"            : git_hash,
                    "Scaling Factor Sim"  : scaling_factor_sim,
                    "Scaling Factor Cost" : scaling_factor_cost,
                    "Gain Guess Given"    : give_gains_guess,
                    "Optimizer"           : optim_type,
                }
                gains_real_guess = None
                if give_gains_guess:
                    start_load_gains_guess_time = time.time()
                    print(f"Loading gains guess")
                    guess_list = hkl.load(f"{cwd}/calico/data/{guess_filename}.hkl")
                    target_sf = 1000000.0   # 1.0 for unical, 1000000.0 for skycal
                    candidates = [
                        g for g in guess_list
                        if np.isclose(g["scaling_factor_cost"], target_sf)
                    ]
                    sigma_n = np.array([c["sigma_n"] for c in candidates])
                    sigma_e = np.array([c["sigma_e"] for c in candidates])
                    distance  = (sigma_n - sigma_t)**2 + (sigma_e - sigma_m)**2
                    best_idx = int(np.argmin(distance))
                    gains_real_guess = np.asarray(candidates[best_idx]["g_arr_real"]) + \
                                       1.0j*np.asarray(candidates[best_idx]["g_arr_imag"])
                    print(f"Loading gains guess time - {time.time() - start_load_gains_guess_time:.3f} seconds")
                __import__('many_realizations_study').init_many_realizations(
                    fhd_prefix                   = '1061316296_',
                    sav_data_filename            = 'tutorial_full_onetime_unflagged',
                    sav_model_filename           = 'tutorial_full_onetime_unflagged',
                    run_params_filename          = f'{filename}_settings',
                    vis_data_writeout_filename   = 'tutorial_full_onetime_unflagged',
                    model_data_writeout_filename = 'tutorial_full_onetime_unflagged',
                    verbose                      = True,
                    simulate_visibilities        = True,
                    calibrate                    = True,
                    reconstruct_data             = False,
                    reconstruct_model            = False,
                    metadata                     = this_metadata,
                    suffix                       = suffix,
                    optimization_scheme          = optim_type,
                    calibration_type             = cal_type,
                    gains_multiply_model         = gains_multiply_model,
                    threshold_length             = 0,
                    force_fit_to_true_vis        = test_torch,
                    gains_real_guess             = gains_real_guess,
                )
                if verbose:
                    print("Finished realizations.")

                if verbose:
                    print("Reading in calculations...")
                # with open(
                #     f'{data_path}/output_calcs_{suffix}.hkl',
                #     mode='r',
                # ) as file:
                output_calcs = hkl.load(f'{data_path}/output_calcs_{suffix}.hkl')
                for output_calc_dict in output_calcs:
                    output_calcs_list.append(output_calc_dict)
                                   
                if verbose:
                    print("Cleaning up calculated saved files")
                os.system(f"rm {data_path}/output_calcs_{suffix}.hkl")

        if verbose:
            print("Calibration tests done.")

        if verbose:
            print(f"Writing out collection of output calcs...")
        hkl.dump(output_calcs_list, f'{data_path}/output_calcs_list_{file_suffix}.hkl', compression='gzip')

        if verbose:
            print(f"Writing out initial metadata...")
        # with open(f'{data_path}/metadata_{file_suffix}.hkl') as file:
        hkl.dump(metadata, f'{data_path}/metadata_{file_suffix}.hkl', compression='gzip')
        if verbose:
            print(f"Calibration tests done.\n\n*Git ID* {git_hash}\t*Start time ID* {start_time_suffix}\n")
    else:
        file_suffix = f"g{git_id}_t{time_id}"
        if verbose:
            print(f"Reading in initial metadata...")
        # with open(
        #     f'{data_path}/metadata_{file_suffix}.hkl',
        #     mode='r',
        # ) as file:
        metadata = hkl.load(f'{data_path}/metadata_{file_suffix}.hkl') 

    if verbose:
        print(f"Reading in output calcs...")
    # with open(
    #     f'{data_path}/output_calcs_list_{file_suffix}.hkl',
    #     mode='r',
    # ) as file:
    output_calcs_list = hkl.load(f'{data_path}/output_calcs_list_{file_suffix}.hkl')

    vT_minus_m_gaussian              = []
    real_sigma_t_calculated_gaussian = []
    real_g_minus_1_truth_gaussian    = []
    real_g_minus_1_skycal_gaussian   = []
    real_sigma_uvT_truth_gaussian    = []
    real_sigma_uvT_skycal_gaussian   = []
    avg_real_g_left_skycal_gaussian  = []
    avg_real_g_right_skycal_gaussian = []
    std_gain_phase                   = []
    sigma_re_m                       = []
    sigma_re_vT                      = []
    scaling_factor_truth             = None
    e_n_corr_coeff                   = []
    n_m_corr_coeff                   = []
    e_m_corr_coeff                   = []
    e_n_corr_coeff_phase             = []
    n_m_corr_coeff_phase             = []
    e_m_corr_coeff_phase             = []
    avg_cost_func_val_truth          = []
    avg_cost_func_val_skycal         = []

    filename_2d_gains = 'gain_error_vs_model_error_vs_thermal_noise_2d'
    filename_2d_u_err = 'u_error_vs_model_error_vs_thermal_noise_2d'

    sigma_t_scales = [float(i) for i in metadata["Sigma_t Vals"].split(",")]
    sigma_m_scales = [float(i) for i in metadata["Sigma_m Vals"].split(",")]

    if verbose:
        print("Collecting skycal and truth values")
    for i, calc in enumerate(output_calcs_list):
        real_sigma_uvT = calc["sigma_re_u"] - calc["sigma_re_vT"]
        # avg_mag_vTm = calc["sigma_re_vTm"]
        avg_mag_vTm = calc["avg_mag_vTm"]
        read_scaling_factor = calc["scaling_factor_cost"]
        if calc["avg_mag_vT"] < calc["avg_mag_model"]:
            avg_mag_vTm *= -1
        # print(f"{avg_mag_vTm=}")
        if read_scaling_factor - 1 < 1e-5:   
            scaling_factor_truth = read_scaling_factor
            # print(f"\n\n***TRUTH SCALING FACTOR***\n\t{scaling_factor_truth}\n\n")
            real_sigma_uvT_truth_gaussian.append(real_sigma_uvT)
            real_g_minus_1_truth_gaussian.append(calc["avg_re_g_offset"])
            avg_cost_func_val_truth.append(calc["avg_cost_func_val"])
        else:
            scaling_factor_skycal = read_scaling_factor
            # print(f"\n\n***SKYCAL SCALING FACTOR***\n\t{scaling_factor_skycal}\n\n")
            real_sigma_uvT_skycal_gaussian.append(real_sigma_uvT)
            real_g_minus_1_skycal_gaussian.append(calc["avg_re_g_offset"])
            avg_real_g_left_skycal_gaussian.append(calc["avg_re_g_minus_one_left"])   # predicted vals
            avg_real_g_right_skycal_gaussian.append(calc["avg_re_g_minus_one_right"])
            vT_minus_m_gaussian.append(avg_mag_vTm)
            real_sigma_t_calculated_gaussian.append(calc["sigma_re_n"])
            sigma_re_m.append(calc["sigma_re_m"])
            sigma_re_vT.append(calc["sigma_re_vT"])
            std_gain_phase.append(calc["std_gain_phase"])
            e_n_corr_coeff.append(calc["e_n_corr_coeff"])
            n_m_corr_coeff.append(calc["n_m_corr_coeff"])
            e_m_corr_coeff.append(calc["e_m_corr_coeff"])
            e_n_corr_coeff_phase.append(calc["e_n_corr_coeff_phase"])
            n_m_corr_coeff_phase.append(calc["n_m_corr_coeff_phase"])
            e_m_corr_coeff_phase.append(calc["e_m_corr_coeff_phase"])
            avg_cost_func_val_skycal.append(calc["avg_cost_func_val"])

    vT_minus_m_gaussian              = np.asarray(vT_minus_m_gaussian)
    real_sigma_t_calculated_gaussian = np.asarray(real_sigma_t_calculated_gaussian)
    real_g_minus_1_truth_gaussian    = np.asarray(real_g_minus_1_truth_gaussian)
    real_g_minus_1_skycal_gaussian   = np.asarray(real_g_minus_1_skycal_gaussian)
    real_sigma_uvT_truth_gaussian    = np.asarray(real_sigma_uvT_truth_gaussian)
    real_sigma_uvT_skycal_gaussian   = np.asarray(real_sigma_uvT_skycal_gaussian)
    avg_real_g_left_skycal_gaussian  = np.asarray(avg_real_g_left_skycal_gaussian)
    avg_real_g_right_skycal_gaussian = np.asarray(avg_real_g_right_skycal_gaussian)
    std_gain_phase                   = np.asarray(std_gain_phase)
    sigma_re_m                       = np.asarray(sigma_re_m)
    sigma_re_vT                      = np.asarray(sigma_re_vT)
    e_n_corr_coeff                   = np.asarray(e_n_corr_coeff)
    n_m_corr_coeff                   = np.asarray(n_m_corr_coeff)
    e_m_corr_coeff                   = np.asarray(e_m_corr_coeff)
    e_n_corr_coeff_phase             = np.asarray(e_n_corr_coeff_phase)
    n_m_corr_coeff_phase             = np.asarray(n_m_corr_coeff_phase)
    e_m_corr_coeff_phase             = np.asarray(e_m_corr_coeff_phase)
    avg_cost_func_val_truth          = np.asarray(avg_cost_func_val_truth)
    avg_cost_func_val_skycal         = np.asarray(avg_cost_func_val_skycal)

    """
    Plotting
    """
    # angle = -26.57    # degrees; scipy rotates clockwise
    angle = 0
    if verbose:
        print(f"Plotting skycal 2D grid for e-n correlation (abs)")
    dev.plot_3d_data_as_2d_hist(
        x_array       = vT_minus_m_gaussian,
        y_array       = real_sigma_t_calculated_gaussian,
        z_array       = e_n_corr_coeff,
        num_y_vals    = len(sigma_m_scales),
        num_x_vals    = len(sigma_t_scales),
        x_array_2     = sigma_re_m,
        x_array_3     = sigma_re_vT,
        plot_title    = f"|e|-|n| Correlation vs $\\sigma_t$ & $Re(v_T-m)$\n(Calculated, Skycal) - {scaling_factor_skycal:.2f}",
        plot_xlabel   = "$Re(v_T - m)$",
        # plot_xlabel_2 = "$\\sigma Re(m)$",
        plot_xlabel_3 = "$(\\downarrow \\sigma Re(m) \\downarrow) (\\uparrow \\sigma Re(v_T) \\uparrow)$",
        plot_ylabel   = "$\\sigma_t (Re)$",
        plot_vmax     = min([
                            np.abs(max(e_n_corr_coeff)),
                            np.abs(min(e_n_corr_coeff))
                        ]),
        plot_vmin     = min([
                            np.abs(max(e_n_corr_coeff)),
                            np.abs(min(e_n_corr_coeff))
                        ]),
        plot_xlim_h   = max(sigma_m_scales),
        plot_xlim_l   = min(sigma_m_scales),
        plot_ylim_h   = max(sigma_t_scales),
        plot_ylim_l   = min(sigma_t_scales),
        filename      = f'{image_path}/{filename_2d_gains}_e_n_corr_coeff_abs_{file_suffix}_gaussian.png',
        plot_cmap     = "viridis",
        cmap_label    = "|e|-|n| CorrCoef",
        suffix        = file_suffix,
        metadata      = metadata,
        box_text      = f"Optimizer: {optim_type}",
    )
    if verbose:
        print(f"Plotting skycal 2D grid for n-m correlation (abs)")
    dev.plot_3d_data_as_2d_hist(
        x_array       = vT_minus_m_gaussian,
        y_array       = real_sigma_t_calculated_gaussian,
        z_array       = n_m_corr_coeff,
        num_y_vals    = len(sigma_m_scales),
        num_x_vals    = len(sigma_t_scales),
        x_array_2     = sigma_re_m,
        x_array_3     = sigma_re_vT,
        plot_title    = f"|n|-|m| Correlation vs $\\sigma_t$ & $Re(v_T-m)$\n(Calculated, Skycal) - {scaling_factor_skycal:.2f}",
        plot_xlabel   = "$Re(v_T - m)$",
        # plot_xlabel_2 = "$\\sigma Re(m)$",
        plot_xlabel_3 = "$(\\downarrow \\sigma Re(m) \\downarrow) (\\uparrow \\sigma Re(v_T) \\uparrow)$",
        plot_ylabel   = "$\\sigma_t (Re)$",
        plot_vmax     = min([
                            np.abs(max(n_m_corr_coeff)),
                            np.abs(min(n_m_corr_coeff))
                        ]),
        plot_vmin     = min([
                            np.abs(max(n_m_corr_coeff)),
                            np.abs(min(n_m_corr_coeff))
                        ]),
        plot_xlim_h   = max(sigma_m_scales),
        plot_xlim_l   = min(sigma_m_scales),
        plot_ylim_h   = max(sigma_t_scales),
        plot_ylim_l   = min(sigma_t_scales),
        filename      = f'{image_path}/{filename_2d_gains}_n_m_corr_coeff_abs_{file_suffix}_gaussian.png',
        plot_cmap     = "viridis",
        cmap_label    = "|n|-|m| CorrCoef",
        suffix        = file_suffix,
        metadata      = metadata,
        box_text      = f"Optimizer: {optim_type}",
    )
    if verbose:
        print(f"Plotting skycal 2D grid for e-m correlation (abs)")
    dev.plot_3d_data_as_2d_hist(
        x_array       = vT_minus_m_gaussian,
        y_array       = real_sigma_t_calculated_gaussian,
        z_array       = e_m_corr_coeff,
        num_y_vals    = len(sigma_m_scales),
        num_x_vals    = len(sigma_t_scales),
        x_array_2     = sigma_re_m,
        x_array_3     = sigma_re_vT,
        plot_title    = f"|e|-|m| Correlation vs $\\sigma_t$ & $Re(v_T-m)$\n(Calculated, Skycal) - {scaling_factor_skycal:.2f}",
        plot_xlabel   = "$Re(v_T - m)$",
        # plot_xlabel_2 = "$\\sigma Re(m)$",
        plot_xlabel_3 = "$(\\downarrow \\sigma Re(m) \\downarrow) (\\uparrow \\sigma Re(v_T) \\uparrow)$",
        plot_ylabel   = "$\\sigma_t (Re)$",
        plot_vmax     = min([
                            np.abs(max(e_m_corr_coeff)),
                            np.abs(min(e_m_corr_coeff))
                        ]),
        plot_vmin     = min([
                            np.abs(max(e_m_corr_coeff)),
                            np.abs(min(e_m_corr_coeff))
                        ]),
        plot_xlim_h   = max(sigma_m_scales),
        plot_xlim_l   = min(sigma_m_scales),
        plot_ylim_h   = max(sigma_t_scales),
        plot_ylim_l   = min(sigma_t_scales),
        filename      = f'{image_path}/{filename_2d_gains}_e_m_corr_coeff_abs_{file_suffix}_gaussian.png',
        plot_cmap     = "viridis",
        cmap_label    = "|e|-|m| CorrCoef",
        suffix        = file_suffix,
        metadata      = metadata,
        box_text      = f"Optimizer: {optim_type}",
    )
    if verbose:
        print(f"Plotting skycal 2D grid for e-n correlation (phase)")
    dev.plot_3d_data_as_2d_hist(
        x_array       = vT_minus_m_gaussian,
        y_array       = real_sigma_t_calculated_gaussian,
        z_array       = e_n_corr_coeff_phase,
        num_y_vals    = len(sigma_m_scales),
        num_x_vals    = len(sigma_t_scales),
        x_array_2     = sigma_re_m,
        x_array_3     = sigma_re_vT,
        plot_title    = f"Phase e-n Correlation vs $\\sigma_t$ & $Re(v_T-m)$\n(Calculated, Skycal) - {scaling_factor_skycal:.2f}",
        plot_xlabel   = "$Re(v_T - m)$",
        # plot_xlabel_2 = "$\\sigma Re(m)$",
        plot_xlabel_3 = "$(\\downarrow \\sigma Re(m) \\downarrow) (\\uparrow \\sigma Re(v_T) \\uparrow)$",
        plot_ylabel   = "$\\sigma_t (Re)$",
        plot_vmax     = min([
                            np.abs(max(e_n_corr_coeff_phase)),
                            np.abs(min(e_n_corr_coeff_phase))
                        ]),
        plot_vmin     = min([
                            np.abs(max(e_n_corr_coeff_phase)),
                            np.abs(min(e_n_corr_coeff_phase))
                        ]),
        plot_xlim_h   = max(sigma_m_scales),
        plot_xlim_l   = min(sigma_m_scales),
        plot_ylim_h   = max(sigma_t_scales),
        plot_ylim_l   = min(sigma_t_scales),
        filename      = f'{image_path}/{filename_2d_gains}_e_m_corr_coeff_phase_{file_suffix}_gaussian.png',
        plot_cmap     = "viridis",
        cmap_label    = "Phase e-n CorrCoef",
        suffix        = file_suffix,
        metadata      = metadata,
        box_text      = f"Optimizer: {optim_type}",
    )
    if verbose:
        print(f"Plotting skycal 2D grid for n-m correlation (phase)")
    dev.plot_3d_data_as_2d_hist(
        x_array       = vT_minus_m_gaussian,
        y_array       = real_sigma_t_calculated_gaussian,
        z_array       = n_m_corr_coeff_phase,
        num_y_vals    = len(sigma_m_scales),
        num_x_vals    = len(sigma_t_scales),
        x_array_2     = sigma_re_m,
        x_array_3     = sigma_re_vT,
        plot_title    = f"Phase n-m Correlation vs $\\sigma_t$ & $Re(v_T-m)$\n(Calculated, Skycal) - {scaling_factor_skycal:.2f}",
        plot_xlabel   = "$Re(v_T - m)$",
        # plot_xlabel_2 = "$\\sigma Re(m)$",
        plot_xlabel_3 = "$(\\downarrow \\sigma Re(m) \\downarrow) (\\uparrow \\sigma Re(v_T) \\uparrow)$",
        plot_ylabel   = "$\\sigma_t (Re)$",
        plot_vmax     = min([
                            np.abs(max(n_m_corr_coeff_phase)),
                            np.abs(min(n_m_corr_coeff_phase))
                        ]),
        plot_vmin     = min([
                            np.abs(max(n_m_corr_coeff_phase)),
                            np.abs(min(n_m_corr_coeff_phase))
                        ]),
        plot_xlim_h   = max(sigma_m_scales),
        plot_xlim_l   = min(sigma_m_scales),
        plot_ylim_h   = max(sigma_t_scales),
        plot_ylim_l   = min(sigma_t_scales),
        filename      = f'{image_path}/{filename_2d_gains}_n_m_corr_coeff_phase_{file_suffix}_gaussian.png',
        plot_cmap     = "viridis",
        cmap_label    = "Phase n-m CorrCoef",
        suffix        = file_suffix,
        metadata      = metadata,
        box_text      = f"Optimizer: {optim_type}",
    )
    if verbose:
        print(f"Plotting skycal 2D grid for e-m correlation (phase)")
    dev.plot_3d_data_as_2d_hist(
        x_array       = vT_minus_m_gaussian,
        y_array       = real_sigma_t_calculated_gaussian,
        z_array       = e_m_corr_coeff_phase,
        num_y_vals    = len(sigma_m_scales),
        num_x_vals    = len(sigma_t_scales),
        x_array_2     = sigma_re_m,
        x_array_3     = sigma_re_vT,
        plot_title    = f"Phase e-m Correlation vs $\\sigma_t$ & $Re(v_T-m)$\n(Calculated, Skycal) - {scaling_factor_skycal:.2f}",
        plot_xlabel   = "$Re(v_T - m)$",
        # plot_xlabel_2 = "$\\sigma Re(m)$",
        plot_xlabel_3 = "$(\\downarrow \\sigma Re(m) \\downarrow) (\\uparrow \\sigma Re(v_T) \\uparrow)$",
        plot_ylabel   = "$\\sigma_t (Re)$",
        plot_vmax     = min([
                            np.abs(max(e_m_corr_coeff_phase)),
                            np.abs(min(e_m_corr_coeff_phase))
                        ]),
        plot_vmin     = min([
                            np.abs(max(e_m_corr_coeff_phase)),
                            np.abs(min(e_m_corr_coeff_phase))
                        ]),
        plot_xlim_h   = max(sigma_m_scales),
        plot_xlim_l   = min(sigma_m_scales),
        plot_ylim_h   = max(sigma_t_scales),
        plot_ylim_l   = min(sigma_t_scales),
        filename      = f'{image_path}/{filename_2d_gains}_e_m_corr_coeff_phase_{file_suffix}_gaussian.png',
        plot_cmap     = "viridis",
        cmap_label    = "Phase e-m CorrCoef",
        suffix        = file_suffix,
        metadata      = metadata,
        box_text      = f"Optimizer: {optim_type}",
    )
    if verbose:
        print(f"Plotting unical 2D grid for final cost function value")
    dev.plot_3d_data_as_2d_hist(
        x_array       = vT_minus_m_gaussian,
        y_array       = real_sigma_t_calculated_gaussian,
        z_array       = avg_cost_func_val_truth,
        num_y_vals    = len(sigma_m_scales),
        num_x_vals    = len(sigma_t_scales),
        x_array_2     = sigma_re_m,
        x_array_3     = sigma_re_vT,
        plot_title    = f"Avg Final Cost Func. Value vs $\\sigma_t$ & $Re(v_T-m)$\n(Calculated, Unical) - {scaling_factor_truth:.2f}",
        plot_xlabel   = "$Re(v_T - m)$",
        # plot_xlabel_2 = "$\\sigma Re(m)$",
        plot_xlabel_3 = "$(\\downarrow \\sigma Re(m) \\downarrow) (\\uparrow \\sigma Re(v_T) \\uparrow)$",
        plot_ylabel   = "$\\sigma_t (Re)$",
        plot_vmax     = min([
                            np.abs(max(avg_cost_func_val_truth)),
                            np.abs(min(avg_cost_func_val_truth))
                        ]),
        plot_vmin     = min([
                            np.abs(max(avg_cost_func_val_truth)),
                            np.abs(min(avg_cost_func_val_truth))
                        ]),
        plot_xlim_h   = max(sigma_m_scales),
        plot_xlim_l   = min(sigma_m_scales),
        plot_ylim_h   = max(sigma_t_scales),
        plot_ylim_l   = min(sigma_t_scales),
        filename      = f'{image_path}/{filename_2d_gains}_avg_cost_func_val_unical_{file_suffix}_gaussian.png',
        plot_cmap     = "viridis",
        cmap_label    = "Avg. Final Cost Func. Val.",
        suffix        = file_suffix,
        metadata      = metadata,
        box_text      = f"Optimizer: {optim_type}",
    )
    if verbose:
        print(f"Plotting skycal 2D grid for final cost function value")
    dev.plot_3d_data_as_2d_hist(
        x_array       = vT_minus_m_gaussian,
        y_array       = real_sigma_t_calculated_gaussian,
        z_array       = avg_cost_func_val_skycal,
        num_y_vals    = len(sigma_m_scales),
        num_x_vals    = len(sigma_t_scales),
        x_array_2     = sigma_re_m,
        x_array_3     = sigma_re_vT,
        plot_title    = f"Avg Final Cost Func. Value vs $\\sigma_t$ & $Re(v_T-m)$\n(Calculated, Skycal) - {scaling_factor_skycal:.2f}",
        plot_xlabel   = "$Re(v_T - m)$",
        # plot_xlabel_2 = "$\\sigma Re(m)$",
        plot_xlabel_3 = "$(\\downarrow \\sigma Re(m) \\downarrow) (\\uparrow \\sigma Re(v_T) \\uparrow)$",
        plot_ylabel   = "$\\sigma_t (Re)$",
        plot_vmax     = min([
                            np.abs(max(avg_cost_func_val_skycal)),
                            np.abs(min(avg_cost_func_val_skycal))
                        ]),
        plot_vmin     = min([
                            np.abs(max(avg_cost_func_val_skycal)),
                            np.abs(min(avg_cost_func_val_skycal))
                        ]),
        plot_xlim_h   = max(sigma_m_scales),
        plot_xlim_l   = min(sigma_m_scales),
        plot_ylim_h   = max(sigma_t_scales),
        plot_ylim_l   = min(sigma_t_scales),
        filename      = f'{image_path}/{filename_2d_gains}_avg_cost_func_val_skycal_{file_suffix}_gaussian.png',
        plot_cmap     = "viridis",
        cmap_label    = "Avg. Final Cost Func. Val.",
        suffix        = file_suffix,
        metadata      = metadata,
        box_text      = f"Optimizer: {optim_type}",
    )
    if verbose:
        print(f"Plotting truth 2D grid for gain offset")
    dev.plot_3d_data_as_2d_hist(
        x_array     = vT_minus_m_gaussian,
        y_array     = real_sigma_t_calculated_gaussian,
        z_array     = real_g_minus_1_truth_gaussian,
        num_y_vals  = len(sigma_m_scales),
        num_x_vals  = len(sigma_t_scales),
        x_array_2   = sigma_re_m,
        x_array_3   = sigma_re_vT,
        plot_title  = f"Gain Offset vs $\\sigma_t$ & $Re(v_T-m)$\n(Calculated, Truth) - Scale Factor: {scaling_factor_truth:.2f}",
        plot_xlabel = "$Re(v_T - m)$",
        # plot_xlabel_2 = "$\\sigma Re(m)$",
        plot_xlabel_3 = "$(\\downarrow \\sigma Re(m) \\downarrow) (\\uparrow \\sigma Re(v_T) \\uparrow)$",
        plot_ylabel = "$\\sigma_t (Re)$",
        plot_vmax     = min([
                            np.abs(max(real_g_minus_1_truth_gaussian)),
                            np.abs(min(real_g_minus_1_truth_gaussian))
                        ]),
        plot_vmin     = -min([
                            np.abs(max(real_g_minus_1_truth_gaussian)),
                            np.abs(min(real_g_minus_1_truth_gaussian))
                        ]),
        # plot_vmin = -0.001,
        # plot_vmax = 0.001,
        plot_xlim_h = max(sigma_m_scales),
        plot_xlim_l = min(sigma_m_scales),
        plot_ylim_h = max(sigma_t_scales),
        plot_ylim_l = min(sigma_t_scales),
        filename    = f'{image_path}/{filename_2d_gains}_truth_{file_suffix}_gaussian.png',
        plot_cmap   = "PuOr",
        cmap_label  = "<$Re(g)>-1$",
        suffix      = file_suffix,
        metadata    = metadata,
        angle       = angle,
        box_text      = f"Optimizer: {optim_type}",
    )
    if verbose:
        print(f"Plotting skycal 2D grid for gain offset")
    dev.plot_3d_data_as_2d_hist(
        x_array       = vT_minus_m_gaussian,
        y_array       = real_sigma_t_calculated_gaussian,
        z_array       = real_g_minus_1_skycal_gaussian,
        num_y_vals    = len(sigma_m_scales),
        num_x_vals    = len(sigma_t_scales),
        x_array_2     = sigma_re_m,
        x_array_3     = sigma_re_vT,
        plot_title    = f"Gain Offset vs $\\sigma_t$ & $Re(v_T-m)$\n(Calculated, Skycal) - {scaling_factor_skycal:.2f}",
        plot_xlabel   = "$Re(v_T - m)$",
        # plot_xlabel_2 = "$\\sigma Re(m)$",
        plot_xlabel_3 = "$(\\downarrow \\sigma Re(m) \\downarrow) (\\uparrow \\sigma Re(v_T) \\uparrow)$",
        plot_ylabel   = "$\\sigma_t (Re)$",
        plot_vmax     = min([
                            np.abs(max(real_g_minus_1_skycal_gaussian)),
                            np.abs(min(real_g_minus_1_skycal_gaussian))
                        ]),
        plot_vmin     = -min([
                            np.abs(max(real_g_minus_1_skycal_gaussian)),
                            np.abs(min(real_g_minus_1_skycal_gaussian))
                        ]),
        # plot_vmax=0.001,
        # plot_vmin=-0.001,
        plot_xlim_h   = max(sigma_m_scales),
        plot_xlim_l   = min(sigma_m_scales),
        plot_ylim_h   = max(sigma_t_scales),
        plot_ylim_l   = min(sigma_t_scales),
        filename      = f'{image_path}/{filename_2d_gains}_skycal_{file_suffix}_gaussian.png',
        plot_cmap     = "PuOr",
        cmap_label    = "$<Re(g)>-1$",
        suffix        = file_suffix,
        metadata      = metadata,
        angle         = angle,
        box_text      = f"Optimizer: {optim_type}",
    )
    if verbose:
        print(f"Plotting truth-skycal diff 2D grid for gain offset")
    real_g_minus_1_diff_of_abs = np.abs(np.asarray(real_g_minus_1_truth_gaussian)) - \
        np.abs(np.asarray(real_g_minus_1_skycal_gaussian))
    real_g_minus_1_diff = np.asarray(real_g_minus_1_truth_gaussian) - \
        np.asarray(real_g_minus_1_skycal_gaussian)
    dev.plot_3d_data_as_2d_hist(
        x_array       = vT_minus_m_gaussian,
        y_array       = real_sigma_t_calculated_gaussian,
        z_array       = real_g_minus_1_diff,
        num_y_vals    = len(sigma_m_scales),
        num_x_vals    = len(sigma_t_scales),
        x_array_2     = sigma_re_m,
        x_array_3     = sigma_re_vT,
        plot_title    = "Gain Offset vs $\\sigma_t$ & $Re(v_T-m)$\n(Calculated, Truth Skycal Diff)",
        plot_xlabel   = "$Re(v_T - m)$",
        # plot_xlabel_2 = "$\\sigma Re(m)$",
        plot_xlabel_3 = "$(\\downarrow \\sigma Re(m) \\downarrow) (\\uparrow \\sigma Re(v_T) \\uparrow)$",
        plot_ylabel   = "$\\sigma_t (Re)$",
        # log_cmap      = True,
        plot_vmax     = min([
                            np.abs(max(real_g_minus_1_skycal_gaussian)),
                            np.abs(min(real_g_minus_1_skycal_gaussian))
                        ]),
        plot_vmin     = -min([
                            np.abs(max(real_g_minus_1_skycal_gaussian)),
                            np.abs(min(real_g_minus_1_skycal_gaussian))
                        ]),
        # plot_vmax=0.001,
        # plot_vmin=-0.001,
        filename      = f'{image_path}/{filename_2d_gains}_diff_{file_suffix}_gaussian.png',
        plot_cmap     = "PuOr",
        cmap_label    = "$<Re(g)>-1$",
        suffix        = file_suffix,
        metadata      = metadata,
        box_text      = f"Optimizer: {optim_type}",
    )
    dev.plot_3d_data_as_2d_hist(
        x_array       = vT_minus_m_gaussian,
        y_array       = real_sigma_t_calculated_gaussian,
        z_array       = real_g_minus_1_diff_of_abs,
        num_y_vals    = len(sigma_m_scales),
        num_x_vals    = len(sigma_t_scales),
        x_array_2     = sigma_re_m,
        x_array_3     = sigma_re_vT,
        plot_title    = "Gain Offset vs $\\sigma_t$ & $Re(v_T-m)$\n(Calculated, Truth Skycal Diff of Abs)",
        plot_xlabel   = "$Re(v_T - m)$",
        # plot_xlabel_2 = "$\\sigma Re(m)$",
        plot_xlabel_3 = "$(\\downarrow \\sigma Re(m) \\downarrow) (\\uparrow \\sigma Re(v_T) \\uparrow)$",
        plot_ylabel   = "$\\sigma_t (Re)$",
        # log_cmap      = True,
        plot_vmax     = min([
                            np.abs(max(real_g_minus_1_skycal_gaussian)),
                            np.abs(min(real_g_minus_1_skycal_gaussian))
                        ]),
        plot_vmin     = -min([
                            np.abs(max(real_g_minus_1_skycal_gaussian)),
                            np.abs(min(real_g_minus_1_skycal_gaussian))
                        ]),
        # plot_vmax=0.001,
        # plot_vmin=-0.001,
        filename      = f'{image_path}/{filename_2d_gains}_diff_abs_{file_suffix}_gaussian.png',
        plot_cmap     = "PuOr",
        cmap_label    = "$<Re(g)>-1$",
        suffix        = file_suffix,
        metadata      = metadata,
        box_text      = f"Optimizer: {optim_type}",
    )
    if verbose:
        print(f"Plotting prediction 2D grid (left) for gain offset")
    dev.plot_3d_data_as_2d_hist(
        x_array       = vT_minus_m_gaussian,
        y_array       = real_sigma_t_calculated_gaussian,
        z_array       = avg_real_g_left_skycal_gaussian,
        num_y_vals    = len(sigma_m_scales),
        num_x_vals    = len(sigma_t_scales),
        x_array_2     = sigma_re_m,
        x_array_3     = sigma_re_vT,
        plot_title    = "Gain Offset vs $\\sigma_t$ & $Re(v_T-m)$\n(Predicted, Skycal, Left)",
        plot_xlabel   = "$Re(v_T - m)$",
        # plot_xlabel_2 = "$\\sigma Re(m)$",
        plot_xlabel_3 = "$(\\downarrow \\sigma Re(m) \\downarrow) (\\uparrow \\sigma Re(v_T) \\uparrow)$",
        plot_ylabel   = "$\\sigma_t (Re)$",
        plot_vmax     = min([
                            np.abs(max(avg_real_g_left_skycal_gaussian)),
                            np.abs(min(avg_real_g_left_skycal_gaussian))
                        ]),
        plot_vmin     = -min([
                            np.abs(max(avg_real_g_left_skycal_gaussian)),
                            np.abs(min(avg_real_g_left_skycal_gaussian))
                        ]),
        # plot_vmax     = 0.4,
        # plot_vmin     = -0.4,
        plot_xlim_h   = 0,
        plot_xlim_l   = -10,
        plot_ylim_h   = 10,
        plot_ylim_l   = 0,
        filename      = f'{image_path}/{filename_2d_gains}_predict_left_{file_suffix}_gaussian.png',
        plot_cmap     = "PuOr",
        cmap_label    = "$<Re(g)>-1$",
        suffix        = file_suffix,
        metadata      = metadata,
        box_text      = f"Optimizer: {optim_type}",
    )
    if verbose:
        print(f"Plotting prediction 2D grid (right) for gain offset")
    dev.plot_3d_data_as_2d_hist(
        x_array       = vT_minus_m_gaussian,
        y_array       = real_sigma_t_calculated_gaussian,
        z_array       = avg_real_g_right_skycal_gaussian,
        num_y_vals    = len(sigma_m_scales),
        num_x_vals    = len(sigma_t_scales),
        x_array_2     = sigma_re_m,
        x_array_3     = sigma_re_vT,
        plot_title    = "Gain Offset vs $\\sigma_t$ & $Re(v_T-m)$\n(Predicted, Skycal, Right)",
        plot_xlabel   = "$Re(v_T - m)$",
        # plot_xlabel_2 = "$\\sigma Re(m)$",
        plot_xlabel_3 = "$(\\downarrow \\sigma Re(m) \\downarrow) (\\uparrow \\sigma Re(v_T) \\uparrow)$",
        plot_ylabel   = "$\\sigma_t (Re)$",
        plot_vmax     = min([
                            np.abs(max(avg_real_g_right_skycal_gaussian)),
                            np.abs(min(avg_real_g_right_skycal_gaussian))
                        ]),
        plot_vmin     = -min([
                            np.abs(max(avg_real_g_right_skycal_gaussian)),
                            np.abs(min(avg_real_g_right_skycal_gaussian))
                        ]),
        # plot_vmax     = 0.4,
        # plot_vmin     = -0.4,
        plot_xlim_h   = 10,
        plot_xlim_l   = -10,
        plot_ylim_h   = 10,
        plot_ylim_l   = 0,
        filename      = f'{image_path}/{filename_2d_gains}_predict_right_{file_suffix}_gaussian.png',
        plot_cmap     = "PuOr",
        cmap_label    = "$<Re(g)>-1$",
        suffix        = file_suffix,
        metadata      = metadata,
        box_text      = f"Optimizer: {optim_type}",
    )
    # if verbose:
    #     print(f"Plotting prediction 2D grid (joined) for gain offset")
    # dev.plot_3d_data_as_2d_hist(
    #     x_array       = vT_minus_m_gaussian,
    #     y_array       = real_sigma_t_calculated_gaussian,
    #     z_array       = avg_real_g_right_skycal_gaussian,
    #     x_array_2     = sigma_re_m,
    #     x_array_3     = sigma_re_vT,
    #     z_array_2     = avg_real_g_left_skycal_gaussian,
    #     plot_title    = "Gain Offset vs $\\sigma_t$ & $Re(v_T-m)$\n(Predicted, Skycal, Joined)",
    #     plot_xlabel   = "$Re(v_T - m)$",
    #     # plot_xlabel_2 = "$\\sigma Re(m)$",
    #     plot_xlabel_3 = "$(\\downarrow \\sigma Re(m) \\downarrow) (\\uparrow \\sigma Re(v_T) \\uparrow)$",
    #     plot_ylabel   = "$\\sigma_t (Re)$",
    #     plot_vmax     = 0.5,
    #     plot_vmin     = -0.5,
    #     plot_xlim_h   = 10,
    #     plot_xlim_l   = -10,
    #     plot_ylim_h   = 10,
    #     plot_ylim_l   = 0,
    #     filename      = f'{image_path}/{filename_2d_gains}_predict_joined_{file_suffix}_gaussian.png',
    #     plot_cmap     = "PuOr",
    #     cmap_label    = "$Re(g)-1$",
    #     suffix        = file_suffix,
    #     metadata      = metadata,
    # )
    if verbose:
        print(f"Plotting standard deviation in gain phase")
    dev.plot_3d_data_as_2d_hist(
        x_array       = vT_minus_m_gaussian,
        y_array       = real_sigma_t_calculated_gaussian,
        z_array       = std_gain_phase,
        num_y_vals    = len(sigma_m_scales),
        num_x_vals    = len(sigma_t_scales),
        x_array_2     = sigma_re_m,
        x_array_3     = sigma_re_vT,
        plot_title    = "Standard Deviation in Gain Phase (Skycal)",
        plot_xlabel   = "$Re(v_T - m)$",
        # plot_xlabel_2 = "$\\sigma Re(m)$",
        plot_xlabel_3 = "$(\\downarrow \\sigma Re(m) \\downarrow) (\\uparrow \\sigma Re(v_T) \\uparrow)$",
        plot_ylabel   = "$\\sigma_t (Re)$",
        plot_vmax     = np.pi,
        plot_vmin     = 0,
        # log_cmap      = True,
        filename      = f'{image_path}/{filename_2d_gains}_gain_phase_{file_suffix}_gaussian.png',
        plot_cmap     = "viridis",
        cmap_label    = "$Std Phase Re(g)-1$",
        suffix        = file_suffix,
        metadata      = metadata,
        box_text      = f"Optimizer: {optim_type}",
    )
    if verbose:
        print(f"Plotting truth 2D grid for u offset")
    dev.plot_3d_data_as_2d_hist(
        x_array       = vT_minus_m_gaussian,
        y_array       = real_sigma_t_calculated_gaussian,
        z_array       = real_sigma_uvT_truth_gaussian,
        num_y_vals    = len(sigma_m_scales),
        num_x_vals    = len(sigma_t_scales),
        x_array_2     = sigma_re_m,
        x_array_3     = sigma_re_vT,
        plot_title    = "$\\sigma_u - \\sigma_v$ vs $\\sigma_t$ & $Re(v_T-m)$\n(Calculated, Truth)",
        plot_xlabel   = "$Re(v_T - m)$",
        # plot_xlabel_2 = "$\\sigma Re(m)$",
        plot_xlabel_3 = "$(\\downarrow \\sigma Re(m) \\downarrow) (\\uparrow \\sigma Re(v_T) \\uparrow)$",
        plot_ylabel   = "$\\sigma_t (Re)$",
        plot_vmax     = 15,
        plot_vmin     = -15,
        plot_xlim_h   = 0.1,
        plot_xlim_l   = -0.1,
        plot_ylim_h   = 0.1,
        plot_ylim_l   = 0,
        filename      = f'{image_path}/{filename_2d_u_err}_truth_{file_suffix}_gaussian.png',
        plot_cmap     = "seismic",
        suffix        = file_suffix,
        metadata      = metadata,
        box_text      = f"Optimizer: {optim_type}",
    )
    if verbose:
        print(f"Plotting skycal 2D grid for u offset")
    dev.plot_3d_data_as_2d_hist(
        x_array       = vT_minus_m_gaussian,
        y_array       = real_sigma_t_calculated_gaussian,
        z_array       = real_sigma_uvT_skycal_gaussian,
        num_y_vals    = len(sigma_m_scales),
        num_x_vals    = len(sigma_t_scales),
        x_array_2     = sigma_re_m,
        x_array_3     = sigma_re_vT,
        plot_title    = "$\\sigma_u - \\sigma_v$ vs $\\sigma_t$ & $Re(v_T-m)$\n(Calculated, Skycal)",
        plot_xlabel   = "$Re(v_T - m)$",
        # plot_xlabel_2 = "$\\sigma Re(m)$",
        plot_xlabel_3 = "$(\\downarrow \\sigma Re(m) \\downarrow) (\\uparrow \\sigma Re(v_T) \\uparrow)$",
        plot_ylabel   = "$\\sigma_t (Re)$",
        plot_vmax     = 15,
        plot_vmin     = -15,
        plot_xlim_h   = 0.1,
        plot_xlim_l   = -0.1,
        plot_ylim_h   = 0.1,
        plot_ylim_l   = 0,
        filename      = f'{image_path}/{filename_2d_u_err}_skycal_{file_suffix}_gaussian.png',
        plot_cmap     = "seismic",
        suffix        = file_suffix,
        metadata      = metadata,
        box_text      = f"Optimizer: {optim_type}",
    )
    if verbose:
        print(f"Plotting truth-skycal diff 2D grid for u offset")
    sigma_uvT_diff = np.asarray(real_sigma_uvT_truth_gaussian) - \
        np.asarray(real_sigma_uvT_skycal_gaussian)
    dev.plot_3d_data_as_2d_hist(
        x_array       = vT_minus_m_gaussian,
        y_array       = real_sigma_t_calculated_gaussian,
        z_array       = sigma_uvT_diff,
        num_y_vals    = len(sigma_m_scales),
        num_x_vals    = len(sigma_t_scales),
        x_array_2     = sigma_re_m,
        x_array_3     = sigma_re_vT,
        plot_title    = "$\\sigma_u - \\sigma_v$ vs $\\sigma_t$ & $Re(v_T-m)$\n(Calculated, Truth Skycal Diff)",
        plot_xlabel   = "$Re(v_T - m)$",
        # plot_xlabel_2 = "$\\sigma Re(m)$",
        plot_xlabel_3 = "$(\\downarrow \\sigma Re(m) \\downarrow) (\\uparrow \\sigma Re(v_T) \\uparrow)$",
        plot_ylabel   = "$\\sigma_t (Re)$",
        plot_vmax     = 15,
        plot_vmin     = -15,
        plot_xlim_h   = 0.1,
        plot_xlim_l   = -0.1,
        plot_ylim_h   = 0.1,
        plot_ylim_l   = 0,
        filename      = f'{image_path}/{filename_2d_u_err}_diff_{file_suffix}_gaussian.png',
        plot_cmap     = "seismic",
        suffix        = file_suffix,
        metadata      = metadata,
        box_text      = f"Optimizer: {optim_type}",
    )
    if verbose:
        print("Creating 3D plot.")
    filename_3d_scatter_gain = 'gain_error_vs_model_error_vs_thermal_noise_3d'
    dev.build_3d_scatter_plot(
        x_array           = vT_minus_m_gaussian,
        y_array           = real_sigma_t_calculated_gaussian,
        z_array           = real_g_minus_1_truth_gaussian,
        z_array_2         = real_g_minus_1_skycal_gaussian,
        second_plot       = True,
        show_plot         = show_plot,
        plot_title        = "Gain Offset vs Model Error and Thermal Noise",
        plot_xlabel       = "$Re(v_T - m)$",
        plot_ylabel       = "$\\sigma_t (Re)$", 
        plot_zlabel       = "Re(g-1)",
        xlim_hi           = 16,    xlim_lo     = -16,
        ylim_hi           = 16,    ylim_lo     = -1,
        zlim_hi           = 0.5,  zlim_lo     = -0.5,
        filename          = f'{image_path}/{filename_3d_scatter_gain}_{file_suffix}_gaussian.png',
        suffix            = file_suffix,
        metadata          = metadata,
        first_plot_label  = "truth",
        second_plot_label = "skycal",
    )
    filename_3d_scatter_model = 'uvT_vs_model_error_vs_thermal_noise_3d'
    dev.build_3d_scatter_plot(
        x_array           = vT_minus_m_gaussian,
        y_array           = real_sigma_t_calculated_gaussian,
        z_array           = real_sigma_uvT_truth_gaussian,
        z_array_2         = real_sigma_uvT_skycal_gaussian,
        second_plot       = True,
        show_plot         = show_plot,
        plot_title        = "$\\sigma_u - \\sigma_v$ vs Model Error and Thermal Noise",
        plot_xlabel       = "$Re(v_T - m)$",
        plot_ylabel       = "$\\sigma_t (Re)$", 
        plot_zlabel       = "$\\sigma_u - \\sigma_v$",
        xlim_hi           = 16, xlim_lo     = -16,
        ylim_hi           = 16, ylim_lo     = -1,
        zlim_hi           = 15,  zlim_lo     = -15,
        filename          = f'{image_path}/{filename_3d_scatter_model}_{file_suffix}_gaussian.png',
        suffix            = file_suffix,
        metadata          = metadata,
        first_plot_label  = "truth",
        second_plot_label = "skycal",
    )

    print(f"\n\tTime taken for many error vals:\n\t{(time.time() - top_start_time)/3600:.4f} hours ({calibrate=})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Gain Error Offset Analysis")
    parser.add_argument(
        "-c", action="store_true", 
        help="run calibrations",
    )
    parser.add_argument(
        "-v", action="store_true", 
        help="verbose",
    )
    parser.add_argument(
        "-s", action="store_true", 
        help="show 3D scatter plot",
    )
    parser.add_argument(
        "-n", "--nosave", action="store_true", 
        help="don't save files (requires re-running calibration next time)",
    )
    parser.add_argument(
        "--time", type=str,
        help="time ID for loading saved calibration run",
    )
    parser.add_argument(
        "--git", type=str,
        help="git ID for loading saved calibration run",
    )
    parser.add_argument(
        "--optim", type=str,
        help="type of optimization scheme to use for calibration"
    )
    parser.add_argument(
        "--caltype", type=str,
        help="type of calibration to use"
    )
    parser.add_argument(
        "--gmm", action="store_true",
        help="gains multiply model in the calibration"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="test pytorch optim by forcing u = v_T"
    )
    parser.add_argument(
        "--guess", action="store_true",
        help="give initial guess for the gains"
    )
    args = parser.parse_args()

    main(
        calibrate=args.c, 
        verbose=args.v, 
        show_plot=args.s,
        time_id=args.time,
        git_id=args.git,
        optim_type=args.optim,
        cal_type=args.caltype,
        gains_multiply_model=args.gmm,
        test_torch=args.test,
        give_gains_guess=args.guess,
    )