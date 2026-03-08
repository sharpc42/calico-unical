import numpy as np

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
def main(calibrate : bool = True, 
         verbose   : bool = False,
         show_plot : bool = False,
         git_id    : str  = "",
         time_id   : str  = "",
) -> None:
    data_path   = 'calico/data'
    image_path  = 'calico/images'
    file_suffix = ""

    if calibrate:
        scaling_factors = [0.1, 1]  # skycal and truth
        sigma_t_scales  = np.arange(  0, 15, 17/3)
        sigma_m_scales  = np.arange(-15, 15, 17/3)
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
        }

        if verbose:
            print("Beginning sigma loops")
        for sigma_t in sigma_t_scales:
            if np.abs(sigma_t) - 0.0 < 1e-5:
                sigma_t = 0.1
            for sigma_m in sigma_m_scales:
                if np.abs(sigma_m) - 0.0 < 1e-5:
                    sigma_m = 0.1
                if verbose:
                    print(f"Creating settings files\n\tsigma_m {sigma_m}\tsigma_t {sigma_t}")
                suffix = f"{int(sigma_t*100):d}_{int(sigma_m*10):d}"
                suffix += git_time_suffix
                filename = f"gain_error_offset_analysis_{suffix}",
                custom_file = []
                for scaling_factor in scaling_factors:
                    scaling_factor_cost = 1 / scaling_factor**2
                    custom_file.append(
                        {
                            "sigma_t"                    : sigma_t,
                            "sigma_n"                    : sigma_t,
                            "sigma_m"                    : sigma_m,
                            "sigma_e"                    : sigma_m,
                            "thermal_noise_realizations" : thermal_noise_realizations,
                            "model_error_realizations"   : model_error_realizations,
                            "weighting_function"         : weighting_function,
                            "scaling_factor_cost"        : scaling_factor_cost,
                            "scaling_factor_sim"         : scaling_factor_sim,
                        },
                    )
                cwd = os.getcwd()
                with open(
                    f'{cwd}/calico/data/{filename}_settings.json', 
                    mode='w',
                    encoding='utf-8' 
                ) as file:
                    json.dump(custom_file, file)

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
                }
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
                )
                if verbose:
                    print("Finished realizations.")

                if verbose:
                    print("Reading in calculations...")
                with open(
                    f'{data_path}/output_calcs_{suffix}.json',
                    mode='r'
                ) as file:
                    output_calcs = json.load(file)
                output_calcs_list.append(output_calcs)
                                   
                if verbose:
                    print("Cleaning up calculated saved files")
                os.system(f"rm {data_path}/gain_error_offset_analysis_output_calcs.json")
                print(f"\n\n***OUTPUT CALCS LIST (cal)***\n\t{len(output_calcs_list)=}\n{output_calcs_list}\n\n")

        if verbose:
            print("Calibration tests done.")

        if verbose:
            print(f"Writing out collection of output calcs...")
        with open(
            f'{data_path}/output_calcs_list_{file_suffix}.json',
            mode='w',
        ) as file:
            json.dump(output_calcs_list, file)

        if verbose:
            print(f"Writing out initial metadata...")
        with open(
            f'{data_path}/metadata_{file_suffix}.json',
            mode='w',
        ) as file:
            json.dump(metadata, file)
        if verbose:
            print(f"Calibration tests done.\n\n*Git ID* {git_hash}\t*Start time ID* {start_time_suffix}\n")
    else:
        file_suffix = f"g{git_id}_t{time_id}"
        if verbose:
            print(f"Reading in initial metadata...")
        with open(
            f'{data_path}/metadata_{file_suffix}.json',
            mode='r',
        ) as file:
            metadata = json.load(file) 

    if verbose:
        print(f"Reading in output calcs...")
    with open(
        f'{data_path}/output_calcs_list_{file_suffix}.json',
        mode='r',
    ) as file:
        output_calcs_list = json.load(file)

    vT_minus_m_gaussian              = []
    real_sigma_t_calculated_gaussian = []
    real_g_minus_1_truth_gaussian    = []
    real_g_minus_1_skycal_gaussian   = []
    real_sigma_uvT_truth_gaussian    = []
    real_sigma_uvT_skycal_gaussian   = []

    filename_2d_gains = 'gain_error_vs_model_error_vs_thermal_noise_2d'
    filename_2d_u_err = 'u_error_vs_model_error_vs_thermal_noise_2d'

    sigma_t_scales = [float(i) for i in metadata["Sigma_t Vals"].split(",")]
    sigma_m_scales = [float(i) for i in metadata["Sigma_m Vals"].split(",")]

    # sigma_t_vals = []
    # sigma_m_vals = []
    # print(f"\n\n***OUTPUT CALCS LIST (plot)***\n\t{len(output_calcs_list)=}\n{output_calcs_list}\n\n")
    if verbose:
        print("Collecting skycal and truth values")
    # print("\n\n***PLOTTING***")
    for i, sigma_t in enumerate(sigma_t_scales):
        if np.abs(sigma_t) - 0.0 < 1e-5:
            sigma_t = 0.1
        # print(f"\n{sigma_t=:.3f}")
        for j, sigma_m in enumerate(sigma_m_scales):
            this_calcs_list = output_calcs_list[i+j]
            if np.abs(sigma_m) - 0.0 < 1e-5:
                sigma_m = 0.1
            # print(f"\n{sigma_m=:.3f}")
            # print(f"\n{i=}\t{j=}\t{i+j=}")
            for k, calc in enumerate(this_calcs_list):
                real_sigma_uvT = calc["sigma_re_u"] - calc["sigma_re_vT"]
                if k % 2 == 0:
                    # print(f"\n{calc["sigma_re_n"]=:.4f}")
                    # print(f"\n{real_sigma_uvT=:.4f}")
                    # sigma_t_vals.append(calc["sigma_re_n"])
                    # sigma_m_vals.append(real_sigma_uvT)
                    avg_mag_vTm = calc["avg_mag_vTm"]
                    if calc["avg_mag_vT"] < calc["avg_mag_model"]:
                        avg_mag_vTm *= -1
                    vT_minus_m_gaussian.append(avg_mag_vTm)
                    real_sigma_t_calculated_gaussian.append(sigma_t)
                    real_sigma_uvT_truth_gaussian.append(sigma_m)
                    real_g_minus_1_truth_gaussian.append(calc["sigma_re_g"])
                else:
                    real_sigma_uvT_skycal_gaussian.append(real_sigma_uvT)
                    real_g_minus_1_skycal_gaussian.append(calc["sigma_re_g"]) 

    vT_minus_m_gaussian              = np.asarray(vT_minus_m_gaussian)
    real_sigma_t_calculated_gaussian = np.asarray(real_sigma_t_calculated_gaussian)
    real_g_minus_1_truth_gaussian    = np.asarray(real_g_minus_1_truth_gaussian)
    real_g_minus_1_skycal_gaussian   = np.asarray(real_g_minus_1_skycal_gaussian)
    real_sigma_uvT_truth_gaussian    = np.asarray(real_sigma_uvT_truth_gaussian)
    real_sigma_uvT_skycal_gaussian   = np.asarray(real_sigma_uvT_skycal_gaussian)
    # print(f"\n\n***METADATA***\n\nLength: {len(metadata)}\n\n{metadata}\n\n")
    # print(f"\n\n***SIGMA T***\n\t{np.min(real_sigma_t_calculated_gaussian)=:.3f}\t{np.max(real_sigma_t_calculated_gaussian)=:.3f}\n{real_sigma_t_calculated_gaussian.shape}")
    # print(f"\n***SIGMA VT-M***\n\t{np.min(vT_minus_m_gaussian)=:.3f}\t\t{np.max(vT_minus_m_gaussian)=:.3f}\n\n")
    # print(f"\n{real_sigma_t_calculated_gaussian=}\n\n{sigma_t_vals=}\n\n{vT_minus_m_gaussian=}\n\n{sigma_m_vals}\n\n")
    if verbose:
        print(f"Plotting truth 2D grid for gain offset")
    dev.plot_3d_data_as_2d_hist(
        x_array     = vT_minus_m_gaussian,
        y_array     = real_sigma_t_calculated_gaussian,
        z_array     = real_g_minus_1_truth_gaussian,
        plot_title  = "Gain Offset vs Model Error\nand Thermal Noise (Truth)",
        plot_xlabel = "$Re(v_T - m)$",
        plot_ylabel = "$\\sigma_t (Re)$",
        plot_vmax   = 0.25,
        plot_vmin   = -0.1,
        filename    = f'{image_path}/{filename_2d_gains}_truth_{file_suffix}_gaussian.png',
        suffix      = file_suffix,
        metadata    = metadata,
    )
    if verbose:
        print(f"Plotting skycal 2D grid for gain offset")
    dev.plot_3d_data_as_2d_hist(
        x_array     = vT_minus_m_gaussian,
        y_array     = real_sigma_t_calculated_gaussian,
        z_array     = real_g_minus_1_skycal_gaussian,
        plot_title  = "Gain Offset vs Model Error\nand Thermal Noise (Skycal)",
        plot_xlabel = "$Re(v_T - m)$",
        plot_ylabel = "$\\sigma_t (Re)$",
        plot_vmax   = 0.25,
        plot_vmin   = -0.1,
        filename    = f'{image_path}/{filename_2d_gains}_skycal_{file_suffix}_gaussian.png',
        suffix      = file_suffix,
        metadata    = metadata,
    )
    if verbose:
        print(f"Plotting truth-skycal diff 2D grid for gain offset")
    real_g_minus_1_diff = np.asarray(real_g_minus_1_truth_gaussian) - \
        np.asarray(real_g_minus_1_skycal_gaussian)
    dev.plot_3d_data_as_2d_hist(
        x_array     = vT_minus_m_gaussian,
        y_array     = real_sigma_t_calculated_gaussian,
        z_array     = real_g_minus_1_diff,
        plot_title  = "Gain Offset (Truth - Skycal) \nvs Model Error and Thermal Noise",
        plot_xlabel = "$Re(v_T - m)$",
        plot_ylabel = "$\\sigma_t (Re)$",
        log_cmap    = True,
        filename    = f'{image_path}/{filename_2d_gains}_diff_{file_suffix}_gaussian.png',
        suffix      = file_suffix,
        metadata    = metadata,
    )
    if verbose:
        print(f"Plotting truth 2D grid for u offset")
    dev.plot_3d_data_as_2d_hist(
        x_array     = vT_minus_m_gaussian,
        y_array     = real_sigma_t_calculated_gaussian,
        z_array     = real_sigma_uvT_truth_gaussian,
        plot_title  = "$\\sigma_u - \\sigma_v$ vs Model Error\nand Thermal Noise (Truth)",
        plot_xlabel = "$Re(v_T - m)$",
        plot_ylabel = "$\\sigma_t (Re)$",
        plot_vmax   = 15,
        plot_vmin   = -5,
        filename    = f'{image_path}/{filename_2d_u_err}_truth_{file_suffix}_gaussian.png',
        plot_cmap   = "inferno",
        suffix      = file_suffix,
        metadata    = metadata,
    )
    if verbose:
        print(f"Plotting skycal 2D grid for u offset")
    dev.plot_3d_data_as_2d_hist(
        x_array     = vT_minus_m_gaussian,
        y_array     = real_sigma_t_calculated_gaussian,
        z_array     = real_sigma_uvT_skycal_gaussian,
        plot_title  = "$\\sigma_u - \\sigma_v$ vs Model Error\nand Thermal Noise (Skycal)",
        plot_xlabel = "$Re(v_T - m)$",
        plot_ylabel = "$\\sigma_t (Re)$",
        plot_vmax   = 15,
        plot_vmin   = -5,
        filename    = f'{image_path}/{filename_2d_u_err}_skycal_{file_suffix}_gaussian.png',
        plot_cmap   = "inferno",
        suffix      = file_suffix,
        metadata    = metadata,
    )
    if verbose:
        print(f"Plotting truth-skycal diff 2D grid for u offset")
    sigma_uvT_diff = np.asarray(real_sigma_uvT_truth_gaussian) - \
        np.asarray(real_sigma_uvT_skycal_gaussian)
    dev.plot_3d_data_as_2d_hist(
        x_array     = vT_minus_m_gaussian,
        y_array     = real_sigma_t_calculated_gaussian,
        z_array     = sigma_uvT_diff,
        plot_title  = "$\\sigma_u - \\sigma_v$ (Truth - Skycal)\nvs Model Error and Thermal Noise",
        plot_xlabel = "$Re(v_T - m)$",
        plot_ylabel = "$\\sigma_t (Re)$",
        plot_vmax   = 15,
        plot_vmin   = -5,
        filename    = f'{image_path}/{filename_2d_u_err}_diff_{file_suffix}_gaussian.png',
        plot_cmap   = "inferno",
        suffix      = file_suffix,
        metadata    = metadata,
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
        zlim_hi           = 0.15,  zlim_lo     = -0.15,
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

# TODO: Add args for git and time IDs
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
    args = parser.parse_args()

    main(
        calibrate=args.c, 
        verbose=args.v, 
        show_plot=args.s,
        time_id=args.time,
        git_id=args.git,
    )