import numpy as np

import dev_tools as dev

import subprocess
import pickle
import json
import os
import argparse
import time

"""
    Test grid of model error and thermal noise values
    in calibration to find gain offset from Re(g)=1
    and sigma(u) - sigma(v_T)
"""
def main(calibrate : bool = True, 
         verbose   : bool = False,
         show_plot : bool = False,
) -> None:
    data_path  = 'calico/data'
    image_path = 'calico/images'

    if calibrate:
        scaling_factors = [0.1, 1]  # skycal and truth
        sigma_t_scales  = np.arange(0,30,24/6)
        sigma_m_scales  = np.arange(-30,15,24/6)

        vT_minus_m_gaussian              = []
        real_sigma_t_calculated_gaussian = []
        real_g_minus_1_truth_gaussian    = []
        real_g_minus_1_skycal_gaussian   = []
        real_sigma_uvT_truth_gaussian    = []
        real_sigma_uvT_skycal_gaussian   = []

        if verbose:
            print("Beginning sigma loops")

        start_time = str(time.time())[:4]
        git_hash = str(subprocess.check_output(['git','rev-parse','--short','HEAD']).decode('ascii').strip())

        for sigma_t in sigma_t_scales:
            if np.abs(sigma_t) - 0.0 < 1e-5:
                sigma_t = 0.1
            for sigma_m in sigma_m_scales:
                if np.abs(sigma_m) - 0.0 < 1e-5:
                    sigma_m = 0.1

                if verbose:
                    print(f"Creating settings files\n\tsigma_m {sigma_m}\tsigma_t {sigma_t}")
                suffix = f"{int(sigma_t*100):d}_{int(sigma_m*10):d}"
                suffix += f"g{git_hash}_t{start_time}"
                filename = f"gain_error_offset_analysis_{suffix}",
                __import__('make_run_params').generate_custom_file(
                    filename             = filename,
                    scaling_factors      = scaling_factors,
                    sigma_t              = sigma_t,
                    sigma_n              = sigma_t,
                    sigma_m              = sigma_m,
                    sigma_e              = sigma_m,
                    thermal_realizations = 1,
                    model_realizations   = 1,
                )
                custom_file = []
                for scaling_factor in scaling_factors:
                    scaling_factor_cost = 1 / scaling_factor**2
                    custom_file.append(
                        {
                            "sigma_t"                    : sigma_t,
                            "sigma_n"                    : sigma_t,
                            "sigma_m"                    : sigma_m,
                            "sigma_e"                    : sigma_m,
                            "thermal_noise_realizations" : 1,
                            "model_error_realizations"   :   1,
                            "weighting_function"         : "constant_weights",
                            "scaling_factor_cost"        : scaling_factor_cost,
                            "scaling_factor_sim"         : 1,
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
                __import__('many_realizations_study').init_many_realizations(
                    fhd_prefix                   = '1061316296_',
                    sav_data_filename            = 'tutorial_full_onetime_unflagged',
                    sav_model_filename           = 'tutorial_full_onetime_unflagged',
                    run_params_filename          = f'gain_error_offset_analysis_{suffix}_settings',
                    vis_data_writeout_filename   = 'tutorial_full_onetime_unflagged',
                    model_data_writeout_filename = 'tutorial_full_onetime_unflagged',
                    verbose                      = True,
                    simulate_visibilities        = True,
                    calibrate                    = True,
                    reconstruct_data             = False,
                    reconstruct_model            = False,
                )

                if verbose:
                    print("Finished realizations, reading in calculations")
                with open(f'{data_path}/gain_error_offset_analysis_output_calcs.pkl', 'rb') as file:
                    output_calcs_list = pickle.load(file)

                if verbose:
                    print("Collecting skycal and truth values")
                for i, calc in enumerate(output_calcs_list):
                    print(f"i\t{i}\ttype(calc)\t{type(calc)}")
                    real_sigma_uvT = calc["sigma_re_u"] - calc["sigma_re_vT"]
                    if i % 2 == 0:
                        avg_mag_vTm = calc["avg_mag_vTm"]
                        if calc["avg_mag_vT"] < calc["avg_mag_model"]:
                            avg_mag_vTm *= -1
                        vT_minus_m_gaussian.append(avg_mag_vTm)
                        real_sigma_t_calculated_gaussian.append(calc["sigma_re_n"])
                        real_sigma_uvT_truth_gaussian.append(real_sigma_uvT)
                        real_g_minus_1_truth_gaussian.append(calc["sigma_re_g"])
                    else:
                        real_sigma_uvT_skycal_gaussian.append(real_sigma_uvT)
                        real_g_minus_1_skycal_gaussian.append(calc["sigma_re_g"]) 
                                   
                if verbose:
                    print("Cleaning up calculated saved files")
                os.system(f"rm {data_path}/gain_error_offset_analysis_{suffix}_run_params.json")
                os.system(f"rm {data_path}/gain_error_offset_analysis_output_calcs.json")
                
        if verbose: 
            print("Writing out run-aggregate files...")
        with open(f'{data_path}/vT_minus_m_{suffix}.json', 'w') as file:
            json.dump(vT_minus_m_gaussian, file)
        with open(f'{data_path}/real_sigma_t_calculated_gaussian_{suffix}.json', 'w') as file:
            json.dump(real_sigma_t_calculated_gaussian, file)
        with open(f'{data_path}/real_sigma_uvT_truth_gaussian_{suffix}.json', 'w') as file:
            json.dump(real_sigma_uvT_truth_gaussian, file)
        with open(f'{data_path}/real_sigma_uvT_skycal_gaussian_{suffix}.json', 'w') as file:
            json.dump(real_sigma_uvT_skycal_gaussian, file)
        with open(f'{data_path}/real_g_minus_1_truth_gaussian_{suffix}.json', 'w') as file:
            json.dump(real_g_minus_1_truth_gaussian, file)
        with open(f'{data_path}/real_g_minus_1_skycal_gaussian_{suffix}.json', 'w') as file:
            json.dump(real_g_minus_1_skycal_gaussian, file)
        if verbose: 
            print("Done.")

        if verbose:
            print("Calibration tests done.")

    else:
        if verbose:
            print("Reading in run-aggregate files...")
        with open(f'{data_path}/vT_minus_m_{suffix}.json', 'r') as file:
            vT_minus_m_gaussian = json.load(file)
        with open(f'{data_path}/real_sigma_t_calculated_gaussian_{suffix}.json', 'r') as file:
            real_sigma_t_calculated_gaussian = json.load(file)
        with open(f'{data_path}/real_sigma_uvT_truth_gaussian_{suffix}.json', 'r') as file:
            real_sigma_uvT_truth_gaussian = json.load(file)
        with open(f'{data_path}/real_sigma_uvT_skycal_gaussian_{suffix}.json', 'r') as file:
            real_sigma_uvT_skycal_gaussian = json.load(file)
        with open(f'{data_path}/real_g_minus_1_truth_gaussian_{suffix}.json', 'r') as file:
            real_g_minus_1_truth_gaussian = json.load(file)
        with open(f'{data_path}/real_g_minus_1_skycal_gaussian_{suffix}.json', 'r') as file:
            real_g_minus_1_skycal_gaussian = json.load(file)

    filename_2d_gains = 'gain_error_vs_model_error_vs_thermal_noise_2d'
    filename_2d_u_err = 'u_error_vs_model_error_vs_thermal_noise_2d'

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
        filename    = f'{image_path}/{filename_2d_gains}_truth_{suffix}_gaussian.png',
        suffix      = suffix,
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
        filename    = f'{image_path}/{filename_2d_gains}_skycal_{suffix}_gaussian.png',
        suffix      = suffix,
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
        filename    = f'{image_path}/{filename_2d_gains}_diff_{suffix}_gaussian.png',
        suffix      = suffix,
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
        filename    = f'{image_path}/{filename_2d_u_err}_truth_{suffix}_gaussian.png',
        plot_cmap   = "inferno",
        suffix      = suffix,
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
        filename    = f'{image_path}/{filename_2d_u_err}_skycal_{suffix}_gaussian.png',
        plot_cmap   = "inferno",
        suffix      = suffix,
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
        filename    = f'{image_path}/{filename_2d_u_err}_diff_{suffix}_gaussian.png',
        plot_cmap   = "inferno",
        suffix      = suffix,
    )
    if verbose:
        print("Creating 3D plot.")
    filename_3d_scatter = 'gain_error_vs_model_error_vs_thermal_noise_3d'
    dev.build_3d_scatter_plot(
        x_array     = vT_minus_m_gaussian,
        y_array     = real_sigma_t_calculated_gaussian,
        z_array     = real_g_minus_1_truth_gaussian,
        z_array_2   = real_g_minus_1_skycal_gaussian,
        second_plot = True,
        show_plot   = show_plot,
        plot_title  = "Gain Offset vs Model Error and Thermal Noise",
        plot_xlabel = "$Re(v_T - m)$",
        plot_ylabel = "$\\sigma_t (Re)$", 
        plot_zlabel = "Re(g-1)",
        xlim_hi     = 16, xlim_lo     = -36,
        ylim_hi     = 16, ylim_lo     = -1,
        zlim_hi     = 1,  zlim_lo     = -1,
        filename    = f'{image_path}/{filename_3d_scatter}_{suffix}_gaussian.png',
        suffix      = suffix,
        # metadata    =
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Gain Error Offset Analysis")
    parser.add_argument("-c", action="store_true")
    parser.add_argument("-v", action="store_true")
    parser.add_argument("-s", action="store_true")
    args = parser.parse_args()
    main(calibrate=args.c, verbose=args.v, show_plot=args.s)