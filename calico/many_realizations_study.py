import caldata
import os, os.path
import pyuvdata as uv
import numpy as np
import matplotlib.pyplot as plt
import time

import dev_tools
import sys
from pyuvdata import UVData, UVFlag
import noise_and_error_simulation as sim
from pyuvdata import UVFlag
import make_run_params

# def update_calico()

def display_all_images():

    path = os.path.abspath(os.getcwd()) + '/images/'
    files = [name for name in os.listdir('./images') if not os.path.isdir(os.path.join(path, name))]
    # print("Files:",files)
    for file in files:
        if file[0] != '.':
            img = plt.imread("./images/"+file)
            _ = plt.imshow(img)
            plt.axis('off')
            plt.tight_layout()
            plt.show()

def examine_flags(uvd):
    print("\n***beginning flag waterfall***")
    uvf = UVFlag(uvd)
    uvf.to_waterfall()
    uvf.to_flag()
    print(f"***all flagged?***\n\t{np.all(uvf.flag_array == True)}\n")
    print(f"***any flagged?***\n\t{np.any(uvf.flag_array == True)}\n")

    plt.pcolormesh(np.squeeze(uvf.flag_array[:,:,0]))
    plt.title("Waterfall of Flag Array (uvf)")
    plt.ylabel("Time")
    plt.xlabel("Frequency")
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.savefig("calico/images/flag_watterfall_uvf.png")
    plt.close()

    plt.pcolormesh(np.squeeze(uvd.flag_array[:,:,0]))
    plt.title("Waterfall-ish of Flag Array (uvd)")
    plt.ylabel("Blts")
    plt.xlabel("Frequency")
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.savefig("calico/images/flag_watterfall_uvd.png")
    plt.close()
    
    print("***finished with flag watefall***\n")

def prepare_data_files(
    fhd_prefix = None,
    sav_data_filename = None,
    sav_model_filename = None,
    vis_data_writeout_filename = None,
    model_data_writeout_filename = None,
    reconstruct_data = False,
    reconstruct_model = False,
):
    if fhd_prefix is None:
        print("ERROR: FHD prefix is missing")
        return -1
    if sav_data_filename is None:
        print("ERROR: SAV data filename is missing")
        return -1
    if sav_model_filename is None:
        print("ERROR: SAV model filename is missing")
        return -1
    if vis_data_writeout_filename is None:
        print("ERROR: uvfits data filename is missing")
        return -1
    if model_data_writeout_filename is None:
        print("ERROR: uvfits model filename is missing")
        return -1

    sav_data_path = os.getcwd() + f'/calico/data/{sav_data_filename}'
    uv_data_path = os.getcwd() + f'/calico/data/{vis_data_writeout_filename}.uvfits'
    print("uv data path", uv_data_path)
    if fhd_prefix[-1] != '_':
        fhd_prefix += '_'
    freq_ind = 379  # null init values
    time_ind = 298

    if os.path.isfile(uv_data_path) and not reconstruct_data:
        print("Data uvits file exists - skipping")
    else:
        print("Data uvfits file not found - creating")
        # Set up the files we need
        data_vis_files = os.path.join(sav_data_path, "vis_data", fhd_prefix + "vis_model_XX.sav")
        data_flags_file = os.path.join(sav_data_path, "vis_data", fhd_prefix + "flags.sav")
        data_layout_file = os.path.join(sav_data_path, "metadata", fhd_prefix + "layout.sav")
        data_params_file = os.path.join(sav_data_path, "metadata", fhd_prefix + "params.sav")
        data_settings_file = os.path.join(sav_data_path, "metadata", fhd_prefix + "settings.txt")

        uvd_data = UVData.from_file(
            data_vis_files,
            flags_file=data_flags_file,
            layout_file=data_layout_file,
            params_file=data_params_file,
            settings_file=data_settings_file,
        )

        # examine_flags(uvd_data)

        # exclude autos (invert select on ant1=ant2)
        uvd_data.select(ant_str='auto', invert=True)

        # select on one frequency (for now)
        uvd_data.select(frequencies=[uvd_data.freq_array[1]])

        # keep only one time (for now)
        uvd_data.select(times=[uvd_data.time_array[uvd_data.Nbls*2]])

        print(f"\n***all flagged? before***\n\t{np.all(uvd_data.flag_array == True)}")
        print(f"\n***any flagged? before***\n\t{np.any(uvd_data.flag_array == True)}")

        # remove flagged data (need to handle in unical code in future)
        print(f"\n***shape before removing flags***\n\t{uvd_data.data_array.shape}")
        flagged_bls_data = np.nonzero(np.squeeze(uvd_data.flag_array))[0]
        print(f"\n***flagged bls shape***\n\t{flagged_bls_data.shape}")
        uvd_data.select(blt_inds=flagged_bls_data, invert=True)
        print(f"\n***shape after removing flags***\n\t{uvd_data.data_array.shape}\n")
        print(f"\n***all flagged? after***\n\t{np.all(uvd_data.flag_array == True)}")
        print(f"\n***any flagged? after***\n\t{np.any(uvd_data.flag_array == True)}")
        uvd_data.write_uvfits(uv_data_path)

    sav_model_path = os.getcwd() + f'/calico/data/{sav_model_filename}'
    uv_model_path = os.getcwd() + f'/calico/data/{model_data_writeout_filename}.uvfits'

    if os.path.isfile(uv_model_path) and not reconstruct_model:
        print("Model uvfits file exists - skipping")
    else:
        print("Model uvfits file not found - creating")
        # Set up the files we need
        model_vis_files = os.path.join(sav_model_path, "vis_data", fhd_prefix + "vis_model_XX.sav")
        model_flags_file = os.path.join(sav_model_path, "vis_data", fhd_prefix + "flags.sav")
        model_layout_file = os.path.join(sav_model_path, "metadata", fhd_prefix + "layout.sav")
        model_params_file = os.path.join(sav_model_path, "metadata", fhd_prefix + "params.sav")
        model_settings_file = os.path.join(sav_model_path, "metadata", fhd_prefix + "settings.txt")

        uvd_model = UVData.from_file(
            model_vis_files,
            flags_file=model_flags_file,
            layout_file=model_layout_file,
            params_file=model_params_file,
            settings_file=model_settings_file,
        )
        
        # probably full repeat selections from before
        uvd_model.select(ant_str='auto', invert=True)
        uvd_model.select(frequencies=[uvd_model.freq_array[1]])
        uvd_model.select(times=[uvd_model.time_array[uvd_model.Nbls*2]])
        flagged_bls_model = np.nonzero(np.squeeze(uvd_model.flag_array))[0]
        uvd_model.select(blt_inds=flagged_bls_model, invert=True)
        uvd_model.write_uvfits(uv_model_path)
        print(f"\n***shape after removing flags***\n\t{uvd_model.data_array.shape}\n")

    return 1

def init_many_realizations(
    fhd_prefix = '1061316296_',
    sav_data_filename = 'tutorial_full_onetime_unflagged',                  # sav directory name (gaussian sim)
    sav_model_filename = 'tutorial_full_onetime_unflagged',                 # sav directory name (gaussian sim)
    run_params_filename = 'baseline_dependence_runs_large_noise',
    vis_data_writeout_filename = 'tutorial_full_onetime_unflagged',     # uvfits filename (gaussian sim) 
    model_data_writeout_filename = 'tutorial_full_onetime_unflagged',   # uvfits filename (using FHD)
    verbose=True,
    simulate_visibilities=False,
    calibrate=True,
    reconstruct_data=False,
    reconstruct_model=False,
    metadata=None,
    suffix="",
    optimization_scheme="powell",
    calibration_type="unical",
    gains_multiply_model=False,
    threshold_length=None,
    force_fit_to_true_vis=False,
    gains_real_guess=None,
):
    dev = dev_tools.DevTools()
    if prepare_data_files(
        fhd_prefix=fhd_prefix,
        sav_data_filename=sav_data_filename,
        sav_model_filename=sav_model_filename,
        vis_data_writeout_filename=vis_data_writeout_filename,
        model_data_writeout_filename=model_data_writeout_filename,
        reconstruct_data=reconstruct_data,
        reconstruct_model=reconstruct_model,
    ) > 0:
        if threshold_length == None:
            raise ValueError(f"Need threshold length even if zero -- Init Many Realizations")
        __import__('make_run_params').generate_files()
        model_path = os.getcwd() + f'/calico/data/{model_data_writeout_filename}'
        if calibrate:
            if verbose:
                data_read_start_time = time.time()
            data_file_path = os.getcwd() + f'/calico/data/{vis_data_writeout_filename}.uvfits'
            model_file_path = os.getcwd() + f'/calico/data/{model_data_writeout_filename}.uvfits'
            print_data_read_time = False
            if isinstance(data_file_path, str):  # Read data
                data = UVData()
                data.read_uvfits(data_file_path)
                print_data_read_time = True
            if isinstance(model_file_path, str):  # Read model
                model = UVData()
                model.read_uvfits(model_file_path)
                print_data_read_time = True
            # Ensure data and model are phased the same
            data.phase_to_time(np.mean(data.time_array))
            model.phase_to_time(np.mean(data.time_array))
            if verbose:
                if print_data_read_time:
                    print(
                        f"Done. Data read time {(time.time() - data_read_start_time)/60.} minutes."
                    )
                print("Formatting data...")
                sys.stdout.flush()
                data_format_start_time = time.time()
            caldata_obj = caldata.CalData()
            caldata_obj.load_data(
                data,
                model,
                gain_init_calfile=None,
                gain_init_to_vis_ratio=True,
                gains_multiply_model=gains_multiply_model,
                gain_init_stddev=0.0,
                glim=None,
                ulim=None,
                weighting_function="constant_weights",
                simulate_visibilities=simulate_visibilities,
                sigma_t_0=1,
                sigma_m_0=1,
                scaling_factor_cost=1,
                threshold_length=0,
            )
            if verbose:
                print(
                    f"Done. Data formatting time {(time.time() - data_format_start_time)/60.} minutes."
                )
                print("Running calibration optimization...")
                sys.stdout.flush()
                optimization_start_time = time.time()
            # calwrap.unified_calibration_wrapper(
            #     data=vis_data_writeout_filename,
            #     model=model_data_writeout_filename,
            #     parallel=False,
            #     verbose=verbose,
            #     glim=None,
            #     ulim=None,
            #     antenna_gain_weights=None,
            #     model_baseline_weights=None,
            #     threshold_length=100,
            #     weighting_function='constant_weights',
            #     sigma_t_0=1,
            #     sigma_m_0=1,
            #     many_realizations=True,
            #     run_params_filename=run_params_filename,
            #     scaling_factor_cost=1,
            #     simulate_visibilties=simulate_visibilities,
            #     metadata=metadata,
            #     suffix=suffix,
            #     optimization_scheme=optimization_scheme,
            #     calibration_type=calibration_type,
            #     gains_multiply_model=True,
            # )
            dev = dev_tools.DevTools()
            xtol = 1e-5
            maxiter = 200
            antenna_flagging_iterations = 1
            # if calibration_type == "skycal":
            #     for ant_flag_iter in range(antenna_flagging_iterations):
            #         caldata_obj.sky_based_calibration(
            #             xtol=xtol / 10,  # Lower tolerance for antenna flagging
            #             maxiter=int(maxiter / 2),  # Lower maxiter for antenna flagging
            #             get_crosspol_phase=False,  # No crosspol phase needed for antenna flagging
            #             parallel=False,
            #             verbose=verbose,
            #             pool=None,
            #         )
            #         if verbose:
            #             print(f"Initial calibration optimization done.", end="")
            #             print(f"Antenna flagging iteration {ant_flag_iter+1} of {antenna_flagging_iterations}.")
            #             print(f"Optimization time: {caldata_obj.Nfreqs} frequency channels", end="")
            #             print(f"in {(time.time() - optimization_start_time)/60.} minutes.")
            #             sys.stdout.flush()
                    # caldata_obj.flag_antennas_from_per_ant_cost(
                    #     flagging_threshold=2.5,
                    #     parallel=False,
                    #     pool=None,
                    #     verbose=verbose,
                    # )
            dev.calculate_many_realizations(
                caldata_obj=caldata_obj,
                example_data=data,
                verbose=verbose,
                vis_data_writeout_filename=vis_data_writeout_filename,
                model_data_writeout_filename=model_data_writeout_filename,
                run_params_filename=run_params_filename,
                suffix=suffix,
                metadata=metadata,
                optimization_scheme=optimization_scheme,
                calibration_type=calibration_type,
                xtol=xtol,
                maxiter=maxiter,
                force_fit_to_true_vis=force_fit_to_true_vis,
                gains_real_guess=gains_real_guess,
            )
        dev.plot_many_realizations(data_filepath=model_path,
                                   run_params_filename=run_params_filename,
                                   metadata=metadata,
                                   suffix=suffix,)

    else:
        print("Problem with data files - exiting")

# init_many_realizations(
#     fhd_prefix = '1061316296_',
#     sav_data_filename = 'tutorial_full_onetime_unflagged',                  # sav directory name (gaussian sim)
#     sav_model_filename = 'tutorial_full_onetime_unflagged',                 # sav directory name (gaussian sim)
#     run_params_filename = 'baseline_dependence_runs_large_noise',
#     vis_data_writeout_filename = 'tutorial_full_onetime_unflagged',         # uvfits filename (gaussian sim) 
#     model_data_writeout_filename = 'tutorial_full_onetime_unflagged',      # uvfits filename (using FHD)
#     verbose=False,
#     simulate_visibilities=False,
#     calibrate=True,
# )

# init_many_realizations(
#     fhd_prefix = '1061316296_',
#     sav_data_filename = 'tutorial_full_onetime_unflagged',                  # sav directory name (gaussian sim)
#     sav_model_filename = 'tutorial_full_onetime_unflagged',                 # sav directory name (gaussian sim)
#     run_params_filename = 'baseline_dependence_runs_large_noise',
#     vis_data_writeout_filename = 'tutorial_full_onetime_unflagged',         # uvfits filename (gaussian sim) 
#     model_data_writeout_filename = 'tutorial_full_onetime_unflagged',       # uvfits filename (using FHD)
#     verbose=False,
#     simulate_visibilities=True,
#     calibrate=True,
# )

# init_many_realizations(
#     fhd_prefix = '1061316296_',
#     sav_data_filename = 'tutorial_full_onetime_unflagged',                  # sav directory name (gaussian sim)
#     sav_model_filename = 'tutorial_full_onetime_unflagged',                 # sav directory name (gaussian sim)
#     run_params_filename = 'baseline_dependence_runs_small_noise',
#     vis_data_writeout_filename = 'tutorial_full_onetime_unflagged',         # uvfits filename (gaussian sim) 
#     model_data_writeout_filename = 'tutorial_full_onetime_unflagged',       # uvfits filename (using FHD)
#     verbose=False,
#     simulate_visibilities=False,
#     calibrate=True,
# )

# init_many_realizations(
#     fhd_prefix = '1061316296_',
#     sav_data_filename = 'tutorial_full_onetime_unflagged',                  # sav directory name (gaussian sim)
#     sav_model_filename = 'tutorial_full_onetime_unflagged',                 # sav directory name (gaussian sim)
#     run_params_filename = 'baseline_dependence_runs_small_noise',
#     vis_data_writeout_filename = 'tutorial_full_onetime_unflagged',       # uvfits filename (gaussian sim) 
#     model_data_writeout_filename = 'tutorial_full_onetime_unflagged',     # uvfits filename (using FHD)
#     verbose=False,
#     simulate_visibilities=True,
#     calibrate=True,
# )

"""
    Below FHD runs had model and data swapped
"""

# init_many_realizations(
#     fhd_prefix = '1061316296_',
#     sav_data_filename = 'fhd_data',                                     # sav directory name (using FHD)
#     sav_model_filename = 'fhd_model_01',                                   # sav directory name (using FHD)
#     run_params_filename = 'fhd_runs_medium_noise_01',
#     model_data_writeout_filename = 'fhd_data_one_freq',                   # uvfits filename (using FHD)
#     vis_data_writeout_filename = 'fhd_model_one_freq_01',                # uvfits filename (using FHD)
#     verbose=True,
#     simulate_visibilities=False,
#     calibrate=False,
#     reconstruct_data=False,
#     reconstruct_model=False,
# )

# init_many_realizations(
#     fhd_prefix = '1061316296_',
#     sav_data_filename = 'fhd_data',                                     # sav directory name (using FHD)
#     sav_model_filename = 'fhd_model_015',                                   # sav directory name (using FHD)
#     run_params_filename = 'fhd_runs_medium_noise_015',
#     model_data_writeout_filename = 'fhd_data_one_freq',                   # uvfits filename (using FHD)
#     vis_data_writeout_filename = 'fhd_model_one_freq_015',                # uvfits filename (using FHD)
#     verbose=True,
#     simulate_visibilities=False,
#     calibrate=False,
#     reconstruct_data=False,
#     reconstruct_model=False,
# )

# init_many_realizations(
#     fhd_prefix = '1061316296_',
#     sav_data_filename = 'fhd_data',                                     # sav directory name (using FHD)
#     sav_model_filename = 'fhd_model_05',                                   # sav directory name (using FHD)
#     run_params_filename = 'fhd_runs_medium_noise_05',
#     model_data_writeout_filename = 'fhd_data_one_freq',                   # uvfits filename (using FHD)
#     vis_data_writeout_filename = 'fhd_model_one_freq_05',                # uvfits filename (using FHD)
#     verbose=True,
#     simulate_visibilities=False,
#     calibrate=False,
#     reconstruct_data=False,
#     reconstruct_model=False,
# )

# init_many_realizations(
#     fhd_prefix = '1061316296_',
#     sav_data_filename = 'fhd_data',                                     # sav directory name (using FHD)
#     sav_model_filename = 'fhd_model_1',                                   # sav directory name (using FHD)
#     run_params_filename = 'fhd_runs_medium_noise_1',
#     model_data_writeout_filename = 'fhd_data_one_freq',                   # uvfits filename (using FHD)
#     vis_data_writeout_filename = 'fhd_model_one_freq_1',                # uvfits filename (using FHD)
#     verbose=True,
#     simulate_visibilities=False,
#     calibrate=False,
#     reconstruct_data=False,
#     reconstruct_model=False,
# )

# init_many_realizations(
#     fhd_prefix = '1061316296_',
#     sav_data_filename = 'fhd_data',                                     # sav directory name (using FHD)
#     sav_model_filename = 'fhd_model_01',                                   # sav directory name (using FHD)
#     run_params_filename = 'fhd_runs_small_noise_01',
#     model_data_writeout_filename = 'fhd_data_one_freq',                   # uvfits filename (using FHD)
#     vis_data_writeout_filename = 'fhd_model_one_freq_01',                # uvfits filename (using FHD)
#     verbose=True,
#     simulate_visibilities=False,
#     calibrate=False,
#     reconstruct_data=False,
#     reconstruct_model=False,
# )

# init_many_realizations(
#     fhd_prefix = '1061316296_',
#     sav_data_filename = 'fhd_data',                                     # sav directory name (using FHD)
#     sav_model_filename = 'fhd_model_015',                                   # sav directory name (using FHD)
#     run_params_filename = 'fhd_runs_small_noise_015',
#     model_data_writeout_filename = 'fhd_data_one_freq',                   # uvfits filename (using FHD)
#     vis_data_writeout_filename = 'fhd_model_one_freq_015',                # uvfits filename (using FHD)
#     verbose=True,
#     simulate_visibilities=False,
#     calibrate=True,
#     reconstruct_data=False,
#     reconstruct_model=False,
# )

# init_many_realizations(
#     fhd_prefix = '1061316296_',
#     sav_data_filename = 'fhd_data',                                     # sav directory name (using FHD)
#     sav_model_filename = 'fhd_model_05',                                   # sav directory name (using FHD)
#     run_params_filename = 'fhd_runs_small_noise_05',
#     model_data_writeout_filename = 'fhd_data_one_freq',                   # uvfits filename (using FHD)
#     vis_data_writeout_filename = 'fhd_model_one_freq_05',                # uvfits filename (using FHD)
#     verbose=True,
#     simulate_visibilities=False,
#     calibrate=False,
#     reconstruct_data=False,
#     reconstruct_model=False,
# )

# init_many_realizations(
#     fhd_prefix = '1061316296_',
#     sav_data_filename = 'fhd_data',                                     # sav directory name (using FHD)
#     sav_model_filename = 'fhd_model_1',                                   # sav directory name (using FHD)
#     run_params_filename = 'fhd_runs_small_noise_1',
#     model_data_writeout_filename = 'fhd_data_one_freq',                   # uvfits filename (using FHD)
#     vis_data_writeout_filename = 'fhd_model_one_freq_1',                # uvfits filename (using FHD)
#     verbose=True,
#     simulate_visibilities=False,
#     calibrate=False,
#     reconstruct_data=False,
#     reconstruct_model=False,
# )

# init_many_realizations(
#     fhd_prefix = '1061316296_',
#     sav_data_filename = 'fhd_data',                                     # sav directory name (using FHD)
#     sav_model_filename = 'fhd_model_1',                                   # sav directory name (using FHD)
#     run_params_filename = 'fhd_runs_large_noise_01',
#     vis_data_writeout_filename = 'fhd_data_one_freq',                   # uvfits filename (using FHD)
#     model_data_writeout_filename = 'fhd_model_one_freq_1',                # uvfits filename (using FHD)
#     verbose=True,
#     simulate_visibilities=False,
#     calibrate=True,
#     reconstruct_data=False,
#     reconstruct_model=False,
# )

# init_many_realizations(
#     fhd_prefix = '1061316296_',
#     sav_data_filename = 'fhd_data',                                     # sav directory name (using FHD)
#     sav_model_filename = 'fhd_model_1',                                   # sav directory name (using FHD)
#     run_params_filename = 'fhd_runs_large_noise_015',
#     vis_data_writeout_filename = 'fhd_data_one_freq',                   # uvfits filename (using FHD)
#     model_data_writeout_filename = 'fhd_model_one_freq_1',                # uvfits filename (using FHD)
#     verbose=True,
#     simulate_visibilities=False,
#     calibrate=True,
#     reconstruct_data=False,
#     reconstruct_model=False,
# )

# init_many_realizations(
#     fhd_prefix = '1061316296_',
#     sav_data_filename = 'fhd_data',                                     # sav directory name (using FHD)
#     sav_model_filename = 'fhd_model_1',                                   # sav directory name (using FHD)
#     run_params_filename = 'fhd_runs_large_noise_05',
#     vis_data_writeout_filename = 'fhd_data_one_freq',                   # uvfits filename (using FHD)
#     model_data_writeout_filename = 'fhd_model_one_freq_1',                # uvfits filename (using FHD)
#     verbose=True,
#     simulate_visibilities=False,
#     calibrate=True,
#     reconstruct_data=False,
#     reconstruct_model=False,
# )

# init_many_realizations(
#     fhd_prefix = '1061316296_',
#     sav_data_filename = 'fhd_data',                                     # sav directory name (using FHD)
#     sav_model_filename = 'fhd_model_1',                                   # sav directory name (using FHD)
#     run_params_filename = 'fhd_runs_large_noise_1',
#     vis_data_writeout_filename = 'fhd_data_one_freq',                   # uvfits filename (using FHD)
#     model_data_writeout_filename = 'fhd_model_one_freq_1',                # uvfits filename (using FHD)
#     verbose=True,
#     simulate_visibilities=False,
#     calibrate=True,
#     reconstruct_data=False,
#     reconstruct_model=False,
# )

# init_many_realizations(
#     fhd_prefix = '1061316296_',
#     sav_data_filename = 'tutorial_full_onetime_unflagged',                  # sav directory name (gaussian sim)
#     sav_model_filename = 'tutorial_full_onetime_unflagged',                 # sav directory name (gaussian sim)
#     run_params_filename = 'add_gaussian_error_large_noise',
#     vis_data_writeout_filename = 'tutorial_full_onetime_unflagged',         # uvfits filename (gaussian sim) 
#     model_data_writeout_filename = 'tutorial_full_onetime_unflagged',       # uvfits filename (using FHD)
#     verbose=True,
#     simulate_visibilities=True,
#     calibrate=True,
#     reconstruct_data=False,
#     reconstruct_model=False,
# )

# init_many_realizations(
#     fhd_prefix = '1061316296_',
#     sav_data_filename = 'tutorial_full_onetime_unflagged',                  # sav directory name (gaussian sim)
#     sav_model_filename = 'tutorial_full_onetime_unflagged',                 # sav directory name (gaussian sim)
#     run_params_filename = 'add_gaussian_error_medium_noise',
#     vis_data_writeout_filename = 'tutorial_full_onetime_unflagged',         # uvfits filename (gaussian sim) 
#     model_data_writeout_filename = 'tutorial_full_onetime_unflagged',       # uvfits filename (using FHD)
#     verbose=True,
#     simulate_visibilities=True,
#     calibrate=True,
#     reconstruct_data=False,
#     reconstruct_model=False, 
# )

# init_many_realizations(
#     fhd_prefix = '1061316296_',
#     sav_data_filename = 'tutorial_full_onetime_unflagged',                  # sav directory name (gaussian sim)
#     sav_model_filename = 'tutorial_full_onetime_unflagged',                 # sav directory name (gaussian sim)
#     run_params_filename = 'add_gaussian_error_small_noise',
#     vis_data_writeout_filename = 'tutorial_full_onetime_unflagged',         # uvfits filename (gaussian sim) 
#     model_data_writeout_filename = 'tutorial_full_onetime_unflagged',       # uvfits filename (using FHD)
#     verbose=True,
#     simulate_visibilities=True,
#     calibrate=True,
#     reconstruct_data=False,
#     reconstruct_model=False,
# )