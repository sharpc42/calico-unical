import os
import os.path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from calico import caldata
from calico.dev import dev_tools, make_run_params
from pyuvdata import UVData, UVFlag

# def update_calico()


def display_all_images():

    path = os.path.abspath(os.getcwd()) + "/images/"
    files = [
        name
        for name in os.listdir("./images")
        if not os.path.isdir(os.path.join(path, name))
    ]
    # print("Files:",files)
    for file in files:
        if file[0] != ".":
            img = plt.imread("./images/" + file)
            _ = plt.imshow(img)
            plt.axis("off")
            plt.tight_layout()
            plt.show()


def examine_flags(uvd):
    print("\n***beginning flag waterfall***")
    cwd = os.getcwd()
    uvf = UVFlag(uvd)
    uvf.to_waterfall()
    uvf.to_flag()
    print(f"***all flagged?***\n\t{np.all(uvf.flag_array)}\n")
    print(f"***any flagged?***\n\t{np.any(uvf.flag_array)}\n")

    plt.pcolormesh(np.squeeze(uvf.flag_array[:, :, 0]))
    plt.title("Waterfall of Flag Array (uvf)")
    plt.ylabel("Time")
    plt.xlabel("Frequency")
    plt.gca().invert_yaxis()
    plt.colorbar()
    image_path = os.path.join(cwd, "calico", "images", "flag_watterfall_uvf.png")
    plt.savefig(image_path)
    plt.close()

    plt.pcolormesh(np.squeeze(uvd.flag_array[:, :, 0]))
    plt.title("Waterfall-ish of Flag Array (uvd)")
    plt.ylabel("Blts")
    plt.xlabel("Frequency")
    plt.gca().invert_yaxis()
    plt.colorbar()
    image_path = os.path.join(cwd, "calico", "images", "flag_watterfall_uvd.png")
    plt.savefig(image_path)
    plt.close()

    print("***finished with flag watefall***\n")


def prepare_data_files(
    fhd_prefix=None,
    sav_data_filename=None,
    sav_model_filename=None,
    vis_data_writeout_filename=None,
    model_data_writeout_filename=None,
    reconstruct_data=False,
    reconstruct_model=False,
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
    cwd = os.getcwd()
    sav_data_path = os.path.join(cwd, "calico", "data", f"{sav_data_filename}")
    uv_data_path = os.path.join(
        cwd, "calico", "data", f"{vis_data_writeout_filename}.uvfits"
    )
    print("uv data path", uv_data_path)
    if fhd_prefix[-1] != "_":
        fhd_prefix += "_"
    # null init values
    freq_ind = 379
    # time_ind = 298

    if os.path.isfile(uv_data_path) and not reconstruct_data:
        print("Data uvits file exists - skipping")
    else:
        print("Data uvfits file not found - creating")
        # Set up the files we need
        data_vis_files = os.path.join(
            sav_data_path, "vis_data", fhd_prefix + "vis_model_XX.sav"
        )
        data_flags_file = os.path.join(
            sav_data_path, "vis_data", fhd_prefix + "flags.sav"
        )
        data_layout_file = os.path.join(
            sav_data_path, "metadata", fhd_prefix + "layout.sav"
        )
        data_params_file = os.path.join(
            sav_data_path, "metadata", fhd_prefix + "params.sav"
        )
        data_settings_file = os.path.join(
            sav_data_path, "metadata", fhd_prefix + "settings.txt"
        )

        uvd_data = UVData.from_file(
            data_vis_files,
            flags_file=data_flags_file,
            layout_file=data_layout_file,
            params_file=data_params_file,
            settings_file=data_settings_file,
        )

        # examine_flags(uvd_data)

        # exclude autos (invert select on ant1=ant2)
        uvd_data.select(ant_str="auto", invert=True)

        # select on one frequency (for now)
        uvd_data.select(frequencies=[uvd_data.freq_array[freq_ind]])

        # keep only one time (for now)
        # uvd_data.select(times=[uvd_data.time_array[uvd_data.Nbls*2]])
        print(
            f"\n***how many***\n"
            f"times {uvd_data.Ntimes}\tbls {uvd_data.Nbls}\tblts {uvd_data.Nblts}"
        )

        print(f"\n***all flagged? before***\n\t{np.all(uvd_data.flag_array)}")
        print(f"\n***any flagged? before***\n\t{np.any(uvd_data.flag_array)}")

        # remove flagged data (need to handle in unical code in future)
        print(f"\n***shape before removing flags***\n\t{uvd_data.data_array.shape}")
        flagged_bls_data = np.nonzero(np.squeeze(uvd_data.flag_array))[0]
        print(f"\n***flagged bls shape***\n\t{flagged_bls_data.shape}")
        uvd_data.select(blt_inds=flagged_bls_data, invert=True)
        print(f"\n***shape after removing flags***\n\t{uvd_data.data_array.shape}\n")
        print(f"\n***all flagged? after***\n\t{np.all(uvd_data.flag_array)}")
        print(f"\n***any flagged? after***\n\t{np.any(uvd_data.flag_array)}")
        uvd_data.write_uvfits(uv_data_path)

    sav_model_path = os.path.join(cwd, "calico", "data", sav_model_filename)
    uv_model_path = os.path.join(
        cwd, "calico", "data", f"{model_data_writeout_filename}.uvfits"
    )
    if os.path.isfile(uv_model_path) and not reconstruct_model:
        print("Model uvfits file exists - skipping")
    else:
        print("Model uvfits file not found - creating")
        # Set up the files we need
        model_vis_files = os.path.join(
            sav_model_path, "vis_data", fhd_prefix + "vis_model_XX.sav"
        )
        model_flags_file = os.path.join(
            sav_model_path, "vis_data", fhd_prefix + "flags.sav"
        )
        model_layout_file = os.path.join(
            sav_model_path, "metadata", fhd_prefix + "layout.sav"
        )
        model_params_file = os.path.join(
            sav_model_path, "metadata", fhd_prefix + "params.sav"
        )
        model_settings_file = os.path.join(
            sav_model_path, "metadata", fhd_prefix + "settings.txt"
        )

        uvd_model = UVData.from_file(
            model_vis_files,
            flags_file=model_flags_file,
            layout_file=model_layout_file,
            params_file=model_params_file,
            settings_file=model_settings_file,
        )

        # probably full repeat selections from before
        uvd_model.select(ant_str="auto", invert=True)
        uvd_model.select(frequencies=[uvd_model.freq_array[freq_ind]])
        # uvd_model.select(times=[uvd_model.time_array[uvd_model.Nbls*2]])
        flagged_bls_model = np.nonzero(np.squeeze(uvd_model.flag_array))[0]
        uvd_model.select(blt_inds=flagged_bls_model, invert=True)
        uvd_model.write_uvfits(uv_model_path)
        print(f"\n***shape after removing flags***\n\t{uvd_model.data_array.shape}\n")
        print(
            f"\n***how many***\ntimes {uvd_model.Ntimes}"
            f"\tbls {uvd_model.Nbls}\tblts {uvd_model.Nblts}"
        )

    return 1


def init_many_realizations(
    fhd_prefix="1061316296_",
    sav_data_filename="tutorial_full_onetime_unflagged",  # sav
    sav_model_filename="tutorial_full_onetime_unflagged",  # sav
    run_params_filename="baseline_dependence_runs_large_noise",
    vis_data_writeout_filename="tutorial_full_onetime_unflagged",  # uvfits
    model_data_writeout_filename="tutorial_full_onetime_unflagged",  # uvfits
    verbose=True,
    simulate_visibilities=False,
    same_sky_all_times=False,
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
    flatten_blts=False,
):
    dev = dev_tools.DevTools()
    if (
        prepare_data_files(
            fhd_prefix=fhd_prefix,
            sav_data_filename=sav_data_filename,
            sav_model_filename=sav_model_filename,
            vis_data_writeout_filename=vis_data_writeout_filename,
            model_data_writeout_filename=model_data_writeout_filename,
            reconstruct_data=reconstruct_data,
            reconstruct_model=reconstruct_model,
        )
        > 0
    ):
        if threshold_length is None:
            raise ValueError(
                "Need threshold length even if zero -- Init Many Realizations"
            )
        make_run_params.generate_files()
        cwd = os.getcwd()
        model_file_path = os.path.join(
            cwd, "calico", "data", f"{model_data_writeout_filename}.uvfits"
        )
        data_file_path = os.path.join(
            cwd, "calico", "data", f"{vis_data_writeout_filename}.uvfits"
        )
        if calibrate:
            if verbose:
                data_read_start_time = time.time()
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
                        "Done. Data read time "
                        f"{(time.time() - data_read_start_time) / 60.0} minutes."
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
                sigma_t_0=1,
                sigma_m_0=1,
                scaling_factor_cost=1,
                threshold_length=0,
                flatten_blts=flatten_blts,
            )
            print(
                "\n\n***AFTER LOAD***\n  "
                f"data {np.std(np.abs(caldata_obj.data_visibilities))}"
                f"\n  model {np.std(np.abs(caldata_obj.model_visibilities))}\n\n"
            )
            print(f"\n\n***Nfreqs***\n  {caldata_obj.Nfreqs}\n\n")
            print(f"\n\n***Ntimes***\n  {caldata_obj.Ntimes}\n\n")
            if verbose:
                print(
                    "Done. Data formatting time "
                    f"{(time.time() - data_format_start_time) / 60.0} minutes."
                )
                print("Running calibration optimization...")
                sys.stdout.flush()
                # optimization_start_time = time.time()
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
            if calibration_type == "skycal":
                for ant_flag_iter in range(antenna_flagging_iterations):
                    # lower xtol/maxiter and no crosspol for ant flagging
                    caldata_obj.sky_based_calibration(
                        xtol=xtol / 10,
                        maxiter=int(maxiter / 2),
                        get_crosspol_phase=False,
                        parallel=False,
                        verbose=verbose,
                        pool=None,
                    )
                    if verbose:
                        print(f"Initial calibration optimization done.", end="")
                        print(
                            f"Antenna flagging iteration {ant_flag_iter + 1} "
                            f"of {antenna_flagging_iterations}."
                        )
                        sys.stdout.flush()
            caldata_obj.flag_antennas_from_per_ant_cost(
                flagging_threshold=2.5, parallel=False, pool=None, verbose=verbose
            )
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
                simulate_visibilities=simulate_visibilities,
                same_sky_all_times=same_sky_all_times,
                gains_real_guess=gains_real_guess,
            )
        dev.plot_many_realizations(
            data_filepath=model_path,
            run_params_filename=run_params_filename,
            metadata=metadata,
            suffix=suffix,
        )

    else:
        print("Problem with data files - exiting")


prepare_data_files(
    fhd_prefix="1061316296_",
    sav_data_filename="fhd_runs/fhd_baseline",  # FHD sav
    sav_model_filename="fhd_runs/fhd_015",  # FHD sav
    model_data_writeout_filename="fhd_model_one_freq_015",  # uvfits
    vis_data_writeout_filename="fhd_data_one_freq_015",  # uvfits
    reconstruct_data=False,
    reconstruct_model=False,
)
