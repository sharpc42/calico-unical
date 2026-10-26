import argparse
import os
import os.path
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

from calico import caldata
from calico.dev import dev_tools, make_run_params
from pyuvdata import UVCal, UVData, UVFlag

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


def baselines_per_antenna_diagnostics(uvd, min_baselines, filename):
    """
    Print the antennas with fewer than min_baselines baselines and save a bar plot
    of the number of baselines per antenna to calico/images, with the antennas
    below the threshold in red and labeled with their antenna numbers.

    Returns
    -------
    ant_nums : array of int
        Antenna numbers present in the data.
    n_bls : array of int
        Number of baselines each antenna takes part in.
    """
    ant1, ant2 = uvd.baseline_to_antnums(np.unique(uvd.baseline_array))
    ant_nums, n_bls = np.unique(np.concatenate((ant1, ant2)), return_counts=True)
    below = n_bls < min_baselines
    print(
        f"\n***baselines per antenna***\n"
        f"\tantennas {ant_nums.size}\tmin {n_bls.min()}"
        f"\tmedian {np.median(n_bls):g}\tmax {n_bls.max()}"
    )
    print(f"***antennas with fewer than {min_baselines} baselines***")
    if np.any(below):
        for ant, n in zip(ant_nums[below], n_bls[below]):
            print(f"\tantenna {ant}: {n} baselines")
    else:
        print("\tnone")

    plt.figure(figsize=(14, 5))
    plt.bar(ant_nums, n_bls, width=0.8, color=np.where(below, "tab:red", "tab:blue"))
    for ant, n in zip(ant_nums[below], n_bls[below]):
        plt.text(
            ant,
            max(n, min_baselines) + 1,
            str(ant),
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=7,
            color="tab:red",
        )
    plt.axhline(min_baselines, color="tab:red", linestyle="--")
    plt.xlim(ant_nums.min() - 1, ant_nums.max() + 1)
    plt.xlabel("Antenna number")
    plt.ylabel("Number of baselines")
    plt.title(
        f"Baselines per antenna ({ant_nums.size} antennas, "
        f"{np.sum(below)} below the threshold of {min_baselines}, dashed)"
    )
    plt.tight_layout()
    image_path = os.path.join(os.getcwd(), "calico", "images", f"{filename}.png")
    plt.savefig(image_path)
    plt.close()
    return ant_nums, n_bls


def remove_fully_flagged_baselines(uvd):
    """
    Remove baselines flagged at every time, frequency, and polarization from uvd
    (in place). Antennas left with no unflagged baselines are removed with them.
    """
    bl_unique, inv = np.unique(uvd.baseline_array, return_inverse=True)
    blt_all_flagged = np.all(uvd.flag_array, axis=tuple(range(1, uvd.flag_array.ndim)))
    bl_all_flagged = np.ones(bl_unique.size, dtype=bool)
    np.logical_and.at(bl_all_flagged, inv, blt_all_flagged)
    print(
        f"\n***fully flagged baselines***\n\t{np.sum(bl_all_flagged)} of "
        f"{bl_unique.size}"
    )
    if not np.any(bl_all_flagged):
        return
    if np.all(bl_all_flagged):
        raise ValueError("Every baseline is fully flagged")
    ants_before = uvd.get_ants()
    ant1, ant2 = uvd.baseline_to_antnums(bl_unique[~bl_all_flagged])
    uvd.select(bls=list(zip(ant1.tolist(), ant2.tolist())))
    ants_removed = np.setdiff1d(ants_before, uvd.get_ants())
    print(
        f"***antennas with no unflagged baselines (removed)***\n\t"
        f"{ants_removed.tolist() if ants_removed.size else 'none'}"
    )


def downselect_antennas_by_baselines(uvd, min_baselines, filename):
    """
    Remove antennas with fewer than min_baselines baselines from uvd (in place),
    printing and plotting the baselines-per-antenna diagnostics before and after.
    Removing antennas also removes baselines from the antennas that remain, so
    this repeats until no remaining antenna falls below the threshold.
    """
    ant_nums, n_bls = baselines_per_antenna_diagnostics(
        uvd, min_baselines, f"{filename}_before"
    )
    n_passes = 0
    while np.any(n_bls < min_baselines):
        if np.all(n_bls < min_baselines):
            raise ValueError(
                f"Every remaining antenna has fewer than {min_baselines} baselines"
            )
        n_passes += 1
        print(
            f"\nPass {n_passes}: removing {np.sum(n_bls < min_baselines)} antennas "
            f"with fewer than {min_baselines} baselines"
        )
        uvd.select(antenna_nums=ant_nums[n_bls >= min_baselines])
        ant1, ant2 = uvd.baseline_to_antnums(np.unique(uvd.baseline_array))
        ant_nums, n_bls = np.unique(np.concatenate((ant1, ant2)), return_counts=True)
        for ant, n in zip(
            ant_nums[n_bls < min_baselines], n_bls[n_bls < min_baselines]
        ):
            print(f"\tnow below threshold: antenna {ant}: {n} baselines")
    if n_passes > 0:
        print(f"\nDownselect finished after {n_passes} passes")
        baselines_per_antenna_diagnostics(uvd, min_baselines, f"{filename}_after")


def prepare_data_files(
    fhd_prefix=None,
    sav_data_filename=None,
    sav_model_filename=None,
    vis_data_writeout_filename=None,
    model_data_writeout_filename=None,
    reconstruct_data=False,
    reconstruct_model=False,
    min_baselines_per_antenna=10,
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
        print("Creating data uvfits file\n")
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
        uvd_data.set_rectangularity(force=True)
        print(f"***data rectangular***\n\t{uvd_data.blts_are_rectangular}\n")

        # examine_flags(uvd_data)

        # exclude autos (invert select on ant1=ant2)
        uvd_data.select(ant_str="auto", invert=True)

        # keep baseline only if it stays under
        # FHD uv-plane cutoff at all times
        print(f"Excluding baselines outside FHD cutoff (data)\n")
        c = 3e8
        uv_plane_cutoff = 500
        uv_len_m = np.linalg.norm(uvd_data.uvw_array[:, :2], axis=1)
        freq_ref = uvd_data.freq_array.max()
        uv_len_lam = uv_len_m * freq_ref / c
        bl_unique, inv = np.unique(uvd_data.baseline_array, return_inverse=True)
        max_uv_lam = np.zeros(bl_unique.size)
        np.maximum.at(max_uv_lam, inv, uv_len_lam)
        keep_bl_nums = bl_unique[max_uv_lam < uv_plane_cutoff]
        ant1, ant2 = uvd_data.baseline_to_antnums(keep_bl_nums)
        keep_bls = list(zip(ant1.tolist(), ant2.tolist()))
        uvd_data.select(bls=keep_bls)

        # select on one frequency (for now), before the flag-based selections
        # below so they use this frequency's flags
        uvd_data.select(frequencies=[uvd_data.freq_array[freq_ind]])

        # drop baselines (and so antennas) with no unflagged data
        remove_fully_flagged_baselines(uvd_data)

        # drop antennas too poorly connected to constrain their gains
        downselect_antennas_by_baselines(
            uvd_data,
            min_baselines_per_antenna,
            f"baselines_per_antenna_{vis_data_writeout_filename}",
        )

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
        # uvd_data.select(blt_inds=flagged_bls_data, invert=True)
        print(f"\n***shape after removing flags***\n\t{uvd_data.data_array.shape}\n")
        print(f"\n***all flagged? after***\n\t{np.all(uvd_data.flag_array)}")
        print(f"\n***any flagged? after***\n\t{np.any(uvd_data.flag_array)}")
        uvd_data.write_uvfits(uv_data_path)
        print(f"***data rectangular***\n\t{uvd_data.blts_are_rectangular}\n")

    sav_model_path = os.path.join(cwd, "calico", "data", sav_model_filename)
    uv_model_path = os.path.join(
        cwd, "calico", "data", f"{model_data_writeout_filename}.uvfits"
    )
    if os.path.isfile(uv_model_path) and not reconstruct_model:
        print("Model uvfits file exists - skipping")
    else:
        print("Creating model uvfits file\n")
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
        uvd_model.set_rectangularity(force=True)
        print(f"***model rectangular***\n\t{uvd_model.blts_are_rectangular}")

        # probably full repeat selections from before
        uvd_model.select(ant_str="auto", invert=True)

        # keep baseline only if it stays under
        # FHD uv-plane cutoff at all times
        print(f"Excluding baselines outside FHD cutoff (model)\n")
        uv_len_m = np.linalg.norm(uvd_model.uvw_array[:, :2], axis=1)
        freq_ref = uvd_model.freq_array.max()
        uv_len_lam = uv_len_m * freq_ref / c
        bl_unique, inv = np.unique(uvd_model.baseline_array, return_inverse=True)
        max_uv_lam = np.zeros(bl_unique.size)
        np.maximum.at(max_uv_lam, inv, uv_len_lam)
        keep_bl_nums = bl_unique[max_uv_lam < uv_plane_cutoff]
        ant1, ant2 = uvd_model.baseline_to_antnums(keep_bl_nums)
        keep_bls = list(zip(ant1.tolist(), ant2.tolist()))
        uvd_model.select(bls=keep_bls)

        # select on one frequency (for now), before the flag-based selections
        # below so they use this frequency's flags
        uvd_model.select(frequencies=[uvd_model.freq_array[freq_ind]])

        # drop baselines (and so antennas) with no unflagged data
        remove_fully_flagged_baselines(uvd_model)

        # drop antennas too poorly connected to constrain their gains
        downselect_antennas_by_baselines(
            uvd_model,
            min_baselines_per_antenna,
            f"baselines_per_antenna_{model_data_writeout_filename}",
        )

        # uvd_model.select(times=[uvd_model.time_array[uvd_model.Nbls*2]])
        flagged_bls_model = np.nonzero(np.squeeze(uvd_model.flag_array))[0]
        # uvd_model.select(blt_inds=flagged_bls_model, invert=True)
        uvd_model.write_uvfits(uv_model_path)
        print(f"\n***shape after removing flags***\n\t{uvd_model.data_array.shape}\n")
        print(
            f"\n***how many***\ntimes {uvd_model.Ntimes}"
            f"\tbls {uvd_model.Nbls}\tblts {uvd_model.Nblts}"
        )
        print(f"***model rectangular***\n\t{uvd_model.blts_are_rectangular}")

    return 1


def prepare_cal_file(
    fhd_prefix=None,
    sav_cal_filename=None,
    vis_data_writeout_filename=None,
    model_data_writeout_filename=None,
    cal_writeout_filename=None,
    reconstruct_cal=False,
):
    """
    Read FHD file, downselect to frequencies, polarizations and antennas of the
    data/model uvfits files written by prepare_data_files (so those must exist first),
    and write them out as a calh5 file. The FHD run is assumed to come from the same
    underlying data as the uvfits files; a warning is raised for anything suggesting
    otherwise.
    """
    if fhd_prefix is None:
        print("Error: FHD prefix is missing")
        return -1
    if sav_cal_filename is None:
        print("Error: sav cal filename is missing")
        return -1
    if vis_data_writeout_filename is None:
        print("Error: uvfits data filename is missing")
        return -1
    if model_data_writeout_filename is None:
        print("Error: uvfits model filename is missing")
        return -1
    if cal_writeout_filename is None:
        print("Error: calh5 cal filename is missing")
        return -1
    cwd = os.getcwd()
    cal_path = os.path.join(cwd, "calico", "data", f"{cal_writeout_filename}.calh5")

    print(f"\n***cal path {cal_path}***\n")

    if os.path.isfile(cal_path) and not reconstruct_cal:
        print("\nCal calh5 file exists - skipping\n")
        return 1
    uv_data_path = os.path.join(
        cwd, "calico", "data", f"{vis_data_writeout_filename}.uvfits"
    )
    uv_model_path = os.path.join(
        cwd, "calico", "data", f"{model_data_writeout_filename}.uvfits"
    )
    for path in (uv_data_path, uv_model_path):
        if not os.path.isfile(path):
            print(f"Error: {path} not found - run prepare_data_files first")
            return -1
    if fhd_prefix[-1] != "_":
        fhd_prefix += "_"

    print("Creating cal calh5 file\n")
    sav_cal_path = os.path.join(cwd, "calico", "data", sav_cal_filename)
    uvc = UVCal.from_file(
        os.path.join(sav_cal_path, "calibration", fhd_prefix + "cal.sav"),
        obs_file=os.path.join(sav_cal_path, "metadata", fhd_prefix + "obs.sav"),
        layout_file=os.path.join(sav_cal_path, "metadata", fhd_prefix + "layout.sav"),
        settings_file=os.path.join(
            sav_cal_path, "metadata", fhd_prefix + "settings.txt"
        ),
    )
    print(
        f"\n***FHD cal***\n\tfreqs {uvc.Nfreqs}\tants {uvc.Nants_data}"
        f"\tjones {uvc.jones_array}\tgain shape {uvc.gain_array.shape}"
    )

    # data flags needed for time check, model metadata only
    uvd_data = UVData.from_file(uv_data_path)
    uvd_model = UVData.from_file(uv_model_path, read_data=False)

    mismatches = []
    if not np.allclose(uvd_data.freq_array, uvd_model.freq_array):
        mismatches.append(
            f"data freqs {uvd_data.freq_array} and model freqs "
            f"{uvd_model.freq_array} differ"
        )

    # match each data frequency to an FHD channel
    freq_tol = 1e-3 * np.min(uvc.channel_width)
    freq_inds = []
    for freq in uvd_data.freq_array:
        ind = np.argmin(np.abs(uvc.freq_array - freq))
        if np.abs(uvc.freq_array[ind] - freq) > freq_tol:
            mismatches.append(
                f"data freq {freq} Hz has no FHD cal channel (closest "
                f"{uvc.freq_array[ind]} Hz)"
            )
        else:
            freq_inds.append(ind)
    if not np.allclose(
        uvc.channel_width[freq_inds],
        np.broadcast_to(uvd_data.channel_width, (uvd_data.Nfreqs,))[: len(freq_inds)],
    ):
        mismatches.append("FHD cal and data channel widths don't match")

    jones = [j for j in uvc.jones_array if j in uvd_data.polarization_array]
    if len(jones) == 0:
        mismatches.append(
            f"no FHD cal jones {uvc.jones_array} match the data polarizations "
            f"{uvd_data.polarization_array}"
        )

    # keep antennas left in either file after prepare data files cuts
    ants = np.union1d(uvd_data.get_ants(), uvd_model.get_ants())
    missing_ants = np.setdiff1d(ants, uvc.ant_array)
    if missing_ants.size:
        mismatches.append(f"antennas {missing_ants.tolist()} have no FHD gains")
    ants = np.intersect1d(ants, uvc.ant_array)

    # the same antenna numbers should name the same tiles
    cal_names = dict(zip(uvc.telescope.antenna_numbers, uvc.telescope.antenna_names))
    renamed = [
        num
        for num, name in zip(
            uvd_data.telescope.antenna_numbers, uvd_data.telescope.antenna_names
        )
        if num in cal_names and cal_names[num] != name
    ]
    if renamed:
        mismatches.append(
            f"antenna numbers {renamed} have different names in the FHD cal"
        )

    # FHD gives one solution over a time range, which should cover the data
    if uvc.time_range is not None:
        t_min, t_max = uvc.time_range.min(), uvc.time_range.max()
        # allow half an integration either side for time stamps at bin centers
        t_pad = np.max(uvd_data.integration_time) / 2 / 86400
        # FHD's range starts at the first unflagged time, so skip fully flagged
        # times (e.g. the quack time at the start of MWA observations)
        blt_unflagged = ~np.all(
            uvd_data.flag_array, axis=tuple(range(1, uvd_data.flag_array.ndim))
        )
        data_times = np.unique(uvd_data.time_array[blt_unflagged])
        if data_times.min() < t_min - t_pad or data_times.max() > t_max + t_pad:
            mismatches.append(
                f"data times [{data_times.min()}, {data_times.max()}] fall outside "
                f"the FHD cal time range [{t_min}, {t_max}]"
            )

    if mismatches:
        warnings.warn(
            "FHD cal may not match the data/model files:\n\t" + "\n\t".join(mismatches),
            stacklevel=2,
        )
    if len(freq_inds) == 0 or len(jones) == 0 or ants.size == 0:
        print("Error: nothing left in FHD cal after matching to data/model files")
        return -1

    uvc.select(freq_chans=freq_inds, jones=jones, antenna_nums=ants)
    print(
        f"\n***cal after downselect***\n\tfreqs {uvc.freq_array}\tants "
        f"{uvc.Nants_data}\tjones {uvc.jones_array}\tgain shape "
        f"{uvc.gain_array.shape}"
    )
    uvc.write_calh5(cal_path, clobber=True)
    return 1


def prepare_matched_data_file(
    raw_data_filename=None,
    reference_filename=None,
    writeout_filename=None,
    reconstruct=False,
):
    """
    Downselect raw uvfits file to frequencies, polarizations, baselines and
    times of reference uvfits file, and write it out as a uvfits file. A
    warning is raised for anything in the raw file that doesn't line up with
    the reference.
    """
    if raw_data_filename is None:
        print("Error: raw uvfits data filename is missing")
        return -1
    if reference_filename is None:
        print("Error: reference uvfits filename is missing")
        return -1
    if writeout_filename is None:
        print("Error: uvfits writeout filename is missing")
        return -1
    cwd = os.getcwd()
    out_path = os.path.join(cwd, "calico", "data", f"{writeout_filename}.uvfits")
    print("matched data path", out_path)
    if os.path.isfile(out_path) and not reconstruct:
        print("Matched data uvfits file exists - skipping")
        return 1
    raw_path = os.path.join(cwd, "calico", "data", f"{raw_data_filename}.uvfits")
    ref_path = os.path.join(cwd, "calico", "data", f"{reference_filename}.uvfits")
    for path in (raw_path, ref_path):
        if not os.path.isfile(path):
            print(f"ERROR: {path} not found")
            return -1

    print("Creating matched data uvfits file\n")
    uvd_ref = UVData.from_file(ref_path)
    uvd_raw_meta = UVData.from_file(raw_path, read_data=False)

    mismatches = []
    # match each reference frequency to raw channel within tolerance
    freq_tol = 1e-3 * np.min(uvd_raw_meta.channel_width)
    freq_chans = []
    for freq in uvd_ref.freq_array:
        ind = np.argmin(np.abs(uvd_raw_meta.freq_array - freq))
        if np.abs(uvd_raw_meta.freq_array[ind] - freq) > freq_tol:
            mismatches.append(
                f"reference freq {freq} Hz has no raw channel (closest "
                f"{uvd_raw_meta.freq_array[ind]} Hz)"
            )
        else:
            freq_chans.append(ind)

    pols = [
        p for p in uvd_ref.polarization_array if p in uvd_raw_meta.polarization_array
    ]
    if len(pols) < uvd_ref.Npols:
        mismatches.append(
            f"reference pols {uvd_ref.polarization_array} not all in the raw pols "
            f"{uvd_raw_meta.polarization_array}"
        )

    # match times to raw file's own values
    time_tol = 1e-2 / 86400  # 0.01 s, in days (86400 seconds in a day)
    raw_times = np.unique(uvd_raw_meta.time_array)
    times = []
    for t in np.unique(uvd_ref.time_array):
        ind = np.argmin(np.abs(raw_times - t))
        if np.abs(raw_times[ind] - t) > time_tol:
            mismatches.append(f"reference time {t} has no raw time")
        else:
            times.append(raw_times[ind])

    ref_bls = np.unique(uvd_ref.baseline_array)
    missing_bls = np.setdiff1d(ref_bls, uvd_raw_meta.baseline_array)
    if missing_bls.size:
        mismatches.append(f"{missing_bls.size} reference baselines not in the raw file")
    ant1, ant2 = uvd_ref.baseline_to_antnums(
        np.intersect1d(ref_bls, uvd_raw_meta.baseline_array)
    )
    bls = list(zip(ant1.tolist(), ant2.tolist()))

    if len(freq_chans) == 0 or len(pols) == 0 or len(times) == 0 or len(bls) == 0:
        warnings.warn(
            "Raw data may not match the reference file:\n\t" + "\n\t".join(mismatches),
            stacklevel=2,
        )
        print("ERROR: nothing left in raw data after matching to reference file")
        return -1

    uvd_raw = UVData.from_file(
        raw_path, freq_chans=freq_chans, polarizations=pols, bls=bls, times=times
    )

    # check they describe same visibilities
    uvd_raw.reorder_blts(order="time", minor_order="baseline")
    uvd_ref.reorder_blts(order="time", minor_order="baseline")
    if uvd_raw.Nblts != uvd_ref.Nblts:
        mismatches.append(f"raw Nblts {uvd_raw.Nblts} != reference {uvd_ref.Nblts}")
    else:
        if not np.array_equal(uvd_raw.baseline_array, uvd_ref.baseline_array):
            mismatches.append("raw and reference baseline orders differ")
        if not np.allclose(
            uvd_raw.time_array, uvd_ref.time_array, rtol=0, atol=time_tol
        ):
            mismatches.append("raw and reference times differ")
        if not np.allclose(uvd_raw.uvw_array, uvd_ref.uvw_array, rtol=0, atol=1e-3):
            mismatches.append(
                "raw and reference uvws differ (max "
                f"{np.max(np.abs(uvd_raw.uvw_array - uvd_ref.uvw_array)):.3g} m)"
            )
        if uvd_raw.flag_array.shape == uvd_ref.flag_array.shape and not np.array_equal(
            uvd_raw.flag_array, uvd_ref.flag_array
        ):
            print(
                "Note: raw and reference flags differ "
                f"(raw {uvd_raw.flag_array.mean():.3f}, "
                f"reference {uvd_ref.flag_array.mean():.3f} flagged)"
            )
    raw_cat = list(uvd_raw.phase_center_catalog.values())
    ref_cat = list(uvd_ref.phase_center_catalog.values())
    if len(raw_cat) == 1 and len(ref_cat) == 1:
        d_lon = raw_cat[0]["cat_lon"] - ref_cat[0]["cat_lon"]
        d_lat = raw_cat[0]["cat_lat"] - ref_cat[0]["cat_lat"]
        if np.hypot(d_lon * np.cos(ref_cat[0]["cat_lat"]), d_lat) > 1e-5:
            mismatches.append(
                f"phase centers differ by {np.hypot(d_lon, d_lat):.3g} rad"
            )

    if mismatches:
        warnings.warn(
            "Raw data may not match the reference file:\n\t" + "\n\t".join(mismatches),
            stacklevel=2,
        )
    print(
        f"\n***matched raw data***\n\tfreqs {uvd_raw.freq_array}\tpols "
        f"{uvd_raw.get_pols()}\tbls {uvd_raw.Nbls}\ttimes {uvd_raw.Ntimes}"
        f"\tvis_units {uvd_raw.vis_units}"
    )
    uvd_raw.write_uvfits(out_path)
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
                lambda_val=100,
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
                    caldata_obj.xtol = xtol / 10
                    caldata_obj.maxiter = int(maxiter / 2)
                    caldata_obj.get_crosspol_phase = False
                    caldata_obj.parallel = False
                    caldata_obj.verbose = verbose
                    caldata_obj.sky_based_calibration()
                    if verbose:
                        print(f"Initial calibration optimization done.", end="")
                        print(
                            f"Antenna flagging iteration {ant_flag_iter + 1} "
                            f"of {antenna_flagging_iterations}."
                        )
                        sys.stdout.flush()
            caldata_obj.flag_antennas_from_per_ant_cost(flagging_threshold=2.5)
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
        # dev.plot_many_realizations(
        #     data_filepath=model_path,
        #     run_params_filename=run_params_filename,
        #     metadata=metadata,
        #     suffix=suffix,
        # )

    else:
        print("Problem with data files - exiting")


parser = argparse.ArgumentParser(description="Creating uvfits from FHD files")
parser.add_argument(
    "--suffix", type=str, help="file suffix for FHD files only (the cutoff number)"
)
parser.add_argument(
    "-r",
    "--reconstruct",
    action="store_true",
    help="reconstruct data, model and cal files",
)
parser.add_argument(
    "--min-baselines",
    type=int,
    default=10,
    help="drop antennas with fewer than this many baselines (default 10)",
)
parser.add_argument(
    "--raw-data",
    type=str,
    default=None,
    help="raw uvfits file in calico/data, selected down to match FHD model",
)
args = parser.parse_args()
if args.suffix:
    prepare_data_files(
        fhd_prefix="1061316296_",
        sav_data_filename="fhd_runs/fhd_baseline",  # FHD sav
        sav_model_filename=f"fhd_runs/fhd_{args.suffix}",  # FHD sav
        model_data_writeout_filename=f"fhd_model_one_freq_{args.suffix}",  # uvfits
        vis_data_writeout_filename=f"fhd_data_one_freq_{args.suffix}",  # uvfits
        reconstruct_data=args.reconstruct,
        reconstruct_model=args.reconstruct,
        min_baselines_per_antenna=args.min_baselines,
    )
    prepare_cal_file(
        fhd_prefix="1061316296_",
        sav_cal_filename=f"fhd_runs/fhd_{args.suffix}",  # FHD sav
        vis_data_writeout_filename=f"fhd_data_one_freq_{args.suffix}",  # uvfits
        model_data_writeout_filename=f"fhd_model_one_freq_{args.suffix}",  # uvfits
        cal_writeout_filename=f"fhd_cal_one_freq_{args.suffix}",  # calh5
        reconstruct_cal=args.reconstruct,
    )
    if args.raw_data:
        prepare_matched_data_file(
            raw_data_filename=args.raw_data,  # uvfits
            reference_filename=f"fhd_model_one_freq_{args.suffix}",  # uvfits
            writeout_filename=f"obs_data_one_freq_{args.suffix}",  # uvfits
            reconstruct=args.reconstruct,
        )
