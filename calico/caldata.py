import numpy as np
import sys
import pyuvdata
from astropy.units import Quantity
from calico import calibration_qa, calibration_optimization, dev_tools
import multiprocessing


class CalData:
    """
    Object containing all data and parameters needed for calibration.

    Attributes
    -------
    gains : array of complex
        Shape (Nants, Nfreqs, N_feed_pols,).
    fit_vis : array of complex
        Shape (Ntimes, Nbls,  Nfreqs, N_vis_pols).
    abscal_params : array of float
        Shape (3, Nfreqs, N_feed_pols). abscal_params[0, :, :] are the overall amplitudes,
        abscal_params[1, :, :] are the x-phase gradients in units 1/m, and abscal_params[2, :, :]
        are the y-phase gradients in units 1/m.
    Nants : int
        Number of antennas.
    Nbls : int
        Number of baselines.
    Ntimes : int
        Number of time intervals.
    Nfreqs : int
        Number of frequency channels.
    N_feed_pols : int
        Number of gain polarizations.
    N_vis_pols : int
        Number of visibility polarizations.
    feed_polarization_array : array of int
        Shape (N_feed_pols). Array of polarization integers. Indicates the
        ordering of the polarization axis of the gains. X is -5 and Y is -6.
    vis_polarization_array : array of int
        Shape (N_vis_pols,). Array of polarization integers. Indicates the
        ordering of the polarization axis of the model_visibilities,
        data_visibilities, and visibility_weights. XX is -5, YY is -6, XY is -7,
        and YX is -8.
    model_visibilities : array of complex
        Shape (Ntimes, Nbls, Nfreqs, N_vis_pols,).
    data_visibilities : array of complex
        Shape (Ntimes, Nbls, Nfreqs, N_vis_pols,).
    visibility_weights : array of float
        Shape (Ntimes, Nbls, Nfreqs, N_vis_pols,).
    model_weights : array of float
        Shape (Ntimes, Nbls, Nfreqs, N_vis_pols)
    dwcal_inv_covariance : array of complex
        Matrix defining frequency-frequency covariances used in delay-weighted
        calibration. Needed only if delay weighting is used in calibration.
        If dwcal_memory_save_mode is False, dwcal_inv_covariance has shape
        (Ntimes, Nbls, Nfreqs, Nfreqs, N_vis_pols,). If dwcal_memory_save_mode
        is True, dwcal_inv_covariance has shape (Ntimes, Nbls, Nfreqs, N_vis_pols,).
        Alternatively, if the time or polarization axes have length 1,
        dw_inv_covariance is assumed to be identical across time steps or polarization.
    dwcal_memory_save_mode : bool
        Defines the format of dwcal_inv_covariance. If True, dwcal_inv_covariance
        is assumed to be Toeplitz and is stored in a more compact form.
    ant1_inds : array of int
        Shape (Nbls,).
    ant2_inds : array of int
        Shape (Nbls,).
    bl_inds : array of int
        Shape (Nbls,).
    gains_multiply_model : bool
        If True, measurement equation is defined as v_ij ≈ g_i g_j^* m_ij. If False,
        measurement equation is defined as g_i g_j^* v_ij ≈ m_ij.
    antenna_names : array of str
        Shape (Nants,). Ordering matches the ordering of the gains attribute.
    antenna_numbers : array of int
        Shape (Nants,). Ordering matches the ordering of the gains attribute.
    antenna_positions : array of float
        Shape (Nants, 3,). Units meters, relative to telescope location.
    uv_array : array of float
        Shape (Nbls, 2,). Baseline positions in the UV plane, units meters.
    channel_width : float
        Width of frequency channels in Hz.
    freq_array : array of float
        Shape (Nfreqs,). Units Hz.
    integration_time : float
        Length of integration in seconds.
    time : float
        Time of observation in Julian Date.
    telescope_name : str
    lst : str
        Local sidereal time (LST), in radians.
    telescope_location : array of float
    lambda_val : float
        Weight of the phase regularization term; must be positive. Default 100.
    gains_real : array of float
        Shape (Nants,). Real part of gains per frequency and per polarization.
    gains_imag : array of float
        Shape (Nants,). Imaginary part of gains per frequency and per polarization.
    fit_vis_real : array of float
        Shape (). Real part of fitted visibilities per frequency and per polarization.
    fit_vis_imag : array of float
        Shape (). Imaginary part of fitted visibilities per frequency and per polarization.
    data_vis_reshaped : array of complex
        Shape (Ntimes, Nbls). Reshaped for optimization function linear algebra.
    model_vis_reshaped : array of complex
        Shape (Ntimes, Nbls). Reshaped for optimization function linear algebra.
    vis_weights_reshaped : array of float
        Shape (Ntimes, Nbls). Reshaped for optimization function linear algebra.
    ant_inds : array of int
        Shape (Nants_unflagged). Indices of unflagged antennas to be calibrated.
    """

    def __init__(self):
        self.gains = None
        self.abscal_params = None
        self.fit_vis = None
        self.Nants = 0
        self.Nbls = 0
        self.Ntimes = 0
        self.Nfreqs = 0
        self.N_feed_pols = 0
        self.N_vis_pols = 0
        self.feed_polarization_array = None
        self.vis_polarization_array = None
        self.model_visibilities = None
        self.data_visibilities = None
        self.visibility_weights = None
        self.model_weights = None
        self.dwcal_inv_covariance = None
        self.dwcal_memory_save_mode = None
        self.ant1_inds = None
        self.ant2_inds = None
        self.bl_inds = None
        self.gains_multiply_model = None
        self.antenna_names = None
        self.antenna_numbers = None
        self.antenna_positions = None
        self.uv_array = None
        self.uv_norm = None
        self.channel_width = None
        self.freq_array = None
        self.integration_time = None
        self.time = None
        self.telescope_name = None
        self.lst = None
        self.telescope_location = None
        self.lambda_val = None
        self.gains_real = None
        self.gains_imag = None
        self.fit_vis_real = None
        self.fit_vis_imag = None
        self.data_vis_reshaped = None
        self.model_vis_reshaped = None
        self.vis_weights_reshaped = None
        self.ant_inds = None
        self.bl_inds = None
        self.gain_init_stddev = 0.0
        self.fit_vis_init_stddev = 0.0
        # dev
        self.data_vis_orig = None
        self.fit_vis_orig = None
        self.gains_orig = None
        self.gain_params_realizations = None
        self.model_params_realizations = None
        self.glim = None
        self.ulim = None
        self.sigma_t = None
        self.sigma_m = None
        self.sigma_n = None
        self.sigma_e = None
        self.gain_realizations = None
        self.model_realizations = None

    def set_gains_from_calfile(self, calfile):
        """
        Use a pyuvdata-formatted calfits file to set gains.

        Parameters
        ----------
        calfile : str
            Path to a pyuvdata-formatted calfits file or a CASA-formatted .bcal file.
        """

        uvcal = pyuvdata.UVCal()
        if calfile.endswith(".calfits"):
            uvcal.read_calfits(calfile)
        elif calfile.endswith(".bcal"):
            uvcal.read_ms_cal(calfile)
        else:
            print(f"ERROR: Unknown file extension for file {calfile}. Exiting.")
            sys.exit(1)
        uvcal.select(frequencies=self.freq_array, antenna_names=self.antenna_names)
        if self.feed_polarization_array is None:
            self.feed_polarization_array = uvcal.jones_array
        else:
            uvcal.select(jones=self.feed_polarization_array)
        uvcal.reorder_freqs(channel_order="freq")
        uvcal.reorder_jones()
        use_gains = np.mean(uvcal.gain_array, axis=2)  # Average over times

        # Make antenna ordering match
        cal_ant_names = np.array([uvcal.antenna_names[ant] for ant in uvcal.ant_array])
        cal_ant_inds = np.array(
            [list(cal_ant_names).index(name) for name in self.antenna_names]
        )

        if self.gains_multiply_model:
            if uvcal.gain_convention != "divide":
                use_gains = 1 / use_gains
        else:
            if uvcal.gain_convention != "multiply":
                use_gains = 1 / use_gains

        self.gains = use_gains[cal_ant_inds, :]

    def load_data(
        self,
        data,
        model,
        gain_init_calfile=None,
        gain_init_to_vis_ratio=True,
        gains_multiply_model=False,
        gain_init_stddev=0.0,
        fit_vis_init_stddev=0.0,
        N_feed_pols=None,
        feed_polarization_array=None,
        min_cal_baseline_m=None,
        max_cal_baseline_m=None,
        min_cal_baseline_lambda=None,
        max_cal_baseline_lambda=None,
        lambda_val=100,
        glim=(-1,1),
        ulim=(-10,10),
        gain_realizations=100,
        model_realizations=1,
    ):
        """
        Format CalData object with parameters from data and model UVData
        objects.

        Parameters
        ----------
        data : pyuvdata UVData object
            Data to be calibrated.
        model : pyuvdata UVData object
            Model visibilities to be used in calibration. Must have the same
            parameters at data.
        gain_init_calfile : str or None
            Default None. If not None, provides a path to a pyuvdata-formatted
            calfits file containing gains values for calibration initialization.
        gain_init_to_vis_ratio : bool
            Used only if gain_init_calfile is None. If True, initializes gains
            to the median ratio between the amplitudes of the model and data
            visibilities. If False, the gains are initialized to 1. Default
            True.
        gains_multiply_model : bool
            If True, measurement equation is defined as v_ij ≈ g_i g_j^* m_ij. If
            False, measurement equation is defined as g_i g_j^* v_ij ≈ m_ij. This
            parameter affects how calibration is performed, and whether the data is
            multiplied or divided by the gains when calibration solutions are applied.
            Default False.
        gain_init_stddev : float
            Default 0.0. Standard deviation of a random complex Gaussian
            perturbation to the initial gains.
        fit_vis_init_stddev : float
            Default 0.0. Standard deviation of a random complex Gaussian
            perturbation to the initial fitted visibility values.
        N_feed_pols : int
            Default min(2, N_vis_pols). Number of feed polarizations, equal to
            the number of gain values to be calculated per antenna.
        feed_polarization_array : array of int or None
            Feed polarizations to calibrate. Shape (N_feed_pols,). Options are
            -5 for X or -6 for Y. Default None. If None, feed_polarization_array
            is set to ([-5, -6])[:N_feed_pols].
        min_cal_baseline_m : float or None
            Minimum baseline length, in meters, to use in calibration. If both
            min_cal_baseline_m and min_cal_baseline_lambda are None, arbitrarily
            short baselines are used. Default None.
        max_cal_baseline_m : float or None
            Maximum baseline length, in meters, to use in calibration. If both
            max_cal_baseline_m and max_cal_baseline_lambda are None, arbitrarily
            long baselines are used. Default None.
        min_cal_baseline_lambda : float or None
            Minimum baseline length, in wavelengths, to use in calibration. If
            both min_cal_baseline_m and min_cal_baseline_lambda are None,
            arbitrarily short baselines are used. Default None.
        max_cal_baseline_lambda : float or None
            Maximum baseline length, in wavelengths, to use in calibration. If
            both max_cal_baseline_m and max_cal_baseline_lambda are None,
            arbitrarily long baselines are used. Default None.
        lambda_val : float
            Weight of the phase regularization term; must be positive. Default
            100.
        """

        # Autocorrelations are not currently supported
        data.select(ant_str="cross")
        model.select(ant_str="cross")

        # Ensure polarizations match
        if model.Npols > data.Npols:
            model.select(polarizations=data.polarization_array)

        # Ensure times match
        time_match_tol = 1e-5
        if (
            np.max(
                np.abs(
                    np.sort(list(set(data.time_array)))
                    - np.sort(list(set(model.time_array)))
                )
            )
            > time_match_tol
        ):
            print("ERROR: Data and model times do not match. Exiting.")
            sys.exit(1)

        # Ensure frequencies match
        freq_match_tol = 1e-5
        if (
            np.max(np.abs(np.sort(data.freq_array) - np.sort(model.freq_array)))
            > freq_match_tol
        ):
            print("ERROR: Data and model frequencies do not match. Exiting.")
            sys.exit(1)

        # Downselect baselines
        if (
            (min_cal_baseline_m is not None)
            or (max_cal_baseline_m is not None)
            or (min_cal_baseline_lambda is not None)
            or (max_cal_baseline_lambda is not None)
        ):
            if min_cal_baseline_m is None:
                min_cal_baseline_m = 0.0
            if max_cal_baseline_m is None:
                max_cal_baseline_m = np.inf
            if min_cal_baseline_lambda is None:
                min_cal_baseline_lambda = 0.0
            if max_cal_baseline_lambda is None:
                max_cal_baseline_lambda = np.inf

            max_cal_baseline_m = np.min(
                [
                    max_cal_baseline_lambda * 3e8 / np.min(data.freq_array),
                    max_cal_baseline_m,
                ]
            )
            min_cal_baseline_m = np.max(
                [
                    min_cal_baseline_lambda * 3e8 / np.max(data.freq_array),
                    min_cal_baseline_m,
                ]
            )

            data_baseline_lengths_m = np.sqrt(np.sum(data.uvw_array**2.0, axis=1))
            data_use_baselines = np.where(
                (data_baseline_lengths_m >= min_cal_baseline_m)
                & (data_baseline_lengths_m <= max_cal_baseline_m)
            )
            data.select(blt_inds=data_use_baselines)

            model_baseline_lengths_m = np.sqrt(np.sum(model.uvw_array**2.0, axis=1))
            model_use_baselines = np.where(
                (model_baseline_lengths_m >= min_cal_baseline_m)
                & (model_baseline_lengths_m <= max_cal_baseline_m)
            )
            model.select(blt_inds=model_use_baselines)

        # Ensure baselines match
        data.conjugate_bls()
        data.reorder_blts()
        model.conjugate_bls()
        model.reorder_blts()
        if data.Nblts != model.Nblts:
            select_baselines = True
        elif (np.max(np.abs(data.ant_1_array - model.ant_1_array)) > 0) or (
            np.max(np.abs(data.ant_2_array - model.ant_2_array)) > 0
        ):
            select_baselines = True
        else:
            select_baselines = False
        if select_baselines:
            data_baselines = list(set(zip(data.ant_1_array, data.ant_2_array)))
            model_baselines = list(set(zip(model.ant_1_array, model.ant_2_array)))
            use_baselines = [
                baseline for baseline in data_baselines if baseline in model_baselines
            ]
            if len(use_baselines) < data.Nbls:
                print(
                    f"WARNING: Model does not contain all baselines. Downselecting from {data.Nbls} to {len(use_baselines)}."
                )
            data.select(bls=use_baselines)
            model.select(bls=use_baselines)

        self.Nants = data.Nants_data
        self.Nbls = data.Nbls
        self.Ntimes = data.Ntimes
        self.Nfreqs = data.Nfreqs
        self.N_vis_pols = data.Npols

        # Format visibilities
        self.data_visibilities = np.zeros(
            (
                self.Ntimes,
                self.Nbls,
                self.Nfreqs,
                self.N_vis_pols,
            ),
            dtype=complex,
        )
        self.model_visibilities = np.zeros(
            (
                self.Ntimes,
                self.Nbls,
                self.Nfreqs,
                self.N_vis_pols,
            ),
            dtype=complex,
        )
        flag_array = np.zeros(
            (self.Ntimes, self.Nbls, self.Nfreqs, self.N_vis_pols), dtype=bool
        )
        for time_ind, time_val in enumerate(np.unique(data.time_array)):
            data_copy = data.select(times=time_val, inplace=False)
            model_times = list(set(model.time_array))
            model_copy = model.select(
                times=model_times[
                    np.where(
                        np.abs(model_times - time_val)
                        == np.min(np.abs(model_times - time_val))
                    )[0][
                        0
                    ]  # Account for times that are close but not exactly equal
                ],
                inplace=False,
            )
            data_copy.reorder_blts()
            model_copy.reorder_blts()
            data_copy.reorder_pols(order="AIPS")
            model_copy.reorder_pols(order="AIPS")
            data_copy.reorder_freqs(channel_order="freq")
            model_copy.reorder_freqs(channel_order="freq")

            if time_ind == 0:
                metadata_reference = data_copy.copy(metadata_only=True)

            # self.model_visibilities[time_ind, :, :, :] = np.reshape(
            #     model_copy.data_array,
            #     (model_copy.Nblts, model_copy.Nfreqs, model_copy.Npols),
            # )
            # self.data_visibilities[time_ind, :, :, :] = np.reshape(
            #     data_copy.data_array,
            #     (data_copy.Nblts, data_copy.Nfreqs, data_copy.Npols),
            # )
            # DEV
            np.random.seed(42)
            self.data_visibilities[time_ind, :, :, :] = np.random.normal(
                0,
                14,
                size=(
                    1,
                    self.Nbls,
                    self.Nfreqs,
                    self.N_vis_pols,
                ),
            ) + 1.0j * np.random.normal(
                0,
                14,
                size=(
                    1,
                    self.Nbls,
                    self.Nfreqs,
                    self.N_vis_pols,
                ),
            )
            self.model_visibilities = self.data_visibilities.copy()

            # flag_array[time_ind, :, :, :] = np.max(
            #     np.stack(
            #         [
            #             np.reshape(
            #                 model_copy.flag_array,
            #                 (model_copy.Nblts, model_copy.Nfreqs, model_copy.Npols),
            #             ),
            #             np.reshape(
            #                 data_copy.flag_array,
            #                 (data_copy.Nblts, data_copy.Nfreqs, data_copy.Npols),
            #             ),
            #         ]
            #     ),
            #     axis=0,
            # )

        # Free memory
        data = model = data_copy = model_copy = None

        # Grab other metadata from uvfits
        self.channel_width = np.mean(metadata_reference.channel_width)
        self.freq_array = np.reshape(metadata_reference.freq_array, (self.Nfreqs))
        self.integration_time = np.mean(metadata_reference.integration_time)
        self.time = np.mean(metadata_reference.time_array)
        self.telescope_name = metadata_reference.telescope_name
        self.lst = np.mean(metadata_reference.lst_array)
        self.telescope_location = metadata_reference.telescope_location

        if (min_cal_baseline_lambda is not None) or (
            max_cal_baseline_lambda is not None
        ):
            baseline_lengths_m = np.sqrt(
                np.sum(metadata_reference.uvw_array**2.0, axis=1)
            )
            baseline_lengths_lambda = (
                baseline_lengths_m[:, np.newaxis]
                * np.reshape(
                    metadata_reference.freq_array, (1, metadata_reference.Nfreqs)
                )
                / 3e8
            )
            flag_array[
                :,
                np.where(
                    (baseline_lengths_lambda < min_cal_baseline_lambda)
                    & (baseline_lengths_lambda > max_cal_baseline_lambda)
                ),
                :,
            ] = True

        # Create gains expand matrices
        self.ant1_inds = np.zeros(self.Nbls, dtype=int)
        self.ant2_inds = np.zeros(self.Nbls, dtype=int)
        self.antenna_numbers = np.unique(
            [metadata_reference.ant_1_array, metadata_reference.ant_2_array]
        )
        for baseline in range(metadata_reference.Nbls):
            self.ant1_inds[baseline] = np.where(
                self.antenna_numbers == metadata_reference.ant_1_array[baseline]
            )[0]
            self.ant2_inds[baseline] = np.where(
                self.antenna_numbers == metadata_reference.ant_2_array[baseline]
            )[0]

        # Get ordered list of antenna names
        self.antenna_names = np.array(
            [
                np.array(metadata_reference.antenna_names)[
                    np.where(metadata_reference.antenna_numbers == ant_num)[0][0]
                ]
                for ant_num in self.antenna_numbers
            ]
        )
        self.antenna_positions = np.array(
            [
                np.array(metadata_reference.antenna_positions)[
                    np.where(metadata_reference.antenna_numbers == ant_num)[0][0], :
                ]
                for ant_num in self.antenna_numbers
            ]
        )

        # Get UV locations
        antpos_ecef = self.antenna_positions + Quantity(
            metadata_reference.telescope.location.geocentric
        ).to_value(
            "m"
        )  # Get antennas positions in ECEF
        antpos_enu = pyuvdata.utils.ENU_from_ECEF(
            antpos_ecef, center_loc=metadata_reference.telescope.location
        )  # Convert to topocentric (East, North, Up or ENU) coords.
        uvw_array = antpos_enu[self.ant1_inds, :] - antpos_enu[self.ant2_inds, :]
        self.uv_array = uvw_array[:, :2]
        self.uv_norm = np.linalg.norm(self.uv_array, axis=0)

        # Get polarization ordering
        self.vis_polarization_array = np.array(metadata_reference.polarization_array)

        if N_feed_pols is None:
            self.N_feed_pols = np.min([2, self.N_vis_pols])
        else:
            self.N_feed_pols = N_feed_pols

        if feed_polarization_array is None:
            self.feed_polarization_array = np.array([], dtype=int)
            if (
                (-5 in self.vis_polarization_array)
                or (-7 in self.vis_polarization_array)
                or (-8 in self.vis_polarization_array)
            ):
                self.feed_polarization_array = np.append(
                    self.feed_polarization_array, -5
                )
            if (
                (-6 in self.vis_polarization_array)
                or (-7 in self.vis_polarization_array)
                or (-8 in self.vis_polarization_array)
            ):
                self.feed_polarization_array = np.append(
                    self.feed_polarization_array, -6
                )
            self.feed_polarization_array = self.feed_polarization_array[
                : self.N_feed_pols
            ]
        else:
            self.feed_polarization_array = feed_polarization_array

        # Initialize gains
        self.gains_multiply_model = gains_multiply_model
        if gain_init_calfile is None:
            self.gains = np.ones(
                (
                    self.Nants,
                    self.Nfreqs,
                    self.N_feed_pols,
                ),
                dtype=complex,
            )
            if gain_init_to_vis_ratio:  # Use mean ratio of visibility amplitudes
                vis_amp_ratio = np.abs(self.model_visibilities) / np.abs(
                    self.data_visibilities
                )
                vis_amp_ratio[np.where(self.data_visibilities == 0.0)] = np.nan
                if self.gains_multiply_model:
                    self.gains[:, :, :] = 1 / np.sqrt(np.nanmedian(vis_amp_ratio))
                else:
                    self.gains[:, :, :] = np.sqrt(np.nanmedian(vis_amp_ratio))
        else:  # Initialize from file
            self.set_gains_from_calfile(gain_init_calfile)
            # Capture nan-ed gains as flags
            for feed_pol_ind, feed_pol in enumerate(self.feed_polarization_array):
                nan_gains = np.where(~np.isfinite(self.gains[:, :, feed_pol_ind]))
                if len(nan_gains[0]) > 0:
                    if feed_pol == -5:
                        flag_pols = np.where(
                            (metadata_reference.polarization_array == -5)
                            | (metadata_reference.polarization_array == -7)
                            | (metadata_reference.polarization_array == -8)
                        )[0]
                    elif feed_pol == -6:
                        flag_pols = np.where(
                            (metadata_reference.polarization_array == -6)
                            | (metadata_reference.polarization_array == -7)
                            | (metadata_reference.polarization_array == -8)
                        )[0]
                    for flag_ind in range(len(nan_gains[0])):
                        flag_bls = np.unique(
                            np.concatenate(
                                (
                                    np.where(self.ant1_inds == nan_gains[0][flag_ind])[
                                        0
                                    ],
                                    np.where(self.ant2_inds == nan_gains[0][flag_ind])[
                                        0
                                    ],
                                )
                            )
                        )
                        flag_freq = nan_gains[1][flag_ind]
                        for flag_pol in flag_pols:
                            flag_array[
                                :,
                                flag_bls,
                                flag_freq,
                                flag_pol,
                            ] = True
                    self.gains[nan_gains, feed_pol_ind] = (
                        0.0  # Nans in the gains produce matrix multiplication errors, set to zero
                    )

        # Free memory
        metadata_reference = None

        # Random perturbation of initial gains
        self.gain_init_stddev = gain_init_stddev
        # if gain_init_stddev != 0.0:
        #     self.gains += np.random.normal(
        #         0.0,
        #         gain_init_stddev,
        #         size=(
        #             self.Nants,
        #             self.Nfreqs,
        #             self.N_feed_pols,
        #         ),
        #     ) + 1.0j * np.random.normal(
        #         0.0,
        #         gain_init_stddev,
        #         size=(
        #             self.Nants,
        #             self.Nfreqs,
        #             self.N_feed_pols,
        #         ),
        #     )

        # Initialize abscal parameters
        self.abscal_params = np.zeros((3, self.Nfreqs, self.N_feed_pols), dtype=float)
        self.abscal_params[0, :, :] = 1.0

        # Initialize unical fitted visibility parameters
        self.fit_vis = self.model_visibilities.copy()
        # Random perturbation with stddev passed by user
        self.fit_vis_init_stddev = fit_vis_init_stddev
        # if fit_vis_init_stddev != 0.0:
        #     self.fit_vis += np.random.normal(
        #         0.0,
        #         self.fit_vis_init_stddev,
        #         size=(
        #             self.Ntimes,
        #             self.Nbls,
        #             1,
        #             1,
        #         ),
        #     ) + 1.0j * np.random.normal(
        #         0.0,
        #         self.fit_vis_init_stddev,
        #         size=(
        #             self.Ntimes,
        #             self.Nbls,
        #             1,
        #             1,
        #         ),
        #     )

        # Initialize visibility weights
        # self.visibility_weights = np.zeros(
        #     (
        #         self.Ntimes,
        #         self.Nbls,
        #         self.Nfreqs,
        #         self.N_vis_pols,
        #     ),
        #     dtype=float,
        # )
        # self.visibility_weights /= self.sigma_t**2

        # if np.max(flag_array):  # Apply flagging
        #     self.visibility_weights[np.where(flag_array)] = 0.0

        # Initialize model weights
        # self.model_weights = np.ones(
        #     (
        #         self.Ntimes,
        #         self.Nbls,
        #         self.Nfreqs,
        #         self.N_vis_pols,
        #     ),
        #     dtype=float,
        # )
        # self.model_weights /= self.sigma_m**2

        self.lambda_val = lambda_val

        # DEV: make copies of original data vis, fit vis, and gains
        self.data_vis_orig = self.data_visibilities.copy() 
        self.fit_vis_orig = self.fit_vis.copy()
        self.gains_orig = self.gains.copy()

        self.glim = glim
        self.ulim = ulim

        self.gain_realizations = gain_realizations
        self.model_realizations = model_realizations

    def expand_in_frequency(self):
        """
        Converts a caldata object into a list of caldata objects each
        corresponding to one frequency.

        Returns
        -------
        caldata_list : list of caldata objects
        """

        caldata_list = []
        for freq_ind in range(self.Nfreqs):
            caldata_per_freq = CalData()
            caldata_per_freq.gains = self.gains[:, [freq_ind], :]
            caldata_per_freq.abscal_params = self.abscal_params[:, [freq_ind], :]
            caldata_per_freq.Nants = self.Nants
            caldata_per_freq.Nbls = self.Nbls
            caldata_per_freq.Ntimes = self.Ntimes
            caldata_per_freq.Nfreqs = 1
            caldata_per_freq.N_feed_pols = self.N_feed_pols
            caldata_per_freq.N_vis_pols = self.N_vis_pols
            caldata_per_freq.feed_polarization_array = self.feed_polarization_array
            caldata_per_freq.vis_polarization_array = self.vis_polarization_array
            caldata_per_freq.model_visibilities = self.model_visibilities[
                :, :, [freq_ind], :
            ]
            caldata_per_freq.data_visibilities = self.data_visibilities[
                :, :, [freq_ind], :
            ]
            caldata_per_freq.visibility_weights = self.visibility_weights[
                :, :, [freq_ind], :
            ]
            caldata_per_freq.ant1_inds = self.ant1_inds
            caldata_per_freq.ant2_inds = self.ant2_inds
            caldata_per_freq.gains_multiply_model = self.gains_multiply_model
            caldata_per_freq.antenna_names = self.antenna_names
            caldata_per_freq.antenna_numbers = self.antenna_numbers
            caldata_per_freq.antenna_positions = self.antenna_positions
            caldata_per_freq.uv_array = self.uv_array
            caldata_per_freq.channel_width = self.channel_width
            caldata_per_freq.freq_array = self.freq_array[[freq_ind]]
            caldata_per_freq.integration_time = self.integration_time
            caldata_per_freq.time = self.time
            caldata_per_freq.telescope_name = self.telescope_name
            caldata_per_freq.lst = self.lst
            caldata_per_freq.telescope_location = self.telescope_location
            caldata_per_freq.lambda_val = self.lambda_val
            if self.dwcal_inv_covariance is not None:
                print(
                    "WARNING: Discarding dwcal_inv_covariance in frequency expansion."
                )
            caldata_per_freq.dwcal_inv_covariance = None
            caldata_per_freq.dwcal_memory_save_mode = None
            caldata_list.append(caldata_per_freq)

        return caldata_list

    def expand_in_polarization(self):
        """
        Converts a caldata object into a list of caldata objects each
        corresponding to one feed polarization. List does not include
        cross-polarization visibilities.

        Returns
        -------
        caldata_list : list of caldata objects
        """

        caldata_list = []
        for feed_pol_ind, pol in enumerate(self.feed_polarization_array):
            caldata_per_pol = CalData()
            sky_pol_ind = np.where(self.vis_polarization_array == pol)[0][0]
            caldata_per_pol.gains = self.gains[:, :, [feed_pol_ind]]
            caldata_per_pol.abscal_params = self.abscal_params[:, :, [feed_pol_ind]]
            caldata_per_pol.Nants = self.Nants
            caldata_per_pol.Nbls = self.Nbls
            caldata_per_pol.Ntimes = self.Ntimes
            caldata_per_pol.Nfreqs = self.Nfreqs
            caldata_per_pol.N_feed_pols = 1
            caldata_per_pol.N_vis_pols = 1
            caldata_per_pol.feed_polarization_array = self.feed_polarization_array[
                [feed_pol_ind]
            ]
            caldata_per_pol.vis_polarization_array = self.vis_polarization_array[
                [sky_pol_ind]
            ]
            caldata_per_pol.model_visibilities = self.model_visibilities[
                :, :, :, [sky_pol_ind]
            ]
            caldata_per_pol.data_visibilities = self.data_visibilities[
                :, :, :, [sky_pol_ind]
            ]
            caldata_per_pol.visibility_weights = self.visibility_weights[
                :, :, :, [sky_pol_ind]
            ]
            caldata_per_pol.ant1_inds = self.ant1_inds
            caldata_per_pol.ant2_inds = self.ant2_inds
            caldata_per_pol.gains_multiply_model = self.gains_multiply_model
            caldata_per_pol.antenna_names = self.antenna_names
            caldata_per_pol.antenna_numbers = self.antenna_numbers
            caldata_per_pol.antenna_positions = self.antenna_positions
            caldata_per_pol.uv_array = self.uv_array
            caldata_per_pol.channel_width = self.channel_width
            caldata_per_pol.freq_array = self.freq_array
            caldata_per_pol.integration_time = self.integration_time
            caldata_per_pol.time = self.time
            caldata_per_pol.telescope_name = self.telescope_name
            caldata_per_pol.lst = self.lst
            caldata_per_pol.telescope_location = self.telescope_location
            caldata_per_pol.lambda_val = self.lambda_val
            if self.dwcal_inv_covariance is not None:
                if np.shape(self.dwcal_inv_covariance)[-1] == 1:
                    caldata_per_pol.dwcal_inv_covariance = self.dwcal_inv_covariance
                else:
                    if self.dwcal_memory_save_mode:
                        caldata_per_pol.dwcal_inv_covariance = (
                            self.dwcal_inv_covariance[:, :, :, [sky_pol_ind]]
                        )
                    else:
                        caldata_per_pol.dwcal_inv_covariance = (
                            self.dwcal_inv_covariance[:, :, :, :, [sky_pol_ind]]
                        )
            caldata_per_pol.dwcal_memory_save_mode = self.dwcal_memory_save_mode
            caldata_list.append(caldata_per_pol)

        return caldata_list

    def convert_to_uvcal(self):
        """
        Generate a pyuvdata UVCal object.

        Returns
        -------
        uvcal : pyuvdata UVCal object
        """

        uvcal = pyuvdata.UVCal()
        uvcal.Nants_data = self.Nants
        uvcal.Nants_telescope = self.Nants
        uvcal.Nfreqs = self.Nfreqs
        uvcal.Njones = self.N_feed_pols
        uvcal.Nspws = 1
        uvcal.Ntimes = 1
        uvcal.antenna_names = self.antenna_names
        uvcal.ant_array = self.antenna_numbers
        uvcal.antenna_numbers = self.antenna_numbers
        uvcal.antenna_positions = self.antenna_positions
        uvcal.cal_style = "sky"
        uvcal.cal_type = "gain"
        uvcal.channel_width = np.full((self.Nfreqs), self.channel_width)
        uvcal.freq_array = self.freq_array
        if self.gains_multiply_model:
            uvcal.gain_convention = "divide"
        else:
            uvcal.gain_convention = "multiply"
        uvcal.history = "calibrated with calico"
        uvcal.integration_time = np.array([self.integration_time])
        uvcal.jones_array = self.feed_polarization_array
        uvcal.spw_array = np.array([0])
        uvcal.telescope_name = self.telescope_name
        uvcal.lst_array = np.array([self.lst])
        uvcal.telescope_location = self.telescope_location
        uvcal.time_array = np.array([self.time])
        uvcal.x_orientation = "east"
        uvcal.gain_array = self.gains[:, :, np.newaxis, :]
        uvcal.ref_antenna_name = "none"
        uvcal.sky_catalog = ""
        uvcal.wide_band = False
        uvcal.flex_spw_id_array = np.zeros(self.Nfreqs, dtype=int)

        # Get flags from nan-ed gains and zeroed weights
        uvcal.flag_array = (np.isnan(self.gains))[:, :, np.newaxis, :]

        # Get flags from visibility_weights
        antenna_weights = np.zeros(
            (self.Nants, self.Nfreqs, self.N_feed_pols), dtype=float
        )
        for ant_ind in range(self.Nants):
            for pol_ind in range(self.N_feed_pols):
                if self.feed_polarization_array[pol_ind] == -5:
                    use_vis_pol_inds_ant1 = np.where(
                        (self.vis_polarization_array == -5)
                        | (self.vis_polarization_array == -7)
                    )[0]
                    use_vis_pol_inds_ant2 = np.where(
                        (self.vis_polarization_array == -5)
                        | (self.vis_polarization_array == -8)
                    )[0]
                elif self.feed_polarization_array[pol_ind] == -6:
                    use_vis_pol_inds_ant1 = np.where(
                        (self.vis_polarization_array == -6)
                        | (self.vis_polarization_array == -8)
                    )[0]
                    use_vis_pol_inds_ant2 = np.where(
                        (self.vis_polarization_array == -6)
                        | (self.vis_polarization_array == -7)
                    )[0]
                ant1_antenna_weights = np.zeros((self.Nfreqs))
                ant2_antenna_weights = np.zeros((self.Nfreqs))
                for vis_pol_ind in use_vis_pol_inds_ant1:
                    ant1_antenna_weights += np.sum(
                        self.visibility_weights[
                            :, np.where(self.ant1_inds == ant_ind)[0], :, vis_pol_ind
                        ],
                        axis=(0, 1),
                    )
                for vis_pol_ind in use_vis_pol_inds_ant2:
                    ant2_antenna_weights += np.sum(
                        self.visibility_weights[
                            :, np.where(self.ant2_inds == ant_ind)[0], :, vis_pol_ind
                        ],
                        axis=(0, 1),
                    )
                antenna_weights[ant_ind, :, pol_ind] = (
                    ant1_antenna_weights + ant2_antenna_weights
                )
        uvcal.flag_array[np.where(antenna_weights[:, :, np.newaxis, :] == 0)] = True

        try:
            uvcal.check()
        except:
            print("ERROR: UVCal check failed.")

        return uvcal

    def sky_based_calibration(
        self,
        xtol=1e-5,
        maxiter=200,
        get_crosspol_phase=True,
        crosspol_phase_strategy="crosspol model",
        parallel=False,
        max_processes=40,
        pool=None,
        verbose=False,
    ):
        """
        Run calibration per polarization. Updates the gains attribute with calibrated values.
        Here the XX and YY visibilities are calibrated individually and the cross-polarization
        phase is applied from the XY and YX visibilities after the fact. Option to parallelize
        calibration across frequency.

        Parameters
        ----------
        xtol : float
            Accuracy tolerance for optimizer. Default 1e-5.
        maxiter : int
            Maximum number of iterations for the optimizer. Default 200.
        get_crosspol_phase : bool
            If True, crosspol phase is calculated. Default True.
        crosspol_phase_strategy : str
            Options are "crosspol model" or "pseudo Stokes V". Used only if
            get_crosspol_phase is True. If "crosspol model", contrains the crosspol
            phase using the crosspol model visibilities. If "pseudo Stokes V", constrains
            crosspol phase by minimizing pseudo Stokes V. Default "crosspol model".
        parallel : bool
            Set to True to parallelize across frequency with multiprocessing.
            Default False if pool is None.
        max_processes : int
            Maximum number of multithreaded processes to use. Applicable only if
            parallel is True and pool is None. If None, uses the multiprocessing
            default. Default 40.
        pool : multiprocessing.pool.Pool or None
            Pool for multiprocessing. If None and parallel is True, a new pool will be
            created. Default None.
        verbose : bool
            Set to True to print optimization outputs. Default False.
        """

        if np.max(self.visibility_weights) == 0.0:
            print("ERROR: All data flagged.")
            sys.stdout.flush()
            self.gains[:, :, :] = np.nan + 1j * np.nan
        else:
            if pool is not None:
                parallel = True
                use_pool = pool
            if parallel:
                caldata_list = self.expand_in_frequency()
                args_list = []
                for freq_ind in range(self.Nfreqs):
                    args = (
                        caldata_list[freq_ind],
                        xtol,
                        maxiter,
                        0,
                        verbose,
                        get_crosspol_phase,
                    )
                    args_list.append(args)
                if pool is None:
                    if max_processes is None:
                        use_pool = multiprocessing.Pool()
                    else:
                        use_pool = multiprocessing.Pool(processes=max_processes)
                gains_fit = use_pool.starmap(
                    calibration_optimization.run_skycal_optimization_per_pol_single_freq,
                    args_list,
                )
                for freq_ind in range(self.Nfreqs):
                    self.gains[:, [freq_ind], :] = gains_fit[freq_ind][:, np.newaxis, :]
                if pool is None:  # Leave things how we found them
                    use_pool.terminate()
            else:
                for freq_ind in range(self.Nfreqs):
                    before_gains_arr = self.gains[:, [freq_ind], :]
                    gains_fit = calibration_optimization.run_skycal_optimization_per_pol_single_freq(
                        self,
                        xtol,
                        maxiter,
                        freq_ind=freq_ind,
                        verbose=verbose,
                        get_crosspol_phase=get_crosspol_phase,
                        crosspol_phase_strategy=crosspol_phase_strategy,
                    )
                    self.gains[:, [freq_ind], :] = gains_fit[:, np.newaxis, :]
                    self.temp_test(before_gains_arr, self.model_visibilities[0, :, freq_ind, :])

    def abscal(self, xtol=1e-5, maxiter=200, verbose=False):
        """
        Run absolute calibration ("abscal"). Updates the abscal_params attribute with calibrated values.

        Parameters
        ----------
        xtol : float
            Accuracy tolerance for optimizer. Default 1e-5.
        maxiter : int
            Maximum number of iterations for the optimizer. Default 200.
        verbose : bool
            Set to True to print optimization outputs. Default False.
        """

        # Expand CalData object into per-frequency objects
        caldata_list = self.expand_in_frequency()

        for freq_ind in range(self.Nfreqs):
            abscal_params = (
                calibration_optimization.run_abscal_optimization_single_freq(
                    caldata_list[freq_ind],
                    xtol,
                    maxiter,
                    verbose=verbose,
                )
            )
            self.abscal_params[:, [freq_ind], :] = abscal_params[:, [0], :]

    def dw_abscal(self, xtol=1e-5, maxiter=200, verbose=False):
        """
        Run absolute calibration ("abscal") with delay weighting. Updates the
        abscal_params attribute with calibrated values.

        Parameters
        ----------
        xtol : float
            Accuracy tolerance for optimizer. Default 1e-5.
        maxiter : int
            Maximum number of iterations for the optimizer. Default 200.
        verbose : bool
            Set to True to print optimization outputs. Default False.
        """

        self.abscal_params = calibration_optimization.run_dw_abscal_optimization(
            self,
            xtol,
            maxiter,
            verbose=verbose,
        )

    def flag_antennas_from_per_ant_cost(
        self,
        flagging_threshold=2.5,
        parallel=False,
        max_processes=40,
        pool=None,
        return_antenna_flag_list=False,
        verbose=True,
    ):
        """
        Flags antennas based on the per-antenna cost function. Updates
        visibility_weights according to the flags. The cost function used is the
        standard "sky-based" per frequency, per polarization cost function evaluated
        in cost_function_calculations.cost_function_single_pol.

        Parameters
        ----------
        self : CalData
        flagging_threshold : float
            Flagging threshold. Per antenna cost values equal to flagging_threshold
            times the mean value will be flagged. Default 2.5.
        parallel : bool
            Set to True to parallelize cost evaluation with multiprocessing. Default
            False if pool is None.
        max_processes : int
            Maximum number of multithreaded processes to use. Applicable only if
            parallel is True and pool is None. If None, uses the multiprocessing
            default. Default 40.
        pool : multiprocessing.pool.Pool or None
            Pool for multiprocessing. If None and parallel is True, a new pool will be
            created for this process and terminated.
        return_antenna_flag_list : bool
            If True, returns list of flagged antennas.
        verbose : bool

        Returns
        -------
        flag_antenna_list : list of str or None
            If return_antenna_flag_list is True, returns a list of flagged antenna names.
        """

        per_ant_cost = calibration_qa.calculate_per_antenna_cost(
            self, parallel=parallel, max_processes=max_processes, pool=pool
        )

        where_finite = np.isfinite(per_ant_cost)
        if np.sum(where_finite) > 0:
            mean_per_ant_cost = np.mean(per_ant_cost[where_finite])
            flag_antenna_list = []
            for pol_ind in range(self.N_feed_pols):
                flag_antenna_inds = np.where(
                    np.logical_or(
                        per_ant_cost[:, pol_ind]
                        > flagging_threshold * mean_per_ant_cost,
                        ~np.isfinite(per_ant_cost[:, pol_ind]),
                    )
                )[0]
                flag_antenna_list.append(self.antenna_names[flag_antenna_inds])

                for ant_ind in flag_antenna_inds:
                    bl_inds_1 = np.where(self.ant1_inds == ant_ind)[0]
                    bl_inds_2 = np.where(self.ant2_inds == ant_ind)[0]
                    if self.feed_polarization_array[pol_ind] == -5:
                        if -5 in self.vis_polarization_array:
                            vis_pol_ind = np.where(self.vis_polarization_array == -5)[0]
                            self.visibility_weights[:, bl_inds_1, :, vis_pol_ind] = 0
                            self.visibility_weights[:, bl_inds_2, :, vis_pol_ind] = 0
                        if -7 in self.vis_polarization_array:
                            vis_pol_ind = np.where(self.vis_polarization_array == -7)[0]
                            self.visibility_weights[:, bl_inds_1, :, vis_pol_ind] = 0
                        if -8 in self.vis_polarization_array:
                            vis_pol_ind = np.where(self.vis_polarization_array == -8)[0]
                            self.visibility_weights[:, bl_inds_2, :, vis_pol_ind] = 0
                    elif self.feed_polarization_array[pol_ind] == -6:
                        if -6 in self.vis_polarization_array:
                            vis_pol_ind = np.where(self.vis_polarization_array == -6)[0]
                            self.visibility_weights[:, bl_inds_1, :, vis_pol_ind] = 0
                            self.visibility_weights[:, bl_inds_2, :, vis_pol_ind] = 0
                        if -7 in self.vis_polarization_array:
                            vis_pol_ind = np.where(self.vis_polarization_array == -7)[0]
                            self.visibility_weights[:, bl_inds_2, :, vis_pol_ind] = 0
                        if -8 in self.vis_polarization_array:
                            vis_pol_ind = np.where(self.vis_polarization_array == -8)[0]
                            self.visibility_weights[:, bl_inds_1, :, vis_pol_ind] = 0

        else:  # Flag everything
            flag_antenna_list = []
            for pol_ind in range(self.N_feed_pols):
                flag_antenna_list.append(self.antenna_names)
            self.visibility_weights[:, :, :, :] = 0

        if verbose:
            print("Completed antenna flagging based on per-antenna cost function.")
            print(f"Flagged antennas: {flag_antenna_list}")
            sys.stdout.flush()

        if return_antenna_flag_list:
            return flag_antenna_list

    def get_dwcal_weights_from_delay_spectra(
        self,
        delay_spectrum_variance,
        bl_length_bin_edges,
        delay_axis,
        oversample_factor=128,
    ):
        """
        This function calculates the matrix that captures delay weighting (or frequency
        covariance). The input is an array of expected variances as a function of baseline
        length and delay.

        Parameters
        ----------
        delay_spectrum_variance : array of float
            Array containing the expected variance as a function of baseline length and delay.
            Shape (Nbins, Ndelays,).
        bl_length_bin_edges : array of float
            Defines the baseline length axis of delay_spectrum_variance. Values correspond to
            limits of each baseline length bin. Shape (Nbins+1,).
        delay_axis : array of float
            Defines the delay axis of delay_spectrum_variance. Shape (Ndelays,).
        oversample_factor : int
            Factor by which to oversample the delay axis. Setting > 1 reduces Fourier aliasing
            effects. Default 128.
        """

        bl_lengths = np.sqrt(np.sum(self.uv_array**2.0, axis=1))
        delay_array_use = np.fft.fftfreq(
            self.Nfreqs * int(oversample_factor), d=self.channel_width
        )
        dwcal_variance_use = np.zeros(
            (
                self.Nbls,
                self.Nfreqs * int(oversample_factor),
            ),
            dtype=float,
        )
        for bl_ind, bl_length in enumerate(bl_lengths):
            bin_ind = np.max(np.where(bl_length_bin_edges <= bl_length)[0])
            if (bin_ind == len(bl_length_bin_edges) - 1) or (
                not bl_length_bin_edges[bin_ind + 1] > bl_length
            ):
                print(
                    f"WARNING: Baseline length range does not cover baseline of length {bl_length} m. Skipping."
                )
                continue
            dwcal_variance_use[bl_ind, :] = np.interp(
                delay_array_use, delay_axis, delay_spectrum_variance[bin_ind, :]
            )

        freq_weighting = np.fft.ifft(1.0 / dwcal_variance_use, axis=1)
        freq_weighting = freq_weighting[
            :, : self.Nfreqs
        ]  # Truncate frequency axis to remove oversampling
        weight_mat = np.zeros((self.Nbls, self.Nfreqs, self.Nfreqs), dtype=complex)
        for freq_ind1 in range(self.Nfreqs):
            for freq_ind2 in range(self.Nfreqs):
                if freq_ind1 < freq_ind2:
                    weight_mat[:, freq_ind1, freq_ind2] = np.conj(
                        freq_weighting[:, np.abs(freq_ind1 - freq_ind2)]
                    )
                else:
                    weight_mat[:, freq_ind1, freq_ind2] = freq_weighting[
                        :, np.abs(freq_ind1 - freq_ind2)
                    ]

        # Use the same matrix for all times and polarizations
        # These are included as variables so that time- and polarization-dependence
        # can be built in later if needed
        use_Ntimes = 1
        use_N_vis_pols = 1

        if self.dwcal_memory_save_mode:
            weight_mat = np.repeat(
                np.repeat(weight_mat[np.newaxis, :, :, np.newaxis], use_Ntimes, axis=0),
                use_N_vis_pols,
                axis=3,
            )
        else:
            weight_mat = np.repeat(
                np.repeat(
                    weight_mat[np.newaxis, :, :, :, np.newaxis], use_Ntimes, axis=0
                ),
                use_N_vis_pols,
                axis=4,
            )

        # Deal with nan-ed values
        if self.dwcal_memory_save_mode:
            nan_weight_indices = np.where(~np.isfinite(np.sum(weight_mat, axis=2)))
        else:
            nan_weight_indices = np.where(~np.isfinite(np.sum(weight_mat, axis=(2, 3))))
        if len(nan_weight_indices[0]) > 0:
            print(
                "WARNING: nan values encountered in DWCal inverse convariance matrix. Updating weights."
            )
            for freq_ind in range(self.Nfreqs):
                self.visibility_weights[:, :, freq_ind, :][nan_weight_indices] = 0
            weight_mat[np.where(~np.isfinite(weight_mat))] = (
                0.0 + 1j * 0.0
            )  # Remove nan values to prevent issues later on

        # Fix normalization
        use_visibility_weights = self.visibility_weights
        if self.Ntimes > use_Ntimes:
            use_visibility_weights = np.mean(use_visibility_weights, axis=0)[
                np.newaxis, :, :, :
            ]
        if self.N_vis_pols > use_N_vis_pols:
            use_visibility_weights = np.mean(use_visibility_weights, axis=3)[
                :, :, :, np.newaxis
            ]
        for time_ind in range(use_Ntimes):
            for vis_pol_ind in range(use_N_vis_pols):
                normalization_numerator = np.sum(
                    use_visibility_weights[time_ind, :, :, vis_pol_ind]
                )
                if self.dwcal_memory_save_mode:
                    normalization_denominator = np.real(
                        np.sum(
                            use_visibility_weights[time_ind, :, :, vis_pol_ind]
                            * weight_mat[time_ind, :, :, vis_pol_ind]
                        )
                    )
                else:
                    normalization_denominator = np.real(
                        np.sum(
                            np.trace(
                                np.sqrt(
                                    use_visibility_weights[
                                        time_ind, :, :, np.newaxis, vis_pol_ind
                                    ]
                                )
                                * np.sqrt(
                                    use_visibility_weights[
                                        time_ind, :, np.newaxis, :, vis_pol_ind
                                    ]
                                )
                                * weight_mat[time_ind, :, :, :, vis_pol_ind],
                                axis1=1,
                                axis2=2,
                            )
                        )
                    )
                normalization_factor = (
                    normalization_numerator / normalization_denominator
                )
                if self.dwcal_memory_save_mode:
                    weight_mat[time_ind, :, :, vis_pol_ind] *= normalization_factor
                else:
                    weight_mat[time_ind, :, :, :, vis_pol_ind] *= normalization_factor

        self.dwcal_inv_covariance = weight_mat

    def pack(self, freq_ind, feed_pol_ind, unical=False):
        """
        This function packs the parameters (for now gains, in future the u-params)
        both real and imaginary parts into a single flattened array for use in
        the optimizing function.
        Parameters
        ----------
        freq_ind : int
            Frequency channel index.
        feed_pol_ind : int
            Feed polarization index.
        unical : bool
            Whether to do unified calibration
        """
        self.gains_real = self.gains[self.ant_inds, freq_ind, feed_pol_ind].real
        self.gains_imag = self.gains[self.ant_inds, freq_ind, feed_pol_ind].imag
        # first time only for now, slice all times eventually
        self.fit_vis_real = self.fit_vis[0, self.bl_inds, freq_ind, feed_pol_ind].real
        self.fit_vis_imag = self.fit_vis[0, self.bl_inds, freq_ind, feed_pol_ind].imag

        # interweave real and imaginary parts of gains
        gains_flattened = np.stack(
                (
                    self.gains_real,
                    self.gains_imag,
                ),
                axis=1,
            ).flatten()
        if not unical:
            return gains_flattened
        else:
            # interweave real and imaginary parts of u's
            fit_vis_flattened = np.stack(
                (
                    self.fit_vis_real,
                    self.fit_vis_imag,
                ),
                axis=1,
            ).flatten()
            params_flattened = np.hstack(
                (
                    gains_flattened,
                    fit_vis_flattened,
                )
            )
            return params_flattened

    def reshape_data(self, freq_ind, vis_pol_ind,unical=False):
        """
        This function reshapes data and model visibilities as well
        as visibility weights (and in the future, model weights)
        for use by the cost function, Jacobian, and Hessian
        wrappers called by the optimizing function.
        Parameters
        ----------
        freq_ind : int
            Frequency channel index.
        vis_pol_ind : int
            Visibility polarization index.
        """
        self.data_vis_reshaped = np.reshape(
            self.data_visibilities[:, :, freq_ind, vis_pol_ind],
            (self.Ntimes, self.Nbls),
        )
        self.model_vis_reshaped = np.reshape(
            self.model_visibilities[:, :, freq_ind, vis_pol_ind],
            (self.Ntimes, self.Nbls),
        )
        # self.model_vis_reshaped = np.zeros((1,self.Nbls))
        # self.model_vis_reshaped[0,:] = self.model_visibilities
        self.vis_weights_reshaped = np.reshape(
            self.visibility_weights[:, :, freq_ind, vis_pol_ind],
            (self.Ntimes, self.Nbls),
        )
        # self.vis_weights_reshaped = np.zeros((1,self.Nbls))
        # self.vis_weights_reshaped[0,:] = self.visibility_weights
        if unical:
            self.model_weights_reshaped = np.reshape(
                self.model_weights[:, :, freq_ind, vis_pol_ind],
                (self.Ntimes, self.Nbls),
            )
            # self.model_weights_reshaped = np.zeros((1,self.Nbls))
            # self.model_weights_reshaped = self.model_weights
    
    def set_ant_inds(self, freq_ind, feed_pol_ind):
        """
        This function sets indices of unflagged antennas
        to be calibrated by calculating visibility weight
        per physical antenna and excluding those with zero.

        Parameters
        ----------
        freq_ind : int
            Frequency channel index.
        feed_pol_ind : int
            Feed polarization index.
        """

        # Sum over times per baseline
        vis_weights_summed = np.sum(
            self.visibility_weights[:, :, freq_ind, feed_pol_ind], axis=0
        )
        weight_per_ant = np.bincount(
            self.ant1_inds,
            weights=vis_weights_summed,
            minlength=self.Nants,
        ) + np.bincount(
            self.ant2_inds,
            weights=vis_weights_summed,
            minlength=self.Nants,
        )
        self.ant_inds = np.where(weight_per_ant > 0.0)[0]

    def set_bl_inds(self, freq_ind, feed_pol_ind):
        weights_summed = np.sum(
            (self.model_weights[:, :, freq_ind, feed_pol_ind]
                + self.visibility_weights[:, :, freq_ind, feed_pol_ind]),
            axis=0,
        )
        self.bl_inds = np.where(weights_summed > 0.0)[0]

    def write_fit_vis(self):
        if not np.any(self.ant_inds):
            print("WARNING: All data was flagged. No u_cal file could be written.")  # update this later to a proper warning
            return
        # fit_vis_uvdata = pyuvdata.UVData()
        # fit_vis_uvdata.data_array = self.fit_vis
        # fit_vis_uvdata._Nants_data = len(self.ant_inds)
        # fit_vis_uvdata.write_uvfits("data/fit_vis_cal_out.uvfits")

    # NOTE: for below funcs, duplicating visibility weights line because
    #       I may want to add different behaviors for that array in future
    #       Also may want to move duplicate step-function line to main
    #       weights function; downside is it removes user choice for custom
    #       functions
    # NOTE: do we want to have a weights class in an imported module to
    #       keep this code clean? Then we could just import, instantiate
    #       an object with these funcs, then call those from the main
    #       weights function with getattr
    def constant_weights(self):
        self.visibility_weights[0,:,0,0] += 1/self.sigma_t**2
        self.model_weights[0,:,0,0] += 1/self.sigma_m**2

    def hard_cutoff_weights(self, threshold):
        self.visibility_weights[0,:,0,0] += 1/self.sigma_t**2
        self.model_weights[0,:,0,0] = np.heaviside(self.uv_norm - threshold, 1)
    
    # NOTE: Be careful that these functions are correctly implemented
    #       I had them ordered before but I don't think that should be
    #       necessary as numpy acts on them in the ordering of uv_norm
    #       which should be the same shape as the weights array. But
    #       double check it to be sure.
    def sigmoid_weights(self, threshold):
        self.visibility_weights[0,:,0,0] += 1/self.sigma_t**2
        self.model_weights[0,:,0,0] += self.sigma_m / (1 + np.exp(-self.uv_norm + threshold))

    def exponential_weights(self, threshold):
        self.visibility_weights += 1/self.sigma_t**2
        self.model_weights[0,:,0,0] = np.heaviside(self.uv_norm - threshold, 1)
        self.model_weights[0,:,0,0][self.uv_norm < threshold] += np.exp(self.uv_norm - threshold)

    def power_law_weights(self, threshold):
        self.visibility_weights += 1/self.sigma_t**2
        self.model_weights[0,:,0,0] = np.heaviside(self.uv_norm - threshold, 1)
        try:
            power = int(power)
        except Exception as error:
            print(f"{error}\n\nMaybe you passed a bad value for power for a power law cutoff?", end="")
            print("Defaulting to power=2")
            power = 2
        self.model_weights[0,:,0,0][self.uv_norm < threshold] += (-1/(self.uv_norm - threshold))**power

    def damped_sinusoid_weights(self, threshold):
        self.visibility_weights[0,:,0,0] += 1/self.sigma_t**2
        self.model_weights[0,:,0,0] = np.heaviside(self.uv_norm - threshold, 1)
        self.model_weights[0,:,0,0][self.uv_norm < threshold] += np.exp(self.uv_norm - threshold) * np.cos(self.uv_norm)**2

    """
        Set thermal noise and model errors for simulation. User-passed arrays for weights will be
        applied before any user-passed constants.

        * antenna_gain_weights - ndarray (Nants,)
        
            * Predefined weights for thermal noise on antenna gains. Expected to
            to be of shape (Nants,) and will convert to baselines. Will be applied
            before any passed sigma constants.

        * model_baseline_weights - ndarray (Nbls,)
        
            * Predefined weights for model error on baseline visibilities.
            Excpected to be of shape (Nbls,). Will be applied before any passed
            sigma constants.

        NOTE: I don't think these are compatible with multiple time steps as-is.

        Below are if the user wishes to generate weights per passed parameters.

        * weights_threshold - int
        
            * Characteristic length of baseline (magnitude of vector in uvw-space)
            below which weights will shrink, i.e. error will grow, to de-prioritize
            them in the calibration solution. Exact behavior associated with this
            threshold is defined by whether or not the cuttoff is hard (below).
            Pass as None with hard_cutoff (below) as True to set weights as constant.

        * hard_cutoff - boolean
        
            * Whether or not weights will be reduced at threshold as a hard cutoff. 
            If true then baselines below threshold will receive minimum weighting 
            approximating zero and if false then baselines will be reduced to same 
            minimum weighting via a continuous function (passed by cutoff_function 
            below; default is sigmoid).

        * cutoff_function - string
        
            * Continuous function to be used to lower weights to approximately zero 
            at the baseline length passed by weights_threshold above; defaults to 
            sigmoid. Supported types are "sigmoid", "exponential" (decay), "power law", ...

        * power - int

            * To be used in power law cutoff (default is power=2)

        * sigma_t - float
        
            * True thermal noise passed to the cost function. This sets the max value 
            of the antenna gain/visibility weight array.

        * sigma_m - float
        
            * True model error passed to the cost function. This sets the max value 
            of the model baseline weight array.

        Below are dev tools:

        * sigma_n - float
        
            * Effective thermal noise determining number of iterated realizations of 
            thermal noise; defaults to sigma_t; zero turns off many realizations for 
            thermal noise.

        * sigma_e - float
        
            * Effective model error determining number of iterated realizations of 
            model error; defaults to sigma_m; zero turns off many realizations for 
            model error.
    """
    def set_weights_and_errors(
            self, 
            antenna_gain_weights=None,
            model_baseline_weights=None,
            weights_threshold=0,
            hard_cutoff=True,
            cutoff_function="sigmoid",
            power=2,
            sigma_t=0.1, 
            sigma_m=0.1, 
            sigma_n=None, 
            sigma_e=None, 
    ):
        self.visibility_weights = np.zeros(
            (
                self.Ntimes,
                self.Nbls,
                self.Nfreqs,
                self.N_vis_pols,
            ),
            dtype=float,
        )
        self.model_weights = np.zeros(
            (
                self.Ntimes,
                self.Nbls,
                self.Nfreqs,
                self.N_vis_pols,
            ),
            dtype=float,
        )

        # set to user-passed weights
        # NOTE: I don't want to return here because 1) I need to set the other
        # and 2) they may have only passed one or the other array. Using a
        # boolean check for the second part of the code might be worth it.
        if antenna_gain_weights:
            from calico import utils
            try:
                self.visibility_weights = antenna_gain_weights  # NOTE: Placeholder; convert antenna weights to baseline space
            except Exception as error:
                print(f"{error}\n\nGain weights cannot be used. Make sure antenna gain weight array has shape (Nants,)")
                self.visibility_weights += 1
            return
        if model_baseline_weights:
            try:
                self.model_weights = model_baseline_weights
            except Exception as error:
                print(f"{error}\n\nModel weights cannot be used. Make sure model baseline weight array has shape (Nbls,)")
                self.model_weights += 1
            return
        
        # true noise and error
        self.sigma_t = sigma_t
        self.sigma_m = sigma_m
        # default effective thermal noise to true model noise (for many realizations)
        if sigma_n == None:
            self.sigma_n = sigma_t
        else:
            self.sigma_n = sigma_n
        # default effective model error to true model error (for many realizations)
        if sigma_e == None:
            self.sigma_e = sigma_m
        else:
            self.sigma_e = sigma_e

        # set weights according to user's passed function
        try:
            getattr(self, cutoff_function)
        except Exception as error:
            print(f"{error}\n\nMaybe you passed in a bad cutoff function? Defaulting to hard cutoff")
            self.hard_cutoff_weights(weights_threshold)
        sorted_weight_array /= self.sigma_m**2

    def unified_calibration(
        self,
        xtol=1e-5,
        maxiter=200,
        get_crosspol_phase=True,
        crosspol_phase_strategy="crosspol model",
        parallel=False,
        max_processes=40,
        pool=None,
        verbose=False,
    ):
        """
        Run calibration per polarization. Updates the gains attribute and u parameters with
        calibrated values. Here the XX and YY visibilities are calibrated individually and 
        the cross-polarization phase is applied from the XY and YX visibilities after the fact
        to the gains (models later?). Option to parallelize calibration across frequency.

        Parameters
        ----------
        xtol : float
            Accuracy tolerance for optimizer. Default 1e-5.
        maxiter : int
            Maximum number of iterations for the optimizer. Default 200.
        get_crosspol_phase : bool
            If True, crosspol phase is calculated. Default True.
        crosspol_phase_strategy : str
            Options are "crosspol model" or "pseudo Stokes V". Used only if
            get_crosspol_phase is True. If "crosspol model", contrains the crosspol
            phase using the crosspol model visibilities. If "pseudo Stokes V", constrains
            crosspol phase by minimizing pseudo Stokes V. Default "crosspol model".
        parallel : bool
            Set to True to parallelize across frequency with multiprocessing.
            Default False if pool is None.
        max_processes : int
            Maximum number of multithreaded processes to use. Applicable only if
            parallel is True and pool is None. If None, uses the multiprocessing
            default. Default 40.
        pool : multiprocessing.pool.Pool or None
            Pool for multiprocessing. If None and parallel is True, a new pool will be
            created. Default None.
        verbose : bool
            Set to True to print optimization outputs. Default False.
        """

        if np.max(self.visibility_weights) == 0.0:
            print("ERROR: All data flagged.")
            sys.stdout.flush()
            self.gains[:, :, :] = np.nan + 1j * np.nan
        else:
            if pool is not None:
                parallel = True
                use_pool = pool
            if parallel:
                caldata_list = self.expand_in_frequency()
                args_list = []
                for freq_ind in range(self.Nfreqs):
                    args = (
                        caldata_list[freq_ind],
                        xtol,
                        maxiter,
                        0,
                        verbose,
                        get_crosspol_phase,
                    )
                    args_list.append(args)
                if pool is None:
                    if max_processes is None:
                        use_pool = multiprocessing.Pool()
                    else:
                        use_pool = multiprocessing.Pool(processes=max_processes)
                gains_fit = use_pool.starmap(
                    calibration_optimization.run_unical_optimization,
                    args_list,
                )
                for freq_ind in range(self.Nfreqs):
                    self.gains[:, [freq_ind], :] = gains_fit[freq_ind][:, np.newaxis, :]
                if pool is None:  # Leave things how we found them
                    use_pool.terminate()
            else:
                if self.sigma_n != 0.0:
                    # initial thermal noise
                    try:
                        self.data_visibilities += np.random.normal(
                            0.0,
                            self.sigma_t,
                            size=(
                                self.Ntimes,
                                self.Nbls,
                                1,
                                1,
                            ),
                        ) + 1.0j * np.random.normal(
                            0.0,
                            self.sigma_t,
                            size=(
                                self.Ntimes,
                                self.Nbls,
                                1,
                                1,
                            ),
                        )
                    except Exception as e:
                        print(type(e))
                        print("Initial thermal noise failed. Was sigma_t set correctly?")
                if self.sigma_e != 0.0:
                    # initial model error
                    try:
                        self.model_visibilities += np.random.normal(
                            0.0,
                            self.sigma_m,
                            size=(
                                self.Ntimes,
                                self.Nbls,
                                1,
                                1,
                            ),
                        ) + 1.0j * np.random.normal(
                            0.0,
                            self.sigma_m,
                            size=(
                                self.Ntimes,
                                self.Nbls,
                                1,
                                1,
                            ),
                        )
                    except Exception as e:
                        print(type(e))
                        print("Initial model error failed. Was sigma_m set correctly?")
                for freq_ind in range(self.Nfreqs):
                    self.gain_params_realizations = np.array([])
                    self.model_params_realizations = np.array([])
                    gain_error_per_realization = []
                    realizations = self.gain_realizations if self.gain_realizations >= self.model_realizations else self.model_realizations
                    actual_gain_realizations = 0
                    actual_model_realizations = 0
                    outlier_ants_1 = set()
                    outlier_ants_2 = set()
                    for i in range(realizations):
                        # reset arrays
                        self.gains = self.gains_orig.copy()
                        self.data_visibilities = self.data_vis_orig.copy()
                        self.fit_vis = self.fit_vis_orig.copy()
                        # reset seed
                        import time
                        np.random.seed(i)
                        if i % (realizations / self.gain_realizations) == 0 and self.sigma_n != 0.0:
                            # simulate thermal noise
                            try:
                                actual_gain_realizations += 1
                                # self.data_visibilities[:, :, :, :] = np.random.normal(
                                #     0,
                                #     14,
                                #     size=(
                                #         1,
                                #         self.Nbls,
                                #         self.Nfreqs,
                                #         self.N_vis_pols,
                                #     ),
                                # ) + 1.0j * np.random.normal(
                                #     0,
                                #     14,
                                #     size=(
                                #         1,
                                #         self.Nbls,
                                #         self.Nfreqs,
                                #         self.N_vis_pols,
                                #     ),
                                # )
                                self.data_visibilities += np.random.normal(
                                    0.0,
                                    self.sigma_t,
                                    size=(
                                        self.Ntimes,
                                        self.Nbls,
                                        1,
                                        1,
                                    ),
                                ) + 1.0j * np.random.normal(
                                    0.0,
                                    self.sigma_t,
                                    size=(
                                        self.Ntimes,
                                        self.Nbls,
                                        1,
                                        1,
                                    ),
                                )
                            except Exception as e:
                                print(type(e))
                                print(f"Thermal noise realization {i} failed. Was sigma_t set correctly?")
                        # simulate model error
                        if i % (realizations / self.model_realizations) == 0 and self.sigma_e != 0.0:
                            try:
                                actual_model_realizations += 1
                                self.model_visibilities += np.random.normal(
                                    0.0,
                                    self.sigma_m,
                                    size=(
                                        self.Ntimes,
                                        self.Nbls,
                                        1,
                                        1,
                                    ),
                                ) + 1.0j * np.random.normal(
                                    0.0,
                                    self.sigma_m,
                                    size=(
                                        self.Ntimes,
                                        self.Nbls,
                                        1,
                                        1,
                                    ),
                                )
                            except Exception as e:
                                print(type(e))
                                print(f"Model error realization {i} failed. Was sigma_m set correctly?")
                        # main unical code
                        before_arr_gains = self.gains[:, [freq_ind], :].copy()
                        before_arr_u = self.fit_vis[:, :, [freq_ind], :].copy()
                        dev = dev_tools.DevTools()
                        gains_fit, fit_vis_fit = calibration_optimization.run_unical_optimization(
                            self,
                            xtol,
                            maxiter,
                            dev,
                            freq_ind=freq_ind,
                            verbose=verbose,
                            get_crosspol_phase=get_crosspol_phase,
                            crosspol_phase_strategy=crosspol_phase_strategy,
                        )
                        self.gains[:, [freq_ind], :] = gains_fit[:, np.newaxis, :].copy()
                        self.fit_vis[:1, :, [freq_ind], :] = fit_vis_fit[np.newaxis, :, :, np.newaxis].copy()
                        
                        # identify biggest gain and model errors over time
                        gains_error = self.gains[:,0,0].copy()
                        # gains_error.real -= 1  # do this in dev tool code
                        models_error = self.fit_vis[0,:,0,0].copy()
                        models_error -= self.model_visibilities[0,:,0,0]

                        # combine arrays
                        self.gain_params_realizations = np.concatenate((self.gain_params_realizations, gains_error))
                        self.model_params_realizations = np.concatenate((self.model_params_realizations, models_error))
                        # outlier_ants_1.update(self.ant1_inds[np.nonzero(np.abs(gains_error) > 1)])
                        # outlier_ants_2.update(self.ant2_inds[np.nonzero(np.abs(gains_error) > 2)])

                        gain_error_per_realization.append(gains_error)
                        
                        # import utils
                        
                        # minlength = np.max([np.max(self.ant1_inds), np.max(self.ant2_inds)]) + 1
                        # ant1_bincount = utils.bincount_multidim(self.ant1_inds, minlength=minlength)
                        # ant2_bincount = utils.bincount_multidim(self.ant2_inds, minlength=minlength)

                        # true_vis_ant1_avg = utils.bincount_multidim(
                        #     self.ant1_inds,
                        #     weights=np.abs(self.data_visibilities[0,:,0,0]),  # should be generalized to any time, pol, or freq
                        #     minlength=minlength,
                        # ) / ant1_bincount
                        # true_vis_ant2_avg = utils.bincount_multidim(
                        #     self.ant2_inds,
                        #     weights=np.abs(self.data_visibilities[0,:,0,0]),  # should be generalized to any time, pol, or freq
                        #     minlength=minlength,
                        # ) / ant2_bincount
                        # true_vis_ants = (true_vis_ant1_avg + true_vis_ant2_avg) / 2

                        # model_vis_ant1_avg = utils.bincount_multidim(
                        #     self.ant1_inds,
                        #     weights=np.abs(self.model_visibilities[0,:,0,0]),  # should be generalized to any time, pol, or freq
                        #     minlength=minlength,
                        # ) / ant1_bincount
                        # model_vis_ant2_avg = utils.bincount_multidim(
                        #     self.ant2_inds,
                        #     weights=np.abs(self.model_visibilities[0,:,0,0]),  # should be generalized to any time, pol, or freq
                        #     minlength=minlength,
                        # ) / ant2_bincount
                        # model_vis_ants = (model_vis_ant1_avg + model_vis_ant2_avg) / 2

                        # u_minus_v_ant1_avg = utils.bincount_multidim(
                        #     self.ant1_inds,
                        #     weights=np.abs(self.fit_vis[0,:,0,0] - self.data_visibilities[0,:,0,0]),  # should be generalized to any time, pol, or freq
                        #     minlength=minlength,
                        # ) / ant1_bincount
                        # u_minus_v_ant2_avg = utils.bincount_multidim(
                        #     self.ant2_inds,
                        #     weights=np.abs(self.fit_vis[0,:,0,0] - self.data_visibilities[0,:,0,0]),  # should be generalized to any time, pol, or freq
                        #     minlength=np.max([np.max(self.ant1_inds), np.max(self.ant2_inds)]) + 1,
                        # ) / ant2_bincount
                        # u_minus_v_ants = (u_minus_v_ant1_avg + u_minus_v_ant2_avg) / 2

                        # import matplotlib.pyplot as plt
                        # import math

                        # plt.scatter(true_vis_ants, np.abs(self.gains[:,0,0]), color="blue")
                        # # plt.plot(range(math.ceil(np.max(true_vis_ants))), [1 for i in range(math.ceil(np.max(true_vis_ants)))], color="red")
                        # plt.xlabel("<|v|>")
                        # plt.ylabel("|g|")
                        # plt.title("Gains vs Avg Data Vis per Ant")
                        # plt.show()
                        
                        # plt.scatter(model_vis_ants, np.abs(self.gains[:,0,0]), color="blue")
                        # # plt.plot(range(math.ceil((np.max(model_vis_ants)))), [1 for i in range(math.ceil(np.max(model_vis_ants)))], color="red")
                        # plt.xlabel("<|m|>")
                        # plt.ylabel("|g|")
                        # plt.title("Gains vs Avg Model Vis per Ant")
                        # plt.show()

                        # plt.scatter(u_minus_v_ants, np.abs(self.gains[:,0,0]), color="blue")
                        # # plt.plot(range(math.ceil(np.max(u_minus_v_ants))), [1 for i in range(math.ceil(np.max(u_minus_v_ants)))], color="red")
                        # plt.xlabel("<|u-v|>")
                        # plt.ylabel("|g|")
                        # plt.title("Gains vs Avg Fitted Vis Minus Data Vis per Ant")
                        # plt.show()
                    
                    # dev.plot_gain_error_per_realization(
                    #     gain_error_per_realization,
                    #     plot_type="outlier"
                    # )

                    # plot param changes
                    def sim_error_type():
                        if self.sigma_e == 0.0:
                            return "thermal"
                        elif self.sigma_n == 0.0:
                            return "model"
                        else:
                            return "both"
                    dev.plot_change_in_gain_and_model_params(
                        self.gain_params_realizations,
                        self.model_params_realizations,
                        type="histogram",
                        glim=self.glim,
                        ulim=self.ulim,
                        error_type=sim_error_type(),
                        stddev_thermal=self.sigma_t,
                        stddev_model=self.sigma_m,
                    )

                    # num_vis_total = utils.bincount_multidim(self.ant1_inds) + utils.bincount_multidim(self.ant2_inds)
                    # outlier_ants_1 = np.array(list(outlier_ants_1))
                    # outlier_ants_2 = np.array(list(outlier_ants_2))
                    # num_vis_outliers = np.zeros(self.Nants)
                    # num_vis_total = np.zeros(self.Nants)
                    # for ant in np.unique(np.concatenate((self.ant1_inds, self.ant2_inds))):
                    #     num_vis_total[ant] = np.size(np.nonzero(self.ant1_inds == ant)) + np.size(np.nonzero(self.ant2_inds == ant))
                    #     num_vis_outliers[ant] = np.size(np.nonzero(outlier_ants_1 == ant)) + np.size(np.nonzero(outlier_ants_2 == ant))
                    
                    # # plot biggest gain and model errors over time
                    # import matplotlib.pyplot as plt
                    # import subprocess

                    # fig, ax = plt.subplots()
                    # plt.bar(range(self.Nants), num_vis_total, color="blue", alpha=0.5, label="total", width=1.0)
                    # plt.bar(range(self.Nants), num_vis_outliers, color="red", alpha=0.5, label="outliers", width=1.0, bottom=num_vis_total)
                    # plt.title("Baselines per antenna")
                    # plt.xlabel("Ant #")
                    # plt.ylabel("# of Visibilities")
                    # plt.legend(loc="upper right")
                    # plt.show()

                    #self.write_fit_vis()