import sys

import numpy as np


class VariableWeightsArray:
    """
    Object for setting calibration weights using thermal noise and
    model error uncertainties, as well as simulation variances for
    generating "true" errors for testing, as a function of baseline
    length.

    Attributes
    ----------
    uv_norm_array :
        * Shape (Ntimes, Nbls)
        * Array of the norms of vectors in the uv plane per baseline
          i.e. the baseline length, in wavelengths.
    threshold_length : float
        * Baseline length, in wavelengths, above which all baselines should receive
          one form of weighting and below which all baselines should
          receive another per passed weigting function, e.g. lower
          weighting and higher error below the threshold to reflect
          greater model error on shorter baselines. Typically this
          will act as an offset f(x) -> f(x - a) depending on the
          weighting function.
          * NOTE: Currently all weighting functions support decreasing
          weights i.e. increasing uncertainty for shorter baselines.
    threshold_mask : boolean array
        * shape (Nbls,)
        * Mask set by the threshold baseline length, commonly used
          by different weighting functions.
    power : int
        * Power to be used in power law weighting function. Defaults
          to 2.
    scaling_factor : float
        * Overall scale of the model error weights to multiply weighting
          functions. Default is 1.
    step_down_level : float
        * Weight below the threshold length for step_down_weights, as a
          fraction of the full weight. Default is 0.5.
    transition_width : float
        * Width in wavelengths of the transition for the sigmoid,
          exponential, and damped sinusoid weighting functions.
          Default is 1.
    thermal_noise_weights_array : array of float
        * Shape (Ntimes, Nbls, Nfreqs, N_vis_pols)
        * Array of weights corresponding to inverse sigma_t squared
        where sigma_t is the thermal noise in the data.
        * If initialized according to a user preference, this array
        will be directly applied to the thermal noise calico weights
        array.
        * If uninitialized, the thermal noise calico weights array
        will be set according to the parameters passed by the user
        regarding characteristic model values (which are the user's
        responsibility to ensure are appropriate to their problem),
        the desired weighting function (which may be custom written
        by the user), and a threshold baseline length.
    model_error_weights_array : array of float
        * Shape (Ntimes, Nbls, Nfreqs, N_vis_pols)
        * Array of weights corresponding to inverse sigma_m squared
        where sigma_m is a characteristic error from missing sources
        in the model.
        * If initialized according to a user preference, this array
        will be directly applied to the model error calico weights
        array.
        * If uninitialized, the model error calico weights array
        will be set according to the parameters passed by the user
        regarding characteristic model values (which are the user's
        responsibility to ensure are appropriate to their problem),
        the desired weighting function (which may be custom written
        by the user), and a threshold baseline length.
    """

    def __init__(self):
        self.uv_norm_array = None
        self.threshold_length = None
        self.weighting_function = ""
        self.power = 0
        self.scaling_factor = (0.0,)
        self.step_down_level = 0.5
        self.transition_width = 1
        self.thermal_noise_weight_array = (None,)
        self.model_error_weight_array = (None,)

    def set_algorithm_weights(
        self,
        caldata_obj,
        freq_ind=0,
        pol_ind=0,
        sigma_t_0=0.1,
        sigma_m_0=0.1,
        threshold_length=None,
        weighting_function="constant_weights",
        power=2,
        scaling_factor=1,
        step_down_level=0.5,
        transition_width=1,
        thermal_noise_weight_array=None,
        model_error_weight_array=None,
    ):
        """
        This function sets weights according either to a passed user
        array or dynamically according other passed parameters.

        Parameters
        ----------
        threshold_length : float
            * Baseline length in wavelengths used by the weighting
            function. Required unless both user weight arrays are passed.
        scaling_factor : float
            * Overall scale of the model error weights; the weighting
            function's profile (between 0 and 1) is multiplied by it.
        step_down_level : float
            * Weight below the threshold for step_down_weights, as a
            fraction of the full weight. Default is 0.5.
        transition_width : float
            * Width in wavelengths of the transition for sigmoid_weights,
            exponential_weights, and damped_sinusoid_weights; baseline
            lengths are measured from the threshold in units of this.
            Must be positive. Default is 1.
        thermal_noise_weight_array : array of float, optional
            * Shape (Ntimes, Nbls, Nfreqs, N_vis_pols)
            * User weights (inverse sigma_t squared) applied directly to
            caldata_obj.visibility_weights in place of the computed ones.
            Not divided by sigma_t_0 squared.
        model_error_weight_array : array of float, optional
            * Shape (Ntimes, Nbls, Nfreqs, N_vis_pols)
            * User weights (inverse sigma_m squared) applied directly to
            caldata_obj.model_weights in place of the computed ones.
            Not divided by sigma_m_0 squared.
        sigma_t_0 : float
            * The characteristic thermal noise for the user's problem.
            Calculations by weight functions treat this value as unity.
            * It is important that the user makes the scale and units
            comparable to sigma_m_0 since in unified calibration the
            two cost function terms are coupled by the u parameters.
        sigma_m_0 : float
            * The characteristic model error for the user's problem.
            Calculations by weight functions treat this value as unity.
            * It is important that the user makes the scale and units
            comparable to sigma_m_0 since in unified calibration the
            two cost function terms are coupled by the u parameters.
        weighting_function : string
            * One of several functions for distributing weights as a
            function of baseline length and the threshold baseline
            length. The weights are calculated in units of sigma_m_0
            and sigma_t_0 which are then multiplied across the respective
            arrays.
            * Support is built-in for:
                * constant_weights : This is a constant weight across all
                baselines set by sigma_t_0 and sigma_m_0. This is the
                default function.
                * hard_cutoff_weights : This is a Heaviside step function
                at the threshold baseline length, with all baselines
                below having "infinite" uncertainty i.e. zero weights
                * step_down_weights : This a modified Heaviside step function
                where below threshold is set to some lower, nonzero
                weights i.e. higher, finite uncertaintities according to
                step_down_level.
                * sigmoid_weights : This is a sigmoid "logistics" function
                that is centered at the threshold baseline length, smoothly
                increasing from zero weighting below to full value
                above the threshold.
                * exponential_weights : This exponentially raises the weights
                from zero at zero baseline length to full value at the
                threshold baseline length.
                * power_law_weights : This uses a power law set by "power"
                to raise the weights from zero at zero baseline length to
                full value at the threshold baseline length.
                * damped_sinusoid_weights : This uses the square of a
                (mirrored) damped sinusoid to transition the weights from
                zero at zero baseline length to full value at the
                threshold baseline length.
            * NOTE: The user may write their own function (as a Python
            function here in this file) and use that so long as the
            string exactly matches the function name, and the necessary
            parameters are added as class attributes.
        """
        use_user_thermal = thermal_noise_weight_array is not None
        use_user_model = model_error_weight_array is not None
        if threshold_length is None and not (use_user_thermal and use_user_model):
            raise ValueError("Need threshold length even if zero -- Variable Weights")

        caldata_obj.sigma_t_0 = sigma_t_0
        caldata_obj.sigma_m_0 = sigma_m_0

        self.power = power
        self.weighting_function = weighting_function
        self.step_down_level = step_down_level
        if transition_width <= 0:
            raise ValueError("transition_width must be positive -- Variable Weights")
        self.transition_width = transition_width

        if not (use_user_thermal and use_user_model):
            self.thermal_noise_weight_array = np.zeros_like(
                caldata_obj.data_visibilities, dtype=np.float64
            )
            self.model_error_weight_array = np.zeros_like(
                caldata_obj.data_visibilities, dtype=np.float64
            )

            self.threshold_length = threshold_length
            # baseline lengths in wavelengths at the frequency being weighted
            self.uv_norm_array = (
                np.linalg.norm(caldata_obj.uv_blt_array, axis=1)
                * caldata_obj.freq_array[freq_ind]
                / 3e8
            )
            self.uv_norm_array = np.reshape(
                self.uv_norm_array, (caldata_obj.Ntimes, caldata_obj.Nbls)
            )
            self.scaling_factor = scaling_factor

            caldata_obj.threshold_mask = self.uv_norm_array < threshold_length

            getattr(self, self.weighting_function)(caldata_obj, freq_ind, pol_ind)

            # weights from the weighting functions are in units of the
            # characteristic sigmas
            self.thermal_noise_weight_array /= sigma_t_0**2
            self.model_error_weight_array /= sigma_m_0**2

        # user arrays are already inverse variances, so they are used as is
        if use_user_thermal:
            self.thermal_noise_weight_array = self._check_user_weight_array(
                thermal_noise_weight_array, caldata_obj, "thermal_noise_weight_array"
            )
        if use_user_model:
            self.model_error_weight_array = self._check_user_weight_array(
                model_error_weight_array, caldata_obj, "model_error_weight_array"
            )

        caldata_obj.visibility_weights = self.thermal_noise_weight_array
        caldata_obj.model_weights = self.model_error_weight_array

    @staticmethod
    def _check_user_weight_array(weight_array, caldata_obj, name):
        weight_array = np.array(weight_array, dtype=np.float64)
        if weight_array.shape != caldata_obj.data_visibilities.shape:
            raise ValueError(
                f"{name} must have shape {caldata_obj.data_visibilities.shape} "
                f"(Ntimes, Nbls, Nfreqs, N_vis_pols), got {weight_array.shape}"
            )
        return weight_array

    # each weighting function builds a profile in [0, 1] over self.uv_norm_array
    # (Ntimes, Nbls) that reaches 1 at and above the threshold length; the
    # model weights are that profile times scaling_factor
    def _set_weights_from_profile(self, profile, freq_ind, pol_ind):
        self.thermal_noise_weight_array[:, :, freq_ind, pol_ind] += 1
        self.model_error_weight_array[:, :, freq_ind, pol_ind] = (
            profile * self.scaling_factor
        )

    # baseline length minus the threshold, in units of the transition width
    def _distance_from_threshold(self):
        return (self.uv_norm_array - self.threshold_length) / self.transition_width

    def constant_weights(self, caldata_obj, freq_ind, pol_ind):
        self._set_weights_from_profile(
            np.ones_like(self.uv_norm_array), freq_ind, pol_ind
        )

    def hard_cutoff_weights(self, caldata_obj, freq_ind, pol_ind):
        profile = np.heaviside(self.uv_norm_array - self.threshold_length, 1)
        self._set_weights_from_profile(profile, freq_ind, pol_ind)

    def sigmoid_weights(self, caldata_obj, freq_ind, pol_ind):
        # tanh form of the logistic function avoids overflow in np.exp
        x = self._distance_from_threshold()
        profile = 0.5 * (1 + np.tanh(x / 2))
        self._set_weights_from_profile(profile, freq_ind, pol_ind)

    def exponential_weights(self, caldata_obj, freq_ind, pol_ind):
        x = np.minimum(self._distance_from_threshold(), 0)
        profile = np.exp(x)
        self._set_weights_from_profile(profile, freq_ind, pol_ind)

    def power_law_weights(self, caldata_obj, freq_ind, pol_ind):
        try:
            self.power = int(self.power)
        except TypeError as e:
            print(sys.exc_info())
            print(f"Error: {e}", end=" ")
            print("Defaulting to power=2")
            self.power = 2
        if self.threshold_length == 0:
            profile = np.ones_like(self.uv_norm_array)
        else:
            x = np.minimum(self.uv_norm_array / self.threshold_length, 1)
            profile = x**self.power
        self._set_weights_from_profile(profile, freq_ind, pol_ind)

    def damped_sinusoid_weights(self, caldata_obj, freq_ind, pol_ind):
        x = np.minimum(self._distance_from_threshold(), 0)
        profile = np.exp(x) * np.cos(x) ** 2
        self._set_weights_from_profile(profile, freq_ind, pol_ind)

    def step_down_weights(self, caldata_obj, freq_ind, pol_ind):
        profile = np.where(caldata_obj.threshold_mask, self.step_down_level, 1.0)
        self._set_weights_from_profile(profile, freq_ind, pol_ind)

    # basic plot of weights per baseline
    def plot_weights_per_baseline(
        self,
        caldata_obj,
        freq_ind,
        pol_ind,
        scaling_factor=None,
        ylim=None,
        filename_suffix="",
    ):
        if scaling_factor is None:
            scaling_factor = self.scaling_factor
        from calico.dev import dev_tools

        dev = dev_tools.DevTools()
        dev.plot_weights_per_baseline(
            self.uv_norm_array,
            caldata_obj.model_weights[:, :, freq_ind, pol_ind],
            weighting_function=self.weighting_function,
            scaling_factor=scaling_factor,
            threshold_length=self.threshold_length,
            sigma="sigma_m",
            ylim=ylim,
            filename_suffix=filename_suffix,
        )
