import numpy as np
import sys

class VariableWeightsArray:
    """
    Object for setting calibration weights using thermal noise and
    model error uncertainties, as well as simulation variances for
    generating "true" errors for testing, as a function of baseline
    length.

    Attributes
    ------- 
    uv_norm_array : 
        * Shape (Nbls,)
        * Array of the norms of vectors in the uv plane per baseline
          i.e. the baseline length.
    threshold_length : float
        * Baseline length above which all baselines should receive
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
        * How much weights for baselines below the threshold length 
          should be scaled by in relevant weighting functions. Values 
          less than 1 lower weights. Default is 1.
    """

    def __init__(self):
        self.uv_norm_array = None
        self.threshold_length = 0.0
        self.threshold_mask = None
        self.weighting_function = ""
        self.power = 0
        self.scaling_factor = 0.0

    def set_weights(
        self,
        caldata_obj,
        thermal_noise_weight_array = None,
        model_error_weight_array = None,
        sigma_t_0 = 0.1,
        sigma_m_0 = 0.1,
        sigma_n_0 = None,
        sigma_e_0 = None,
        threshold_length = 50.0,
        weighting_function = "constant_weights",
        power = 2,
        scaling_factor = 1,
    ):
        """
        This function sets weights according either to a passed user
        array or dynamically according other passed parameters.

        Parameters
        ------- 
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
        sigma_n_0 : float
            * The characteristic thermal noise to be used in simulating
            data visibilities with thermal noise. The default value of
            None will set this equal to sigma_t_0. 
            * NOTE: Most users will not need to use this unless writing
            their own unified calibration implementations.
        sigma_e_0 : float
            * The characteristic thermal noise to be used in simulating
            data visibilities with thermal noise. The default value of
            None will set this equal to sigma_t_0. 
            * NOTE: Most users will not need to use this unless writing
            their own unified calibration implementations.
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
                the scaling_factor.
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
        caldata_obj.sigma_t_0 = sigma_t_0
        caldata_obj.sigma_m_0 = sigma_m_0
        # for simulations
        caldata_obj.sigma_n_0 = sigma_n_0 if sigma_n_0 else sigma_t_0
        caldata_obj.sigma_e_0 = sigma_e_0 if sigma_e_0 else sigma_m_0

        self.power = power

        # set the weight arrays to user arrays if passed
        if thermal_noise_weight_array:
            try:
                caldata_obj.visibility_weights = thermal_noise_weight_array
            except:
                print(sys.exc_info())
                print("Thermal noise weights can't be used.\nDefaulting to constant weights")
                caldata_obj.visibility_weights = np.ones(
                    caldata_obj.Ntimes,
                    caldata_obj.Nbls,
                    caldata_obj.Nfreqs,
                    caldata_obj.N_vis_pols,
                )
            try:
                caldata_obj.model_weights = model_error_weight_array
            except:
                print(sys.exc_info())
                print("Model error weights can't be used. Defaulting to constant weights")
                caldata_obj.model_weights = np.ones(
                    caldata_obj.Ntimes,
                    caldata_obj.Nbls,
                    caldata_obj.Nfreqs,
                    caldata_obj.N_vis_pols,
                )
        else:
            thermal_noise_weight_array = np.zeros(
                (
                    caldata_obj.Ntimes,
                    caldata_obj.Nbls,
                    caldata_obj.Nfreqs,
                    caldata_obj.N_vis_pols,
                ),
                dtype=float,
            )
            model_error_weight_array = np.zeros(
                (
                    caldata_obj.Ntimes,
                    caldata_obj.Nbls,
                    caldata_obj.Nfreqs,
                    caldata_obj.N_vis_pols,
                ),
                dtype=float,
            )

            self.threshold_length = threshold_length
            self.uv_norm_array = np.linalg.norm(caldata_obj.uv_array, axis=1)
            self.threshold_mask = self.uv_norm_array < threshold_length
            self.scaling_factor = scaling_factor

            try:
                getattr(weighting_function)()
            except:
                print(sys.exc_info())
                print("Maybe you passed in a bad weighting function?")
                print("Defaulting to constant weights.")
                self.hard_cutoff_weights()

            caldata_obj.visibility_weights = thermal_noise_weight_array
            caldata_obj.model_weights = model_error_weight_array

        caldata_obj.visibility_weights *= caldata_obj.sigma_t_0
        caldata_obj.model_weights *= caldata_obj.sigma_m_0

    # NOTE: do we want to have a weights class in an imported module to
    #       keep this code clean? Then we could just import, instantiate
    #       an object with these funcs, then call those from the main
    #       weights function with getattr
    def constant_weights(self):
        self.visibility_weights[0,:,0,0] += 1/self.sigma_t_0**2
        self.model_error_weights_array[0,:,0,0] += 1/self.sigma_m_0**2

    def hard_cutoff_weights(self):
        self.visibility_weights[0,:,0,0] += 1/self.sigma_t_0**2
        self.model_error_weights_array[0,:,0,0] = np.heaviside(self.uv_norm - self.threshold_length, 1)
    
    def sigmoid_weights(self):
        self.visibility_weights[0,:,0,0] += 1/self.sigma_t_0**2
        self.model_error_weights_array[0,:,0,0] += 1 / (1 + np.exp(-self.uv_norm + self.threshold_length))

    def exponential_weights(self):
        self.hard_cutoff_weights()
        x = self.uv_norm[self.threshold_mask] - self.threshold_length
        self.model_error_weights_array[0,:,0,0][self.threshold_mask] += np.exp(x)

    def power_law_weights(self):
        self.model_error_weights_array[0,:,0,0] = np.heaviside(self.uv_norm - self.threshold_length, 1)
        try:
            self.power = int(self.power)
        except:
            print(sys.exc_info())
            print("Maybe you passed a bad value for power for a power law cutoff?", end=" ")
            print("Defaulting to power=2")
            self.power = 2
        x = self.uv_norm[self.uv_norm < self.threshold_length] - self.threshold_length
        self.model_error_weights_array[0,:,0,0][self.threshold_mask] += (-1/x)**self.power

    def damped_sinusoid_weights(self):
        self.model_error_weights_array[0,:,0,0] = np.heaviside(self.uv_norm - self.threshold_length, 1)
        x = self.uv_norm[self.threshold_mask] - self.threshold_length
        self.model_error_weights_array[0,:,0,0][self.threshold_mask] += np.exp(x) * np.cos(x)**2

    def step_down_weights(self):
        self.hard_cutoff_weights()
        self.model_error_weights_array[0,:,0,0][self.threshold_mask] += self.scaling_factor