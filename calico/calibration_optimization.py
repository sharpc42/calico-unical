import numpy as np
import sys
import scipy
import scipy.optimize
import time
import torch
from calico import cost_function_calculations
from numpy.typing import NDArray
import jax
import jax.numpy as jnp


def cost_skycal_wrapper(
    gains_flattened: NDArray[np.floating],
    caldata_obj,
    ant_inds: NDArray[int],
    freq_ind: int,
    vis_pol_ind: int,
) -> float:
    """
    Wrapper for function cost_skycal. Reformats the input gains to be compatible
    with the scipy.optimize.minimize function.

    Parameters
    ----------
    gains_flattened : array of float
        Array of gain values. Even indices correspond to the real components of the
        gains and odd indices correspond to the imaginary components. Shape
        (2*Nants_unflagged,).
    caldata_obj : CalData
    ant_inds : array of int
        Indices of unflagged antennas to be calibrated. Shape (Nants_unflagged,).
    freq_ind : int
        Frequency channel index.
    vis_pol_ind : int
        Index of the visibility polarization.

    Returns
    -------
    cost : float
        Value of the cost function.
    """

    gains_reshaped = np.reshape(gains_flattened, (len(ant_inds), 2))
    gains_reshaped = gains_reshaped[:, 0] + 1.0j * gains_reshaped[:, 1]
    gains = np.ones((caldata_obj.Nants), dtype=complex)
    gains[ant_inds] = gains_reshaped
    if caldata_obj.gains_multiply_model:
        cost = cost_function_calculations.cost_skycal(
            gains[:, np.newaxis, np.newaxis],
            np.reshape(
                caldata_obj.data_visibilities[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
            np.reshape(
                caldata_obj.model_visibilities[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
            np.reshape(
                caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
            caldata_obj.ant1_inds,
            caldata_obj.ant2_inds,
            caldata_obj.lambda_val,
        )
    else:
        cost = cost_function_calculations.cost_skycal(
            gains[:, np.newaxis, np.newaxis],
            np.reshape(
                caldata_obj.model_visibilities[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
            np.reshape(
                caldata_obj.data_visibilities[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
            np.reshape(
                caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
            caldata_obj.ant1_inds,
            caldata_obj.ant2_inds,
            caldata_obj.lambda_val,
        )
    return cost


def jacobian_skycal_wrapper(
    gains_flattened: NDArray[np.floating],
    caldata_obj,
    ant_inds: NDArray[int],
    freq_ind: int,
    vis_pol_ind: int,
) -> NDArray[np.floating]:
    """
    Wrapper for function jacobian_skycal. Reformats the input gains and
    output Jacobian to be compatible with the scipy.optimize.minimize function.

    Parameters
    ----------
    gains_flattened : array of float
        Array of gain values. Even indices correspond to the real components of the
        gains and odd indices correspond to the imaginary components. Shape
        (2*Nants_unflagged,).
    caldata_obj : CalData
    ant_inds : array of int
        Indices of unflagged antennas to be calibrated. Shape (Nants_unflagged,).
    freq_ind : int
        Frequency channel index.
    vis_pol_ind : int
        Index of the visibility polarization.

    Returns
    -------
    jac_flattened : array of float
        Jacobian of the cost function, shape (2*Nants_unflagged,).
        Even indices correspond to the derivatives with respect to the real part
        of the gains and odd indices correspond to derivatives with respect to
        the imaginary part of the gains.
    """

    gains_reshaped = np.reshape(gains_flattened, (len(ant_inds), 2))
    gains_reshaped = gains_reshaped[:, 0] + 1.0j * gains_reshaped[:, 1]
    gains = np.ones((caldata_obj.Nants), dtype=complex)
    gains[ant_inds] = gains_reshaped
    if caldata_obj.gains_multiply_model:
        jac = cost_function_calculations.jacobian_skycal(
            gains[:, np.newaxis, np.newaxis],
            np.reshape(
                caldata_obj.data_visibilities[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
            np.reshape(
                caldata_obj.model_visibilities[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
            np.reshape(
                caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
            caldata_obj.ant1_inds,
            caldata_obj.ant2_inds,
            caldata_obj.lambda_val,
        )
    else:
        jac = cost_function_calculations.jacobian_skycal(
            gains[:, np.newaxis, np.newaxis],
            np.reshape(
                caldata_obj.model_visibilities[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
            np.reshape(
                caldata_obj.data_visibilities[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
            np.reshape(
                caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
            caldata_obj.ant1_inds,
            caldata_obj.ant2_inds,
            caldata_obj.lambda_val,
        )
    jac_flattened = np.stack(
        (np.real(jac[ant_inds, 0, 0]), np.imag(jac[ant_inds, 0, 0])), axis=1
    ).flatten()
    return jac_flattened


def hessian_skycal_wrapper(
    gains_flattened: NDArray[np.floating],
    caldata_obj,
    ant_inds: NDArray[int],
    freq_ind: int,
    vis_pol_ind: int,
) -> NDArray[np.floating]:
    """
    Wrapper for function hessian_skycal. Reformats the input gains and
    output Hessian to be compatible with the scipy.optimize.minimize function.

    Parameters
    ----------
    gains_flattened : array of float
        Array of gain values. Even indices correspond to the real components of the
        gains and odd indices correspond to the imaginary components. Shape
        (2*Nants_unflagged,).
    caldata_obj : CalData
    ant_inds : array of int
        Indices of unflagged antennas to be calibrated. Shape (Nants_unflagged,).
    freq_ind : int
        Frequency channel index.
    vis_pol_ind : int
        Index of the visibility polarization.

    Returns
    -------
    hess_flattened : array of float
        Hessian of the cost function, shape (2*Nants_unflagged, 2*Nants_unflagged,).
    """

    Nants_unflagged = len(ant_inds)
    gains_reshaped = np.reshape(gains_flattened, (Nants_unflagged, 2))
    gains_reshaped = gains_reshaped[:, 0] + 1.0j * gains_reshaped[:, 1]
    gains = np.ones((caldata_obj.Nants), dtype=complex)
    gains[ant_inds] = gains_reshaped
    if caldata_obj.gains_multiply_model:
        (
            hess_real_real,
            hess_real_imag,
            hess_imag_imag,
        ) = cost_function_calculations.hessian_skycal(
            gains[:, np.newaxis, np.newaxis],
            caldata_obj.Nants,
            caldata_obj.Nbls,
            np.reshape(
                caldata_obj.data_visibilities[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
            np.reshape(
                caldata_obj.model_visibilities[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
            np.reshape(
                caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
            caldata_obj.ant1_inds,
            caldata_obj.ant2_inds,
            caldata_obj.lambda_val,
        )
    else:
        (
            hess_real_real,
            hess_real_imag,
            hess_imag_imag,
        ) = cost_function_calculations.hessian_skycal(
            gains[:, np.newaxis, np.newaxis],
            caldata_obj.Nants,
            caldata_obj.Nbls,
            np.reshape(
                caldata_obj.model_visibilities[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
            np.reshape(
                caldata_obj.data_visibilities[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
            np.reshape(
                caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
            caldata_obj.ant1_inds,
            caldata_obj.ant2_inds,
            caldata_obj.lambda_val,
        )
    hess_flattened = np.full(
        (2 * Nants_unflagged, 2 * Nants_unflagged), np.nan, dtype=float
    )
    for unflagged_ant_ind1, ant_ind_1 in enumerate(ant_inds):
        for unflagged_ant_ind2, ant_ind_2 in enumerate(ant_inds):
            hess_flattened[2 * unflagged_ant_ind1, 2 * unflagged_ant_ind2] = (
                hess_real_real[ant_ind_1, ant_ind_2, 0, 0]
            )
            hess_flattened[2 * unflagged_ant_ind1, 2 * unflagged_ant_ind2 + 1] = (
                hess_real_imag[ant_ind_1, ant_ind_2, 0, 0]
            )
            hess_flattened[2 * unflagged_ant_ind1 + 1, 2 * unflagged_ant_ind2] = (
                np.conj(
                    hess_real_imag[ant_ind_2, ant_ind_1, 0, 0]
                )  # I believe this should be real-valued, so does the conjugation make sense?
            )
            hess_flattened[2 * unflagged_ant_ind1 + 1, 2 * unflagged_ant_ind2 + 1] = (
                hess_imag_imag[ant_ind_1, ant_ind_2, 0, 0]
            )
    return hess_flattened


def cost_ddcal_wrapper(
    gains_flattened: NDArray[np.floating],
    caldata_obj,
    ant_inds: NDArray[int],
    freq_ind: int,
    vis_pol_ind: int,
) -> float:
    """
    Direction-dependent calibration cost function. Uses the function cost_ddcal.
    Reformats the input gains to be compatible with the scipy.optimize.minimize function.

    Parameters
    ----------
    gains_flattened : array of float
        Array of gain values. Even indices correspond to the real components of the
        gains and odd indices correspond to the imaginary components. Shape
        (2*Nants_unflagged*n_directions,).
    caldata_obj : CalData
    ant_inds : array of int
        Indices of unflagged antennas to be calibrated. Shape (Nants_unflagged,).
    freq_ind : int
        Frequency channel index.
    vis_pol_ind : int
        Index of the visibility polarization.

    Returns
    -------
    cost : float
        Value of the cost function.
    """

    gains_reshaped = jnp.reshape(
        gains_flattened, (len(ant_inds), caldata_obj.n_directions, 2)
    )
    if caldata_obj.cartesian_optimization:
        gains_reshaped = gains_reshaped[:, :, 0] + 1.0j * gains_reshaped[:, :, 1]
    else:
        gains_reshaped = gains_reshaped[:, :, 0] * jnp.exp(
            2 * jnp.pi * 1j * gains_reshaped[:, :, 1]
        )

    gains = jnp.ones((caldata_obj.Nants, caldata_obj.n_directions), dtype=complex)
    gains = gains.at[jnp.ix_(ant_inds, jnp.arange(caldata_obj.n_directions))].set(
        gains_reshaped
    )

    if caldata_obj.ddcal_max_phase_offset_rad is None:
        use_ddcal_max_phase_offset_rad = None
        use_ddcal_phase_offset_taper_rad = None
        use_ddcal_ant_inds_regularized = None
    else:
        use_ddcal_max_phase_offset_rad = [
            caldata_obj.ddcal_max_phase_offset_rad[freq_ind]
        ]
        use_ddcal_phase_offset_taper_rad = [
            caldata_obj.ddcal_phase_offset_taper_rad[freq_ind]
        ]
        use_ddcal_ant_inds_regularized = [
            caldata_obj.ddcal_ant_inds_regularized[freq_ind]
        ]

    cost = cost_function_calculations.cost_ddcal(
        gains[:, jnp.newaxis, jnp.newaxis, :],
        jnp.reshape(
            caldata_obj.model_visibilities[:, :, freq_ind, vis_pol_ind, ...],
            (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1, caldata_obj.n_directions),
        ),
        jnp.reshape(
            caldata_obj.data_visibilities[:, :, freq_ind, vis_pol_ind],
            (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
        ),
        jnp.reshape(
            caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind],
            (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
        ),
        caldata_obj.ant1_inds,
        caldata_obj.ant2_inds,
        caldata_obj.lambda_val,
        use_ddcal_max_phase_offset_rad,
        use_ddcal_phase_offset_taper_rad,
        use_ddcal_ant_inds_regularized,
    )
    return cost


def cost_dwcal_wrapper(
    gains_flattened: NDArray[np.floating],
    caldata_obj,
    ant_inds: NDArray[int],
    freq_inds: NDArray[int],
    vis_pol_ind: int,
) -> float:
    """
    Wrapper for function cost_dwcal. Reformats the input gains to be compatible
    with the scipy.optimize.minimize function.

    Parameters
    ----------
    gains_flattened : array of float
        Array of gain values. Even indices correspond to the real components of the
        gains and odd indices correspond to the imaginary components. Shape
        (2*Nants_unflagged*Nfreqs_unflagged,).
    caldata_obj : CalData
    ant_inds : array of int
        Indices of unflagged antennas to be calibrated. Shape (Nants_unflagged,).
    freq_inds : int
        Indices of unflagged frequency channels to be calibrated. Shape (Nfreqs_unflagged,).
    vis_pol_ind : int
        Index of the visibility polarization.

    Returns
    -------
    cost : float
        Value of the cost function.
    """

    gains_reshaped = jnp.reshape(gains_flattened, (len(ant_inds), len(freq_inds), 2))
    gains_reshaped = gains_reshaped[:, :, [0]] + 1.0j * gains_reshaped[:, :, [1]]
    gains = jnp.ones((caldata_obj.Nants, caldata_obj.Nfreqs, 1), dtype=complex)
    gains = gains.at[jnp.ix_(ant_inds, freq_inds, jnp.array([0]))].set(gains_reshaped)

    if caldata_obj.dwcal_memory_save_mode:
        cost = cost_function_calculations.cost_dwcal_toeplitz(
            gains,
            caldata_obj.data_visibilities[:, :, :, [vis_pol_ind]],
            caldata_obj.model_visibilities[:, :, :, [vis_pol_ind]],
            caldata_obj.visibility_weights[:, :, :, [vis_pol_ind]],
            caldata_obj.dwcal_inv_covariance[:, :, :, [vis_pol_ind]],
            caldata_obj.ant1_inds,
            caldata_obj.ant2_inds,
            caldata_obj.lambda_val,
        )
    else:
        cost = cost_function_calculations.cost_dwcal(
            gains,
            caldata_obj.data_visibilities[:, :, :, [vis_pol_ind]],
            caldata_obj.model_visibilities[:, :, :, [vis_pol_ind]],
            caldata_obj.visibility_weights[:, :, :, [vis_pol_ind]],
            caldata_obj.dwcal_inv_covariance[:, :, :, :, [vis_pol_ind]],
            caldata_obj.ant1_inds,
            caldata_obj.ant2_inds,
            caldata_obj.lambda_val,
        )

    return cost


def cost_abscal_wrapper(
    abscal_parameters: NDArray[np.floating], caldata_obj, freq_ind, vis_pol_ind
) -> float:
    """
    Wrapper for function cost_function_abs_cal.

    Parameters
    ----------
    abscal_parameters : array of float
        Shape (3,).
    caldata_obj : CalData
    freq_ind : int
    vis_pol_ind : int

    Returns
    -------
    cost : float
        Value of the cost function.
    """

    if caldata_obj.gains_multiply_model:
        cost = cost_function_calculations.cost_function_abs_cal(
            abscal_parameters[0],
            abscal_parameters[1:],
            caldata_obj.data_visibilities[:, :, freq_ind, vis_pol_ind],
            caldata_obj.model_visibilities[:, :, freq_ind, vis_pol_ind],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind],
        )
    else:
        cost = cost_function_calculations.cost_function_abs_cal(
            abscal_parameters[0],
            abscal_parameters[1:],
            caldata_obj.model_visibilities[:, :, freq_ind, vis_pol_ind],
            caldata_obj.data_visibilities[:, :, freq_ind, vis_pol_ind],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind],
        )
    return cost


def jacobian_abscal_wrapper(
    abscal_parameters: NDArray[np.floating], caldata_obj, freq_ind, vis_pol_ind
) -> NDArray[np.floating]:
    """
    Wrapper for function jacobian_abs_cal.

    Parameters
    ----------
    abscal_parameters : array of float
        Shape (3,).
    caldata_obj : CalData
    freq_ind : int
    vis_pol_ind : int

    Returns
    -------
    jac : array of float
        Shape (3,).
    """

    jac = np.zeros((3,), dtype=float)
    if caldata_obj.gains_multiply_model:
        amp_jac, phase_jac = cost_function_calculations.jacobian_abs_cal(
            abscal_parameters[0],
            abscal_parameters[1:],
            caldata_obj.data_visibilities[:, :, freq_ind, vis_pol_ind],
            caldata_obj.model_visibilities[:, :, freq_ind, vis_pol_ind],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind],
        )
    else:
        amp_jac, phase_jac = cost_function_calculations.jacobian_abs_cal(
            abscal_parameters[0],
            abscal_parameters[1:],
            caldata_obj.model_visibilities[:, :, freq_ind, vis_pol_ind],
            caldata_obj.data_visibilities[:, :, freq_ind, vis_pol_ind],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind],
        )
    jac[0] = amp_jac
    jac[1:] = phase_jac
    return jac


def hessian_abscal_wrapper(
    abscal_parameters: NDArray[np.floating], caldata_obj, freq_ind, vis_pol_ind
) -> NDArray[np.floating]:
    """
    Wrapper for function hess_abs_cal.

    Parameters
    ----------
    abscal_parameters : array of float
        Shape (3,).
    caldata_obj : CalData
    freq_ind : int
    vis_pol_ind : int

    Returns
    -------
    hess : array of float
        Shape (3, 3,).
    """

    hess = np.zeros((3, 3), dtype=float)
    if caldata_obj.gains_multiply_model:
        (
            hess_amp_amp,
            hess_amp_phasex,
            hess_amp_phasey,
            hess_phasex_phasex,
            hess_phasey_phasey,
            hess_phasex_phasey,
        ) = cost_function_calculations.hess_abs_cal(
            abscal_parameters[0],
            abscal_parameters[1:],
            caldata_obj.data_visibilities[:, :, freq_ind, vis_pol_ind],
            caldata_obj.model_visibilities[:, :, freq_ind, vis_pol_ind],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind],
        )
    else:
        (
            hess_amp_amp,
            hess_amp_phasex,
            hess_amp_phasey,
            hess_phasex_phasex,
            hess_phasey_phasey,
            hess_phasex_phasey,
        ) = cost_function_calculations.hess_abs_cal(
            abscal_parameters[0],
            abscal_parameters[1:],
            caldata_obj.model_visibilities[:, :, freq_ind, vis_pol_ind],
            caldata_obj.data_visibilities[:, :, freq_ind, vis_pol_ind],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind],
        )
    hess[0, 0] = hess_amp_amp
    hess[0, 1] = hess[1, 0] = hess_amp_phasex
    hess[0, 2] = hess[2, 0] = hess_amp_phasey
    hess[1, 1] = hess_phasex_phasex
    hess[2, 2] = hess_phasey_phasey
    hess[1, 2] = hess[2, 1] = hess_phasex_phasey
    return hess


def cost_dw_abscal_wrapper(
    abscal_parameters_flattened: NDArray[np.floating],
    unflagged_freq_inds: NDArray[int],
    caldata_obj,
    vis_pol_ind: int,
) -> float:
    """
    Wrapper for function cost_function_dw_abscal.

    Parameters
    ----------
    abscal_parameters_flattened : array of float
        Abscal parameters, flattened across the frequency axis. Shape (3 * Nfreqs_unflagged,).
    unflagged_freq_inds : array of int
        Array of indices of frequency channels that are not fully flagged. Shape (Nfreqs_unflagged,).
    caldata_obj : CalData
    vis_pol_ind : int

    Returns
    -------
    cost : float
        Value of the cost function.
    """

    abscal_parameters = np.zeros((3, caldata_obj.Nfreqs))
    abscal_parameters[:, unflagged_freq_inds] = np.reshape(
        abscal_parameters_flattened, (3, len(unflagged_freq_inds))
    )
    if caldata_obj.gains_multiply_model:
        visibility_values_1 = caldata_obj.data_visibilities[:, :, :, vis_pol_ind]
        visibility_values_2 = caldata_obj.model_visibilities[:, :, :, vis_pol_ind]
    else:
        visibility_values_1 = caldata_obj.model_visibilities[:, :, :, vis_pol_ind]
        visibility_values_2 = caldata_obj.data_visibilities[:, :, :, vis_pol_ind]
    if caldata_obj.dwcal_memory_save_mode:
        cost = cost_function_calculations.cost_function_dw_abscal_toeplitz(
            abscal_parameters[0, :],
            abscal_parameters[1:, :],
            visibility_values_1,
            visibility_values_2,
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, :, vis_pol_ind],
            caldata_obj.dwcal_inv_covariance[:, :, :, vis_pol_ind],
        )
    else:
        cost = cost_function_calculations.cost_function_dw_abscal(
            abscal_parameters[0, :],
            abscal_parameters[1:, :],
            visibility_values_1,
            visibility_values_2,
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, :, vis_pol_ind],
            caldata_obj.dwcal_inv_covariance[:, :, :, :, vis_pol_ind],
        )
    return cost


def jacobian_dw_abscal_wrapper(
    abscal_parameters_flattened: NDArray[np.floating],
    unflagged_freq_inds: NDArray[int],
    caldata_obj,
    vis_pol_ind: int,
) -> NDArray[np.floating]:
    """
    Wrapper for function jacobian_dw_abscal.

    Parameters
    ----------
    abscal_parameters_flattened : array of float
        Abscal parameters, flattened across the frequency axis. Shape (3 * Nfreqs_unflagged,).
    unflagged_freq_inds : array of int
        Array of indices of frequency channels that are not fully flagged. Shape (Nfreqs_unflagged,).
    caldata_obj : CalData
    vis_pol_ind : int

    Returns
    -------
    jac_flattened : array of float
        Flattened array of derivatives of the cost function with respect to the abscal
        parameters. Shape (3 * Nfreqs,).
    """

    abscal_parameters = np.zeros((3, caldata_obj.Nfreqs))
    abscal_parameters[:, unflagged_freq_inds] = np.reshape(
        abscal_parameters_flattened, (3, len(unflagged_freq_inds))
    )
    if caldata_obj.gains_multiply_model:
        visibility_values_1 = caldata_obj.data_visibilities[:, :, :, vis_pol_ind]
        visibility_values_2 = caldata_obj.model_visibilities[:, :, :, vis_pol_ind]
    else:
        visibility_values_1 = caldata_obj.model_visibilities[:, :, :, vis_pol_ind]
        visibility_values_2 = caldata_obj.data_visibilities[:, :, :, vis_pol_ind]
    if caldata_obj.dwcal_memory_save_mode:
        amp_jac, phase_jac = cost_function_calculations.jacobian_dw_abscal_toeplitz(
            abscal_parameters[0, :],
            abscal_parameters[1:, :],
            visibility_values_1,
            visibility_values_2,
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, :, vis_pol_ind],
            caldata_obj.dwcal_inv_covariance[:, :, :, vis_pol_ind],
        )
    else:
        amp_jac, phase_jac = cost_function_calculations.jacobian_dw_abscal(
            abscal_parameters[0, :],
            abscal_parameters[1:, :],
            visibility_values_1,
            visibility_values_2,
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, :, vis_pol_ind],
            caldata_obj.dwcal_inv_covariance[:, :, :, :, vis_pol_ind],
        )
    jac_array = np.zeros((3, caldata_obj.Nfreqs), dtype=float)
    jac_array[0, :] = amp_jac
    jac_array[1:, :] = phase_jac
    jac_array = np.take(jac_array, unflagged_freq_inds, axis=1)
    return jac_array.flatten()


def hessian_dw_abscal_wrapper(
    abscal_parameters_flattened: NDArray[np.floating],
    unflagged_freq_inds: NDArray[int],
    caldata_obj,
    vis_pol_ind: int,
) -> NDArray[np.floating]:
    """
    Wrapper for function hess_dw_abscal.

    Parameters
    ----------
    abscal_parameters_flattened : array of float
        Abscal parameters, flattened across the frequency axis. Shape (3 * Nfreqs_unflagged,).
    unflagged_freq_inds : array of int
        Array of indices of frequency channels that are not fully flagged. Shape (Nfreqs_unflagged,).
    caldata_obj : CalData
    vis_pol_ind : int

    Returns
    -------
    hess : array of float
        Array of second derivatives of the cost function with respect to the abscal
        parameters. Shape (3 * Nfreqs, 3 * Nfreqs,).
    """

    abscal_parameters = np.zeros((3, caldata_obj.Nfreqs))
    abscal_parameters[:, unflagged_freq_inds] = np.reshape(
        abscal_parameters_flattened, (3, len(unflagged_freq_inds))
    )
    if caldata_obj.gains_multiply_model:
        visibility_values_1 = caldata_obj.data_visibilities[:, :, :, vis_pol_ind]
        visibility_values_2 = caldata_obj.model_visibilities[:, :, :, vis_pol_ind]
    else:
        visibility_values_1 = caldata_obj.model_visibilities[:, :, :, vis_pol_ind]
        visibility_values_2 = caldata_obj.data_visibilities[:, :, :, vis_pol_ind]
    if caldata_obj.dwcal_memory_save_mode:
        (
            hess_amp_amp,
            hess_amp_phasex,
            hess_amp_phasey,
            hess_phasex_phasex,
            hess_phasey_phasey,
            hess_phasex_phasey,
        ) = cost_function_calculations.hess_dw_abscal_toeplitz(
            abscal_parameters[0, :],
            abscal_parameters[1:, :],
            visibility_values_1,
            visibility_values_2,
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, :, vis_pol_ind],
            caldata_obj.dwcal_inv_covariance[:, :, :, vis_pol_ind],
        )
    else:
        (
            hess_amp_amp,
            hess_amp_phasex,
            hess_amp_phasey,
            hess_phasex_phasex,
            hess_phasey_phasey,
            hess_phasex_phasey,
        ) = cost_function_calculations.hess_dw_abscal(
            abscal_parameters[0, :],
            abscal_parameters[1:, :],
            visibility_values_1,
            visibility_values_2,
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, :, vis_pol_ind],
            caldata_obj.dwcal_inv_covariance[:, :, :, :, vis_pol_ind],
        )
    hess = np.zeros((3, caldata_obj.Nfreqs, 3, caldata_obj.Nfreqs), dtype=float)
    hess[0, :, 0, :] = hess_amp_amp
    hess[0, :, 1, :] = hess_amp_phasex.T
    hess[1, :, 0, :] = hess_amp_phasex
    hess[0, :, 2, :] = hess_amp_phasey.T
    hess[2, :, 0, :] = hess_amp_phasey
    hess[1, :, 1, :] = hess_phasex_phasex
    hess[2, :, 2, :] = hess_phasey_phasey
    hess[1, :, 2, :] = hess_phasex_phasey.T
    hess[2, :, 1, :] = hess_phasex_phasey
    hess = np.take(
        np.take(hess, unflagged_freq_inds, axis=1), unflagged_freq_inds, axis=3
    )
    hess = np.reshape(
        hess, (3 * len(unflagged_freq_inds), 3 * len(unflagged_freq_inds))
    )
    return hess


def _unical_ant_inds(caldata_obj):
    """
    Return the (ant1_inds, ant2_inds) maps that match the current unical
    visibility axis.
    """
    return caldata_obj.ant1_inds, caldata_obj.ant2_inds


def _unpack_unical_fit_vis(u_part_flat, caldata_obj):
    """
    Reshape the optimizer's fitted u-parameter block back to (Ntimes, Nbls)
    complex for storage, inverting whichever layout was used during the solve.
    The flattened (Ntimes*Nbls) layout and the (Ntimes, Nbls) layout share the
    same underlying float ordering, so this is a pure reshape either way.
    """
    u = np.reshape(u_part_flat, (caldata_obj.Ntimes * caldata_obj.Nbls, 2))
    return (u[:, 0] + 1.0j * u[:, 1]).reshape(caldata_obj.Ntimes, caldata_obj.Nbls)


def cost_unical_wrapper(
    params_flattened,
    caldata_obj,
    ant_inds,
    n_ants_unflagged,
    n_times,
    bl_inds,
    freq_ind,
    vis_pol_ind,
    dev_type="",
):
    """
    Wrapper for function cost_unical. Reformats the input gains to be compatible
    with the scipy.optimize.minimize function.

    Parameters
    ----------
    params_flattened : array of float
        Shape (2*Nants_unflagged + 2*Ntimes*Nbls,). Array of values for gain params
        and u params. Even indices correspond to the real components of the and odd
        indices correspond to the imaginary components.
    caldata_obj : CalData
    ant_inds : array of int
        Shape (Nants_unflagged,). Indices of unflagged antennas to be calibrated.
    Nants_unflagged : int
        Number of unflagged antennas to be calibrated.
    bl_inds : array of int
        Shape (Nbls,). Shifted indices of baselines to be calibrated.
    freq_ind : int
        Frequency channel index.
    vis_pol_ind : int
        Index of the visibility polarization.

    Returns
    -------
    cost : float
        Value of the cost function.
    """

    # reshape gain params
    gains_reshaped = np.reshape(
        params_flattened[: 2 * len(ant_inds)], (len(ant_inds), 2)
    )
    gains_reshaped = gains_reshaped[:, 0] + 1.0j * gains_reshaped[:, 1]
    gains = np.ones((caldata_obj.Nants), dtype=complex)
    gains[ant_inds] = gains_reshaped
    if dev_type == "test gains rolled":
        return gains_reshaped
    ant1_inds_use, ant2_inds_use = _unical_ant_inds(caldata_obj)
    fit_vis_reshaped = np.reshape(
        params_flattened[2 * n_ants_unflagged :], (n_times, len(bl_inds), 2)
    )
    fit_vis_reshaped = fit_vis_reshaped[:, :, 0] + 1.0j * fit_vis_reshaped[:, :, 1]
    if dev_type == "test fit vis rolled":
        return fit_vis_reshaped
    if dev_type == "test gains one run skycal":
        return cost_function_calculations.cost_unical(
            gains,
            fit_vis_reshaped,
            caldata_obj.data_vis_reshaped,
            caldata_obj.model_vis_reshaped,
            caldata_obj.vis_weights_reshaped,
            caldata_obj.model_weights_reshaped,
            ant1_inds_use,
            ant2_inds_use,
            caldata_obj.lambda_val,
            force_skycal=True,
        )

    if caldata_obj.gains_multiply_model:
        cost = cost_function_calculations.cost_unical(
            gains,
            fit_vis_reshaped,
            caldata_obj.data_vis_reshaped,
            caldata_obj.model_vis_reshaped,
            caldata_obj.vis_weights_reshaped,
            caldata_obj.model_weights_reshaped,
            ant1_inds_use,
            ant2_inds_use,
            caldata_obj.lambda_val,
        )
    else:
        cost = cost_function_calculations.cost_unical(
            gains,
            caldata_obj.data_vis_reshaped,
            fit_vis_reshaped,
            caldata_obj.model_vis_reshaped,
            caldata_obj.vis_weights_reshaped,
            caldata_obj.model_weights_reshaped,
            ant1_inds_use,
            ant2_inds_use,
            caldata_obj.lambda_val,
            caldata_obj.gains_multiply_model,
        )

    return cost


def _unpack_unical_params(
    params_flattened, caldata_obj, ant_inds, n_ants_unflagged, n_times, bl_inds
):
    """
    Turn the optimizer's flattened real parameters back into complex gains
    (Nants,) and fitted visibilities (Ntimes, Nbls) for the analytic Jacobian
    and Hessian, which only support gains multiplying the fitted visibilities.
    """
    if not caldata_obj.gains_multiply_model:
        raise NotImplementedError(
            "The analytic unical Jacobian and Hessian (used by the BFGS and "
            "Newton-CG schemes) assume gains_multiply_model=True."
        )
    gains_reshaped = np.reshape(
        params_flattened[: 2 * len(ant_inds)], (len(ant_inds), 2)
    )
    gains = np.ones((caldata_obj.Nants), dtype=complex)
    gains[ant_inds] = gains_reshaped[:, 0] + 1.0j * gains_reshaped[:, 1]
    fit_vis_reshaped = np.reshape(
        params_flattened[2 * n_ants_unflagged :], (n_times, len(bl_inds), 2)
    )
    fit_vis_reshaped = fit_vis_reshaped[:, :, 0] + 1.0j * fit_vis_reshaped[:, :, 1]
    return gains, fit_vis_reshaped


def jacobian_unical_wrapper(
    params_flattened,
    caldata_obj,
    ant_inds,
    n_ants_unflagged,
    n_times,
    bl_inds,
    freq_ind,
    vis_pol_ind,
):
    """
    Wrapper for function jacobian_unical. Reformats the input gains and
    output Jacobian to be compatible with the scipy.optimize.minimize function.

    Parameters
    ----------
    params_flattened : array of float
        Shape (2*Nants_unflagged + 2*Ntimes*Nbls,). Array of values for gain params
        and u params. Even indices correspond to the real components of the and odd
        indices correspond to the imaginary components.
    caldata_obj : CalData
    ant_inds : array of int
        Shape (Nants_unflagged,). Indices of unflagged antennas to be calibrated.
    Nants_unflagged : int
        Number of unflagged antennas to be calibrated.
    bl_inds : array of int
        Shape (Nbls,). Shifted indices of baselines to be calibrated.
    freq_ind : int
        Frequency channel index.
    vis_pol_ind : int
        Index of the visibility polarization.

    Returns
    -------
    jac_flattened : array of float
        Jacobian of the cost function, shape
        (2*Nants_unflagged + 2*Ntimes*Nbls,), in the same order as
        params_flattened.
    """

    gains, fit_vis_reshaped = _unpack_unical_params(
        params_flattened, caldata_obj, ant_inds, n_ants_unflagged, n_times, bl_inds
    )
    ant1_inds_use, ant2_inds_use = _unical_ant_inds(caldata_obj)
    return cost_function_calculations.jacobian_unical(
        gains,
        fit_vis_reshaped,
        caldata_obj.data_vis_reshaped,
        caldata_obj.model_vis_reshaped,
        caldata_obj.vis_weights_reshaped,
        caldata_obj.model_weights_reshaped,
        ant1_inds_use,
        ant2_inds_use,
        ant_inds,
        caldata_obj.lambda_val,
    )


def hessian_unical_wrapper(
    params_flattened,
    caldata_obj,
    ant_inds,
    n_ants_unflagged,
    n_times,
    bl_inds,
    freq_ind,
    vis_pol_ind,
):
    """
    Wrapper for function hessian_unical. Reformats the input gains and
    output Hessian to be compatible with the scipy.optimize.minimize function.

    Parameters
    ----------
    params_flattened : array of float
        Shape (2*Nants_unflagged + 2*Ntimes*Nbls,). Array of values for gain params
        and u params. Even indices correspond to the real components of the and odd
        indices correspond to the imaginary components.
    caldata_obj : CalData
    ant_inds : array of int
        Shape (Nants_unflagged,). Indices of unflagged antennas to be calibrated.
    Nants_unflagged : int
        Number of unflagged antennas to be calibrated.
    bl_inds : array of int
        Shape (Nbls,). Shifted indices of baselines to be calibrated.
    freq_ind : int
        Frequency channel index.
    vis_pol_ind : int
        Index of the visibility polarization.

    Returns
    -------
    hess_flattened : array of float
        Dense Hessian of the cost function, shape (N, N,) with
        N = 2*Nants_unflagged + 2*Ntimes*Nbls, in the same order as params_flattened.
    """

    gains, fit_vis_reshaped = _unpack_unical_params(
        params_flattened, caldata_obj, ant_inds, n_ants_unflagged, n_times, bl_inds
    )
    ant1_inds_use, ant2_inds_use = _unical_ant_inds(caldata_obj)
    return cost_function_calculations.hessian_unical(
        gains,
        fit_vis_reshaped,
        caldata_obj.data_vis_reshaped,
        caldata_obj.model_vis_reshaped,
        caldata_obj.vis_weights_reshaped,
        caldata_obj.model_weights_reshaped,
        ant1_inds_use,
        ant2_inds_use,
        ant_inds,
        caldata_obj.lambda_val,
    )


def run_skycal_optimization_per_pol_single_freq(
    caldata_obj,
    freq_ind: int = 0,
    verbose: bool = True,
    get_crosspol_phase: bool = True,
    crosspol_phase_strategy: str = "crosspol model",
) -> NDArray[np.complexfloating]:
    """
    Run calibration per polarization. Here the XX and YY visibilities are
    calibrated individually. If get_crosspol_phase is set, the cross-
    polarization phase is applied from the XY and YX visibilities after the
    fact.

    Parameters
    ----------
    caldata_obj : CalData
    freq_ind : int
        Frequency channel to process. Default 0.

    Returns
    -------
    gains_fit : array of complex
        Fit gain values. Shape (Nants, 1, N_feed_pols,).
    """

    gains_fit = np.full(
        (caldata_obj.Nants, caldata_obj.N_feed_pols),
        np.nan + 1j * np.nan,
        dtype=complex,
    )
    if np.max(caldata_obj.visibility_weights[:, :, freq_ind, :]) == 0.0:
        if caldata_obj.verbose:
            print("WARNING: All data flagged.")
            sys.stdout.flush()
        gains_fit[:, :] = np.nan + 1j * np.nan
        return gains_fit

    for feed_pol_ind, feed_pol in enumerate(caldata_obj.feed_polarization_array):
        vis_pol_ind = np.where(caldata_obj.vis_polarization_array == feed_pol)[0]

        if (
            np.max(caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind]) == 0.0
        ):  # All flagged
            gains_fit[:, feed_pol_ind] = np.nan + 1j * np.nan
        else:
            vis_weights_summed = np.sum(
                caldata_obj.visibility_weights[:, :, freq_ind, feed_pol_ind], axis=0
            )  # Sum over times
            weight_per_ant = np.bincount(
                caldata_obj.ant1_inds,
                weights=vis_weights_summed,
                minlength=caldata_obj.Nants,
            ) + np.bincount(
                caldata_obj.ant2_inds,
                weights=vis_weights_summed,
                minlength=caldata_obj.Nants,
            )
            ant_inds = np.where(weight_per_ant > 0.0)[0]

            gains_init_flattened = np.stack(
                (
                    np.real(caldata_obj.gains[ant_inds, freq_ind, feed_pol_ind]),
                    np.imag(caldata_obj.gains[ant_inds, freq_ind, feed_pol_ind]),
                ),
                axis=1,
            ).flatten()

            # Minimize the cost function
            start_optimize = time.time()
            result = scipy.optimize.minimize(
                cost_skycal_wrapper,
                gains_init_flattened,
                args=(caldata_obj, ant_inds, freq_ind, vis_pol_ind),
                method="Newton-CG",
                jac=jacobian_skycal_wrapper,
                hess=hessian_skycal_wrapper,
                options={
                    "disp": caldata_obj.verbose,
                    "xtol": caldata_obj.xtol,
                    "maxiter": caldata_obj.maxiter,
                },
            )
            end_optimize = time.time()
            if caldata_obj.verbose and not caldata_obj.parallel:
                print(result.message)
                print(
                    f"Freq. {freq_ind} Pol. {feed_pol_ind}, optimization time: {(end_optimize - start_optimize)/60.} minutes"
                )
                sys.stdout.flush()
            gains_fit_single_pol = np.reshape(result.x, (len(ant_inds), 2))
            gains_fit[ant_inds, feed_pol_ind] = (
                gains_fit_single_pol[:, 0] + 1j * gains_fit_single_pol[:, 1]
            )

            # Ensure that the phase of the gains is mean-zero
            # If lambda_val != 0, this should be handled by the phase regularization term, but
            # this step removes any optimizer precision effects.
            avg_angle = np.arctan2(
                np.nanmean(np.sin(np.angle(gains_fit[:, feed_pol_ind]))),
                np.nanmean(np.cos(np.angle(gains_fit[:, feed_pol_ind]))),
            )
            gains_fit[:, feed_pol_ind] *= np.cos(avg_angle) - 1j * np.sin(avg_angle)

    # Constrain crosspol phase
    if (
        caldata_obj.get_crosspol_phase
        and caldata_obj.N_feed_pols == 2
        and caldata_obj.N_vis_pols == 4
    ):
        if (
            caldata_obj.feed_polarization_array[0] == -5
            and caldata_obj.feed_polarization_array[1] == -6
        ):
            crosspol_polarizations = [-7, -8]
        elif (
            caldata_obj.feed_polarization_array[0] == -6
            and caldata_obj.feed_polarization_array[1] == -5
        ):
            crosspol_polarizations = [-8, -7]
        crosspol_indices = np.array(
            [
                np.where(caldata_obj.vis_polarization_array == pol)[0][0]
                for pol in crosspol_polarizations
            ]
        )
        if caldata_obj.crosspol_phase_strategy.lower() == "pseudo stokes v":
            crosspol_phase = cost_function_calculations.set_crosspol_phase_pseudoV(
                gains_fit,
                caldata_obj.data_visibilities[:, :, freq_ind, crosspol_indices],
                caldata_obj.visibility_weights[:, :, freq_ind, crosspol_indices],
                caldata_obj.ant1_inds,
                caldata_obj.ant2_inds,
            )
        elif caldata_obj.crosspol_phase_strategy.lower() == "crosspol model":
            crosspol_phase = cost_function_calculations.set_crosspol_phase(
                gains_fit,
                caldata_obj.model_visibilities[:, :, freq_ind, crosspol_indices],
                caldata_obj.data_visibilities[:, :, freq_ind, crosspol_indices],
                caldata_obj.visibility_weights[:, :, freq_ind, crosspol_indices],
                caldata_obj.ant1_inds,
                caldata_obj.ant2_inds,
            )
        else:
            print(
                "WARNING: Unknown crosspol_phase_strategy. Skipping fitting crosspol phase."
            )
            crosspol_phase = 0.0

        if caldata_obj.gains_multiply_model:
            gains_fit[:, 0] /= np.exp(-1j * crosspol_phase / 2)
            gains_fit[:, 1] /= np.exp(1j * crosspol_phase / 2)
        else:
            gains_fit[:, 0] *= np.exp(-1j * crosspol_phase / 2)
            gains_fit[:, 1] *= np.exp(1j * crosspol_phase / 2)

    return gains_fit


def run_skycal_optimization_per_pol_single_freq_parallel(args):
    """
    Wrapper for run_skycal_optimization_per_pol_single_freq that makes the function compatible with
    multiprocessing by unpacking a tuple or arguments.
    """
    (caldata_subset, freq_ind) = args
    start_optimize = time.time()
    gains_fit = run_skycal_optimization_per_pol_single_freq(
        caldata_subset,
        freq_ind=0,
    )
    end_optimize = time.time()
    if caldata_subset.verbose:
        print(
            f"Freq. {freq_ind}, optimization time: {(end_optimize - start_optimize)/60.} minutes"
        )
        sys.stdout.flush()
    return freq_ind, gains_fit


def run_ddcal_optimization(
    caldata_obj,
    freq_ind: int = 0,
    pol_ind: int = 0,
) -> NDArray[np.complexfloating]:
    """
    Run direction-dependent calibration per frequency and polarization.

    Parameters
    ----------
    caldata_obj : CalData
    freq_ind : int
        Frequency channel to process. Default 0.
    pol_ind : int
        Feed polarization index to process. Default 0.

    Returns
    -------
    gains_fit : array of complex
        Fit gain values. Shape (Nants, n_directions,).
    """

    gains_fit = np.full(
        (caldata_obj.Nants, caldata_obj.n_directions),
        np.nan + 1j * np.nan,
        dtype=complex,
    )

    vis_pol_ind = np.where(
        caldata_obj.vis_polarization_array
        == caldata_obj.feed_polarization_array[pol_ind]
    )[0]

    if (
        np.max(caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind]) == 0.0
    ):  # All flagged
        if caldata_obj.verbose:
            print("WARNING: All data flagged.")
            sys.stdout.flush()
        gains_fit[...] = np.nan + 1j * np.nan
        return gains_fit

    vis_weights_summed = np.sum(
        caldata_obj.visibility_weights[:, :, freq_ind, pol_ind], axis=0
    )  # Sum over times
    weight_per_ant = np.bincount(
        caldata_obj.ant1_inds,
        weights=vis_weights_summed,
        minlength=caldata_obj.Nants,
    ) + np.bincount(
        caldata_obj.ant2_inds,
        weights=vis_weights_summed,
        minlength=caldata_obj.Nants,
    )
    ant_inds = np.where(weight_per_ant > 0.0)[0]

    if caldata_obj.n_directions == 1:
        gains_init = caldata_obj.gains[ant_inds, freq_ind, pol_ind, np.newaxis]
    else:
        gains_init = caldata_obj.gains[ant_inds, freq_ind, pol_ind, :]

    if caldata_obj.cartesian_optimization:
        gains_init_flattened = np.stack(
            (
                np.real(gains_init),
                np.imag(gains_init),
            ),
            axis=2,
        ).flatten()
    else:
        gains_init_flattened = np.stack(
            (np.abs(gains_init), np.angle(gains_init)), axis=2
        ).flatten()

    # Minimize the cost function
    start_optimize = time.time()
    result = scipy.optimize.minimize(
        cost_ddcal_wrapper,
        gains_init_flattened,
        args=(caldata_obj, ant_inds, freq_ind, vis_pol_ind),
        method="Newton-CG",
        jac=jax.jacrev(cost_ddcal_wrapper),
        hess=jax.jacrev(jax.jacrev(cost_ddcal_wrapper)),
        options={
            "disp": caldata_obj.verbose,
            "xtol": caldata_obj.xtol,
            "maxiter": caldata_obj.maxiter,
        },
    )
    end_optimize = time.time()
    if caldata_obj.verbose and not caldata_obj.parallel:
        print(result.message)
        print(
            f"Freq. {freq_ind} Pol. {pol_ind}, optimization time: {(end_optimize - start_optimize)/60.} minutes"
        )
        sys.stdout.flush()
    gains_fit_single_pol = np.reshape(
        result.x, (len(ant_inds), caldata_obj.n_directions, 2)
    )

    if caldata_obj.cartesian_optimization:
        gains_fit[ant_inds, :] = (
            gains_fit_single_pol[:, :, 0] + 1j * gains_fit_single_pol[:, :, 1]
        )
    else:
        gains_fit[ant_inds, :] = gains_fit_single_pol[:, :, 0] * np.exp(
            2 * np.pi * 1j * gains_fit_single_pol[:, :, 1]
        )

    # Ensure that the phase of the gains is mean-zero
    # If lambda_val != 0, this should be handled by the phase regularization term, but
    # this step removes any optimizer precision effects.
    for direction_ind in range(caldata_obj.n_directions):
        avg_angle = np.arctan2(
            np.nanmean(np.sin(np.angle(gains_fit[:, direction_ind]))),
            np.nanmean(np.cos(np.angle(gains_fit[:, direction_ind]))),
        )
        gains_fit[:, direction_ind] *= np.cos(avg_angle) - 1j * np.sin(avg_angle)

    return gains_fit


def run_ddcal_optimization_parallel(args):
    """
    Wrapper for run_ddcal_optimization that makes the function compatible with
    multiprocessing by unpacking a tuple or arguments.
    """
    (caldata_subset, freq_ind, pol_ind) = args
    start_optimize = time.time()
    gains_fit = run_ddcal_optimization(
        caldata_subset,
        freq_ind=0,
        pol_ind=0,
    )
    end_optimize = time.time()
    if caldata_subset.verbose:
        print(
            f"Freq. {freq_ind} Pol. {pol_ind}, optimization time: {(end_optimize - start_optimize)/60.} minutes"
        )
        sys.stdout.flush()
    return freq_ind, pol_ind, gains_fit


def run_dwcal_optimization_per_pol(
    caldata_obj,
    pol_ind: int = 0,
) -> NDArray[np.complexfloating]:
    """
    Run delay-weighted calibration for a single polarization. Uses automatic
    differentiation with jax to compute first and second derivatives for
    optimization with Newton's Method.

    Parameters
    ----------
    caldata_obj : CalData
    pol_ind : int
        Feed polarization index to process. Default 0.

    Returns
    -------
    gains_fit : array of complex
        Fit gain values. Shape (Nants, Nfreqs, 1,).
    """

    gains_fit = np.full(
        (caldata_obj.Nants, caldata_obj.Nfreqs),
        np.nan + 1j * np.nan,
        dtype=complex,
    )

    vis_pol_ind = np.where(
        caldata_obj.vis_polarization_array
        == caldata_obj.feed_polarization_array[pol_ind]
    )[0][0]
    unflagged_freq_inds = np.where(
        np.sum(caldata_obj.visibility_weights[:, :, :, vis_pol_ind], axis=(0, 1)) > 0
    )[0]
    if len(unflagged_freq_inds) == 0:
        print(f"ERROR: Data all flagged.")
        sys.stdout.flush()
        return gains_fit

    vis_weights_summed = np.sum(
        caldata_obj.visibility_weights[:, :, :, vis_pol_ind], axis=(0, 2)
    )  # Sum over times and frequencies
    weight_per_ant = np.bincount(
        caldata_obj.ant1_inds,
        weights=vis_weights_summed,
        minlength=caldata_obj.Nants,
    ) + np.bincount(
        caldata_obj.ant2_inds,
        weights=vis_weights_summed,
        minlength=caldata_obj.Nants,
    )
    ant_inds = np.where(weight_per_ant > 0.0)[0]

    gains_init_flattened = np.stack(
        (
            np.real(
                caldata_obj.gains[:, :, pol_ind][np.ix_(ant_inds, unflagged_freq_inds)]
            ),
            np.imag(
                caldata_obj.gains[:, :, pol_ind][np.ix_(ant_inds, unflagged_freq_inds)]
            ),
        ),
        axis=1,
    ).flatten()

    # Minimize the cost function
    start_optimize = time.time()
    result = scipy.optimize.minimize(
        cost_dwcal_wrapper,
        gains_init_flattened,
        args=(caldata_obj, ant_inds, unflagged_freq_inds, vis_pol_ind),
        method="Newton-CG",
        jac=jax.jacrev(cost_dwcal_wrapper),
        hess=jax.jacrev(jax.jacrev(cost_dwcal_wrapper)),
        options={
            "disp": caldata_obj.verbose,
            "xtol": caldata_obj.xtol,
            "maxiter": caldata_obj.maxiter,
        },
    )
    end_optimize = time.time()
    if caldata_obj.verbose:
        print(result.message)
        print(
            f"Pol. {pol_ind}, optimization time: {(end_optimize - start_optimize)/60.} minutes"
        )
        sys.stdout.flush()
    gains_fit_unflagged = np.reshape(
        result.x, (len(ant_inds), len(unflagged_freq_inds), 2)
    )
    gains_fit[np.ix_(ant_inds, unflagged_freq_inds)] = (
        gains_fit_unflagged[:, :, 0] + 1j * gains_fit_unflagged[:, :, 1]
    )
    return gains_fit


def run_abscal_optimization_single_freq(
    caldata_obj,
    freq_ind: int = 0,
    feed_pol_ind: int = 0,
) -> NDArray[np.complexfloating]:
    """
    Run absolute calibration ("abscal").

    Parameters
    ----------
    caldata_obj : CalData
    freq_ind : int
        Frequency channel to process. Default 0.
    feed_pol_ind : int
        Feed polarization index to process. Default 0.

    Returns
    -------
    abscal_params : array of complex
        Fit abscal parameter values. Shape (3, 1, N_feed_pols,).
    """

    start_optimize = time.time()
    vis_pol_ind = np.where(
        caldata_obj.vis_polarization_array
        == caldata_obj.feed_polarization_array[feed_pol_ind]
    )[0][0]
    result = scipy.optimize.minimize(
        cost_abscal_wrapper,
        caldata_obj.abscal_params[:, freq_ind, feed_pol_ind],
        args=(caldata_obj, freq_ind, vis_pol_ind),
        method="Newton-CG",
        jac=jacobian_abscal_wrapper,
        hess=hessian_abscal_wrapper,
        options={
            "disp": caldata_obj.verbose,
            "xtol": caldata_obj.xtol,
            "maxiter": caldata_obj.maxiter,
        },
    )
    abscal_params = result.x
    end_optimize = time.time()
    if caldata_obj.verbose:
        print(result.message)
        print(f"Optimization time: {(end_optimize - start_optimize)/60.} minutes")
    sys.stdout.flush()

    return abscal_params


def run_dw_abscal_optimization(
    caldata_obj,
    feed_pol_ind: int = 0,
) -> NDArray[np.complexfloating]:
    """
    Run absolute calibration with delay weighting.

    Parameters
    ----------
    caldata_obj : CalData
    feed_pol_ind : int
        Feed polarization index to process. Default 0.

    Returns
    -------
    abscal_params : array of complex
        Fit abscal parameter values. Shape (3, Nfreqs,).
    """

    vis_pol_ind = np.where(
        caldata_obj.vis_polarization_array
        == caldata_obj.feed_polarization_array[feed_pol_ind]
    )[0][0]
    abscal_params = np.zeros_like(caldata_obj.abscal_params[:, :, feed_pol_ind])

    unflagged_freq_inds = np.where(
        np.sum(caldata_obj.visibility_weights, axis=(0, 1, 3)) > 0
    )[0]
    if len(unflagged_freq_inds) == 0:
        print(f"ERROR: Data all flagged.")
        sys.stdout.flush()
        return
    abscal_params_flattened = caldata_obj.abscal_params[
        :, unflagged_freq_inds, feed_pol_ind
    ].flatten()
    # Minimize the cost function
    start_optimize = time.time()
    result = scipy.optimize.minimize(
        cost_dw_abscal_wrapper,
        abscal_params_flattened,
        args=(unflagged_freq_inds, caldata_obj, vis_pol_ind),
        method="Newton-CG",
        jac=jacobian_dw_abscal_wrapper,
        hess=hessian_dw_abscal_wrapper,
        options={
            "disp": caldata_obj.verbose,
            "xtol": caldata_obj.xtol,
            "maxiter": caldata_obj.maxiter,
        },
    )
    abscal_params[:, unflagged_freq_inds] = np.reshape(
        result.x, (3, len(unflagged_freq_inds))
    )
    if caldata_objverbose:
        print(result.message)
        print(f"Optimization time: {(time.time() - start_optimize)/60.} minutes")
    sys.stdout.flush()

    return abscal_params


def _confirm_dense_hessian(n_params, n_times, allow_large_hessian):
    """
    Newton-CG builds a dense Hessian over every gain and fitted-visibility
    parameter, so its memory grows as n_params**2. With more than one time
    there is one fitted visibility per time and baseline, which can make it
    too large to allocate. Ask before continuing when running interactively;
    otherwise require allow_large_hessian=True.
    """
    if n_times <= 1 or allow_large_hessian:
        return
    size_gb = n_params**2 * 8 / 1e9
    message = (
        f"Newton-CG will build a dense {n_params} x {n_params} Hessian "
        f"(about {size_gb:.3g} GB) because Ntimes={n_times} > 1 gives one "
        "fitted visibility per time and baseline; if that doesn't fit in "
        "memory, Python will raise a MemoryError or be killed."
    )
    if sys.stdin is not None and sys.stdin.isatty():
        reply = input(f"{message}\nContinue? [y/N] ")
        if reply.strip().lower() in ("y", "yes"):
            return
        raise RuntimeError("Newton-CG optimization cancelled.")
    raise RuntimeError(
        f"{message} Pass allow_large_hessian=True to run anyway, or use the "
        "'pytorch' or 'bfgs' optimization scheme."
    )


def run_unical_optimization(
    caldata_obj,
    xtol,
    maxiter,
    freq_ind=0,
    verbose=True,
    get_crosspol_phase=True,
    crosspol_phase_strategy="crosspol model",
    optimization_scheme="powell",
    calibrate=True,
    dev_type="",
    torch_maxiter=1000,
    allow_large_hessian=False,
):
    """
    Run calibration per polarization. Here the XX and YY visibilities are
    calibrated individually. If get_crosspol_phase is set, the cross-
    polarization phase is applied from the XY and YX visibilities after the
    fact.

    Parameters
    ----------
    caldata_obj : CalData
    xtol : float
        Accuracy tolerance for optimizer. For the PyTorch L-BFGS schemes this
        is the relative change in the cost, measured over at least 100
        iterations, below which the fit is converged. For BFGS it is passed as
        gtol, the gradient-norm tolerance.
    maxiter : int
        Maximum number of iterations for the SciPy optimizers.
    torch_maxiter : int
        Maximum total number of L-BFGS iterations for the PyTorch schemes
        ("pytorch", "lbfgs", "pytorch lbfgs"). Default 1000.
    allow_large_hessian : bool
        Newton-CG builds a dense Hessian whose size grows with
        (Nants + Ntimes*Nbls)**2. When Ntimes > 1 it asks before continuing
        (in an interactive terminal) or raises an error; set True to skip
        that check. Default False.
    freq_ind : int
        Frequency channel to process. Default 0.
    verbose : bool
        Set to True to print optimization outputs. Default True.
    get_crosspol_phase : bool
        Set to True to constrain the cross-polarizaton phase from the XY and YX
        visibilities. Default True.
    crosspol_phase_strategy : str
        Options are "crosspol model" or "pseudo Stokes V". Used only if
        get_crosspol_phase is True. If "crosspol model", contrains the crosspol
        phase using the crosspol model visibilities. If "pseudo Stokes V", constrains
        crosspol phase by minimizing pseudo Stokes V. Default "crosspol model".

    Returns
    -------
    gains_fit : array of complex
        Fit gain values. Shape (Nants, 1, N_feed_pols,).
    fit_vis_fit : array of complex
        Fit model parameter values. Shape (Ntimes, Nbls, N_vis_pols)
    """

    optimization_methods = {
        "scipy",
        "pytorch",
        "powell",
        "scipy powell",
        "bfgs",
        "scipy bfgs",
        "newton-cg",
        "scipy newton-cglbfgs",
        "pytorch lbfgs",
        "adam",
        "pytorch adam",
    }

    if optimization_scheme not in optimization_methods:
        print("Optimization scheme not known, defaulting to SciPy Powell")
        print(f"Available options:\n{optimization_methods=}")
        optimization_scheme = "powell"
    if optimization_scheme in ("adam", "pytorch adam"):
        raise NotImplementedError(
            "Adam optimization is not implemented yet; "
            "use 'pytorch' (L-BFGS) or a SciPy scheme."
        )

    gains_fit = np.full(
        (caldata_obj.Nants, caldata_obj.N_feed_pols),
        np.nan + 1j * np.nan,
        dtype=complex,
    )
    fit_vis_fit = np.full(
        # (caldata_obj.Ntimes, caldata_obj.Nbls, caldata_obj.N_vis_pols),
        (
            np.size(caldata_obj.data_visibilities, axis=0),
            np.size(caldata_obj.data_visibilities, axis=1),
            caldata_obj.N_vis_pols,
        ),
        # (1, caldata_obj.Ntimes * caldata_obj.Nbls, caldata_obj.N_vis_pols),
        np.nan + 1j * np.nan,
        dtype=complex,
    )
    if np.max(caldata_obj.visibility_weights[:, :, freq_ind, :]) == 0.0:
        print("ERROR: All data flagged.")
        return gains_fit, fit_vis_fit

    for feed_pol_ind, feed_pol in enumerate(caldata_obj.feed_polarization_array):
        vis_pol_ind = np.where(caldata_obj.vis_polarization_array == feed_pol)[0]
        if (
            np.max(caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind]) == 0.0
        ):  # All flagged
            gains_fit[:, feed_pol_ind] = np.nan + 1j * np.nan
        elif np.max(caldata_obj.model_weights[:, :, freq_ind, vis_pol_ind]) == -1.0:
            fit_vis_fit[:, :, vis_pol_ind] = np.nan + 1j * np.nan
        else:
            start_prep = time.time()
            # indices of antennas with nonzero weight
            vis_weights_summed = np.sum(
                caldata_obj.visibility_weights[:, :, freq_ind, feed_pol_ind], axis=0
            )  # Sum over times
            weight_per_ant = np.bincount(
                caldata_obj.ant1_inds,
                weights=vis_weights_summed,
                minlength=caldata_obj.Nants,
            ) + np.bincount(
                caldata_obj.ant2_inds,
                weights=vis_weights_summed,
                minlength=caldata_obj.Nants,
            )
            caldata_obj.ant_inds = np.where(weight_per_ant > 0.0)[0]
            # indices of baselines with nonzero data or model weight
            weights_summed = np.sum(
                (
                    caldata_obj.model_weights[:, :, freq_ind, vis_pol_ind]
                    + caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind]
                ),
                axis=0,
            )
            caldata_obj.bl_inds = np.where(weights_summed > 0.0)[0]
            n_ants_unflagged = len(caldata_obj.ant_inds)

            # data, model, and weights as (Ntimes, Nbls) for the cost wrappers
            reshaped_shape = (
                np.size(caldata_obj.data_visibilities, axis=0),
                np.size(caldata_obj.data_visibilities, axis=1),
            )
            caldata_obj.data_vis_reshaped = np.reshape(
                caldata_obj.data_visibilities[:, :, freq_ind, vis_pol_ind],
                reshaped_shape,
            )
            caldata_obj.model_vis_reshaped = np.reshape(
                caldata_obj.model_visibilities[:, :, freq_ind, vis_pol_ind],
                reshaped_shape,
            )
            caldata_obj.vis_weights_reshaped = np.reshape(
                caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind],
                reshaped_shape,
            )
            caldata_obj.model_weights_reshaped = np.reshape(
                caldata_obj.model_weights[:, :, freq_ind, vis_pol_ind], reshaped_shape
            )

            # gains then fit_vis, real and imaginary parts interleaved, for the
            # SciPy optimizers
            gains_init = caldata_obj.gains[caldata_obj.ant_inds, freq_ind, feed_pol_ind]
            fit_vis_init = caldata_obj.fit_vis[
                :, caldata_obj.bl_inds, freq_ind, feed_pol_ind
            ]
            params_init_flattened = np.hstack(
                (
                    np.stack(
                        (np.real(gains_init), np.imag(gains_init)), axis=1
                    ).flatten(),
                    np.stack(
                        (np.real(fit_vis_init), np.imag(fit_vis_init)), axis=-1
                    ).flatten(),
                )
            )
            if (
                optimization_scheme == "lbfgs"
                or optimization_scheme == "pytorch lbfgs"
                or optimization_scheme == "pytorch"
            ):
                """
                    Do PyTorch L-BFGS optimization
                    (use complex variables directly)
                """
                # MPS (Apple GPU) has no float64/complex128 support, so optimize on CPU
                device = torch.device("cpu")
                if verbose:
                    print("Running PyTorch optimization on CPU.")
                sys.stdout.flush()
                start_optimize = time.time()
                # main params arrays as tensors
                gains_fit_tensor = torch.from_numpy(
                    caldata_obj.gains[
                        caldata_obj.ant_inds, freq_ind, feed_pol_ind
                    ].copy()
                ).to(device=device, dtype=torch.complex128)
                _fit_vis_init = np.reshape(
                    caldata_obj.fit_vis[..., freq_ind, vis_pol_ind],
                    (
                        np.size(caldata_obj.data_visibilities, axis=0), 
                        np.size(caldata_obj.data_visibilities, axis=1),
                    ),
                )
                fit_vis_fit_tensor = torch.from_numpy(_fit_vis_init.copy()).to(
                    device=device, dtype=torch.complex128
                )
                # additional arrays as tensors
                data_tensor = torch.from_numpy(caldata_obj.data_vis_reshaped.copy()).to(
                    device=device, dtype=torch.complex128
                )
                model_tensor = torch.from_numpy(
                    caldata_obj.model_vis_reshaped.copy()
                ).to(device=device, dtype=torch.complex128)
                vis_weights_tensor = torch.from_numpy(
                    caldata_obj.vis_weights_reshaped.copy()
                ).to(device=device, dtype=torch.float64)
                model_weights_tensor = torch.from_numpy(
                    caldata_obj.model_weights_reshaped.copy()
                ).to(device=device, dtype=torch.float64)
                ant_inds_tensor = torch.from_numpy(caldata_obj.ant_inds.copy()).to(
                    device=device, dtype=int
                )
                ant1_inds_tensor = torch.from_numpy(
                    caldata_obj.ant1_inds.copy()
                ).to(
                    device=device, dtype=int
                )
                ant2_inds_tensor = torch.from_numpy(
                    caldata_obj.ant2_inds.copy()
                ).to(
                    device=device, dtype=int
                )
                gains_fit_tensor.requires_grad_(True)
                fit_vis_fit_tensor.requires_grad_(True)
                params = [gains_fit_tensor, fit_vis_fit_tensor]
                # absolute L-BFGS tolerances; convergence is normally decided by
                # the relative loss change (xtol) checked between chunks below
                tolerance_grad = 1e-9
                tolerance_change = 1e-9
                # run L-BFGS in chunks so the best finite solution can be kept
                chunk = min(100, torch_maxiter)
                optimizer = torch.optim.LBFGS(
                    params,
                    lr=1.0,
                    max_iter=chunk,
                    history_size=10,
                    tolerance_grad=tolerance_grad,
                    tolerance_change=tolerance_change,
                    line_search_fn="strong_wolfe",
                )

                # helper functions for two tensor grads
                def _all_finite(tensors):
                    return all(
                        torch.isfinite(torch.view_as_real(tensor)).all()
                        for tensor in tensors
                    )

                def _grads_finite(tensors):
                    return all(
                        tensor.grad is not None
                        and torch.isfinite(torch.view_as_real(tensor.grad)).all()
                        for tensor in tensors
                    )

                # Euclidean norm across two tensor grads
                def _joint_grad_norm(tensors):
                    return torch.sqrt(
                        sum(
                            torch.view_as_real(tensor.grad).pow(2).sum()
                            for tensor in tensors
                            if tensor.grad is not None
                        )
                    )

                # joint gradient norm from the last closure call
                grad_norm_holder = [np.nan]

                def _cost():
                    return cost_function_calculations.cost_unical_torch(
                        gains=gains_fit_tensor,
                        fit_vis=fit_vis_fit_tensor,
                        data_vis=data_tensor,
                        model_vis=model_tensor,
                        vis_weights=vis_weights_tensor,
                        model_weights=model_weights_tensor,
                        ant_inds=ant_inds_tensor,
                        ant1_inds=ant1_inds_tensor,
                        ant2_inds=ant2_inds_tensor,
                        num_ants=caldata_obj.Nants,
                        lambda_val=caldata_obj.lambda_val,
                    )

                def closure():
                    optimizer.zero_grad()
                    loss = _cost()
                    loss.backward()
                    if not _grads_finite(params):
                        grad_norm_holder[0] = np.nan
                        for p in params:
                            if p.grad is not None:
                                p.grad.zero_()
                    else:
                        grad_norm_holder[0] = _joint_grad_norm(params).item()
                    return loss

                best_loss = np.inf
                best_params = [p.detach().clone() for p in params]
                stop_reason = "maxiter"
                n_iter_total = 0
                # the relative loss change is measured over at least this many
                # iterations, so short chunks and plateaus in the loss can't look
                # falsely converged
                rtol_window = 100
                previous_loss = None
                iters_since_check = 0
                while n_iter_total < torch_maxiter:
                    allowed = min(chunk, torch_maxiter - n_iter_total)
                    max_eval = int(allowed * 1.25)
                    optimizer.param_groups[0]["max_iter"] = allowed
                    optimizer.param_groups[0]["max_eval"] = max_eval
                    evals_before = optimizer.state[gains_fit_tensor].get(
                        "func_evals", 0
                    )
                    optimizer.step(closure)
                    # n_iter and func_evals are cumulative across steps
                    state = optimizer.state[gains_fit_tensor]
                    iters_this_step = state["n_iter"] - n_iter_total
                    evals_this_step = state["func_evals"] - evals_before
                    n_iter_total = state["n_iter"]
                    # step() returns the loss from the start of the chunk, so
                    # evaluate it at the parameters the chunk ended on
                    with torch.no_grad():
                        loss_val = _cost().item()
                    if not np.isfinite(loss_val) or not _all_finite(params):
                        stop_reason = "non-finite"
                        break
                    if loss_val < best_loss:
                        best_loss = loss_val
                        best_params = [p.detach().clone() for p in params]
                    # L-BFGS returned early without hitting its eval cap, so one
                    # of its absolute tolerances was met
                    if iters_this_step < allowed and evals_this_step < max_eval:
                        stop_reason = "converged"
                        break
                    iters_since_check += iters_this_step
                    if previous_loss is None:
                        previous_loss = loss_val
                        iters_since_check = 0
                    elif iters_since_check >= rtol_window:
                        if abs(previous_loss - loss_val) <= xtol * max(
                            abs(loss_val), np.finfo(float).tiny
                        ):
                            stop_reason = "converged-rtol"
                            break
                        previous_loss = loss_val
                        iters_since_check = 0
                gains_best, fit_vis_best = best_params
                gains_final = gains_best.detach().cpu().numpy().astype(gains_fit.dtype)
                fit_vis_final = (
                    fit_vis_best.detach().cpu().numpy().astype(fit_vis_fit.dtype)
                )
                if verbose:
                    # recreate scipy
                    print("***PyTorch Result***")
                    state = optimizer.state[gains_fit_tensor]
                    stop_messages = {
                        "converged": "Converged (L-BFGS tolerance reached)",
                        "converged-rtol": (
                            f"Converged (relative loss change <= xtol={xtol:g})"
                        ),
                        "maxiter": "Max iterations reached (may not have converged)",
                        "non-finite": (
                            "Stopped: non-finite values "
                            f"(kept best finite solution, loss={best_loss:.3e})"
                        ),
                    }
                    print(f"\tIterations:         {n_iter_total}")
                    print(f"\tFunction evals:     {state.get('func_evals', 'N/A')}")
                    print(f"\tGradient norm:      {grad_norm_holder[0]:.3e}")
                    print(f"\tMessage: {stop_messages[stop_reason]}")
                    print(
                        f"\tOptimization time:  {
                            (time.time() - start_optimize) / 60
                        } minutes"
                    )
                sys.stdout.flush()
                gains_fit[caldata_obj.ant_inds, feed_pol_ind] = gains_final
                fit_vis_fit[:, caldata_obj.bl_inds, vis_pol_ind] = fit_vis_final
                # fit_vis_fit[0, caldata_obj.Ntimes * caldata_obj.bl_inds, vis_pol_ind] = fit_vis_final

            if (
                optimization_scheme == "powell"
                or optimization_scheme == "scipy powell"
                or optimization_scheme == "scipy"
            ):
                """
                Do SciPy Powell optimization (break up complex variables
                into real and imaginary parts for minimizer function)
                """
                print(
                    f"Preparing params time - {(time.time() - start_prep) / 60} minutes"
                )
                if dev_type == "test gains flattened":
                    return params_init_flattened[: 2 * len(caldata_obj.ant_inds)]
                elif len(dev_type) > 0:
                    return cost_unical_wrapper(
                        params_init_flattened,
                        caldata_obj,
                        caldata_obj.ant_inds,
                        n_ants_unflagged,
                        caldata_obj.Ntimes,
                        caldata_obj.bl_inds + n_ants_unflagged,
                        freq_ind,
                        vis_pol_ind,
                        dev_type=dev_type,
                    )
                # Minimize the cost function
                start_optimize = time.time()
                result = scipy.optimize.minimize(
                    cost_unical_wrapper,
                    params_init_flattened,
                    args=(
                        caldata_obj,
                        caldata_obj.ant_inds,
                        n_ants_unflagged,
                        caldata_obj.Ntimes,
                        caldata_obj.bl_inds + n_ants_unflagged,
                        freq_ind,
                        vis_pol_ind,
                    ),
                    method="Powell",
                    options={"disp": verbose, "xtol": xtol, "maxiter": maxiter},
                )
                print(
                    f"Optimization time: {(time.time() - start_optimize) / 60} minutes"
                )
                end_optimize = time.time()
                if verbose:
                    print(result.message)
                    print(
                        f"Optimization time: {(end_optimize - start_optimize) / 60.0} minutes"
                    )
                sys.stdout.flush()
                gains_fit_single_pol = np.reshape(
                    result.x[: 2 * len(caldata_obj.ant_inds)],
                    (len(caldata_obj.ant_inds), 2),
                )
                gains_fit[caldata_obj.ant_inds, feed_pol_ind] = (
                    gains_fit_single_pol[:, 0] + 1j * gains_fit_single_pol[:, 1]
                )
                # Reshape the fitted u-block back to (Ntimes, Nbls) for storage,
                # inverting the flat or (Ntimes, Nbls) layout used during the solve.
                fit_vis_fit[:, caldata_obj.bl_inds, vis_pol_ind] = (
                    _unpack_unical_fit_vis(
                        result.x[2 * len(caldata_obj.ant_inds) :], caldata_obj
                    )
                )

            if optimization_scheme == "bfgs" or optimization_scheme == "scipy bfgs":
                """
                Do SciPy BFGS optimization (break up complex variables
                into real and imaginary parts for minimizer function)
                """
                print(
                    f"Preparing params time - {(time.time() - start_prep) / 60} minutes"
                )
                start_optimize = time.time()
                result = scipy.optimize.minimize(
                    cost_unical_wrapper,
                    params_init_flattened,
                    args=(
                        caldata_obj,
                        caldata_obj.ant_inds,
                        n_ants_unflagged,
                        caldata_obj.Ntimes,
                        caldata_obj.bl_inds + n_ants_unflagged,
                        freq_ind,
                        vis_pol_ind,
                    ),
                    method="BFGS",
                    jac=jacobian_unical_wrapper,
                    # BFGS has no xtol; it stops on the gradient norm (gtol)
                    options={"disp": verbose, "gtol": xtol, "maxiter": maxiter},
                )
                print(
                    f"Optimization time: {(time.time() - start_optimize) / 60} minutes"
                )
                end_optimize = time.time()
                if verbose:
                    print(result.message)
                    print(
                        f"Optimization time: {(end_optimize - start_optimize) / 60.0} minutes"
                    )
                sys.stdout.flush()
                gains_fit_single_pol = np.reshape(
                    result.x[: 2 * len(caldata_obj.ant_inds)],
                    (len(caldata_obj.ant_inds), 2),
                )
                gains_fit[caldata_obj.ant_inds, feed_pol_ind] = (
                    gains_fit_single_pol[:, 0] + 1j * gains_fit_single_pol[:, 1]
                )
                fit_vis_fit[:, caldata_obj.bl_inds, vis_pol_ind] = (
                    _unpack_unical_fit_vis(
                        result.x[2 * len(caldata_obj.ant_inds) :], caldata_obj
                    )
                )

            if (
                optimization_scheme == "newton-cg"
                or optimization_scheme == "scipy newton-cg"
            ):
                """
                Do SciPy Newton-CGS optimization (break up complex variables
                into real and imaginary parts for minimizer function)
                """
                print(
                    f"Preparing params time - {(time.time() - start_prep) / 60} minutes"
                )
                _confirm_dense_hessian(
                    len(params_init_flattened),
                    caldata_obj.Ntimes,
                    allow_large_hessian,
                )
                start_optimize = time.time()
                result = scipy.optimize.minimize(
                    cost_unical_wrapper,
                    params_init_flattened,
                    args=(
                        caldata_obj,
                        caldata_obj.ant_inds,
                        n_ants_unflagged,
                        caldata_obj.Ntimes,
                        caldata_obj.bl_inds + n_ants_unflagged,
                        freq_ind,
                        vis_pol_ind,
                    ),
                    method="Newton-CG",
                    jac=jacobian_unical_wrapper,
                    hess=hessian_unical_wrapper,
                    options={"disp": verbose, "xtol": xtol, "maxiter": maxiter},
                )
                print(
                    f"Optimization time: {(time.time() - start_optimize) / 60} minutes"
                )
                end_optimize = time.time()
                if verbose:
                    print(result.message)
                    print(
                        f"Optimization time: {(end_optimize - start_optimize) / 60.0} minutes"
                    )
                sys.stdout.flush()
                gains_fit_single_pol = np.reshape(
                    result.x[: 2 * len(caldata_obj.ant_inds)],
                    (len(caldata_obj.ant_inds), 2),
                )
                gains_fit[caldata_obj.ant_inds, feed_pol_ind] = (
                    gains_fit_single_pol[:, 0] + 1j * gains_fit_single_pol[:, 1]
                )
                fit_vis_fit[:, caldata_obj.bl_inds, vis_pol_ind] = (
                    _unpack_unical_fit_vis(
                        result.x[2 * len(caldata_obj.ant_inds) :], caldata_obj
                    )
                )

            # Ensure that the phase of the gains is mean-zero
            # This adds should be handled by the phase regularization term, but
            # this step removes any optimizer precision effects.
            avg_angle = np.arctan2(
                np.nanmean(np.sin(np.angle(gains_fit[:, feed_pol_ind]))),
                np.nanmean(np.cos(np.angle(gains_fit[:, feed_pol_ind]))),
            )
            gains_fit[:, feed_pol_ind] *= np.cos(avg_angle) - 1j * np.sin(avg_angle)

    # Constrain crosspol phase
    if (
        get_crosspol_phase
        and caldata_obj.N_feed_pols == 2
        and caldata_obj.N_vis_pols == 4
    ):
        if (
            caldata_obj.feed_polarization_array[0] == -5
            and caldata_obj.feed_polarization_array[1] == -6
        ):
            crosspol_polarizations = [-7, -8]
        elif (
            caldata_obj.feed_polarization_array[0] == -6
            and caldata_obj.feed_polarization_array[1] == -5
        ):
            crosspol_polarizations = [-8, -7]
        crosspol_indices = np.array(
            [
                np.where(caldata_obj.vis_polarization_array == pol)[0][0]
                for pol in crosspol_polarizations
            ]
        )
        if crosspol_phase_strategy.lower() == "pseudo stokes v":
            crosspol_phase = cost_function_calculations.set_crosspol_phase_pseudoV(
                gains_fit,
                caldata_obj.data_visibilities[:, :, freq_ind, crosspol_indices],
                caldata_obj.visibility_weights[:, :, freq_ind, crosspol_indices],
                caldata_obj.ant1_inds,
                caldata_obj.ant2_inds,
            )
        elif crosspol_phase_strategy.lower() == "crosspol model":
            crosspol_phase = cost_function_calculations.set_crosspol_phase(
                gains_fit,
                caldata_obj.model_visibilities[:, :, freq_ind, crosspol_indices],
                caldata_obj.data_visibilities[:, :, freq_ind, crosspol_indices],
                caldata_obj.visibility_weights[:, :, freq_ind, crosspol_indices],
                caldata_obj.ant1_inds,
                caldata_obj.ant2_inds,
            )
        else:
            print(
                "WARNING: Unknown crosspol_phase_strategy. Skipping fitting crosspol phase."
            )
            crosspol_phase = 0.0

        if caldata_obj.gains_multiply_model:
            gains_fit[:, 0] /= np.exp(-1j * crosspol_phase / 2)
            gains_fit[:, 1] /= np.exp(1j * crosspol_phase / 2)
        else:
            gains_fit[:, 0] *= np.exp(-1j * crosspol_phase / 2)
            gains_fit[:, 1] *= np.exp(1j * crosspol_phase / 2)

    return gains_fit, fit_vis_fit


def run_unical_optimization_parallel(args):
    """
    Wrapper for run_unical_optimization that makes the function compatible with
    multiprocessing by unpacking a tuple of arguments. The caldata subset holds a
    single frequency, so it is calibrated at freq_ind=0.
    """
    (caldata_subset, freq_ind, unical_kwargs) = args
    gains_fit, fit_vis_fit = run_unical_optimization(
        caldata_subset, freq_ind=0, **unical_kwargs
    )
    return freq_ind, gains_fit, fit_vis_fit
