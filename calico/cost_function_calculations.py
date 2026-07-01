import numpy as np
import torch
import sys
from calico import utils
import time

def cost_skycal(
    gains,
    model_visibilities,
    data_visibilities,
    visibility_weights,
    ant1_inds,
    ant2_inds,
    lambda_val,
):
    """
    Calculate the cost function (chi-squared) value.

    Parameters
    ----------
    gains : array of complex
        Shape (Nants,).
    model_visibilities :  array of complex
        Shape (Ntimes, Nbls,).
    data_visibilities : array of complex
        Shape (Ntimes, Nbls,).
    visibility_weights : array of float
        Shape (Ntimes, Nbls,).
    ant1_inds : array of int
        Shape (Nbls,).
    ant2_inds : array of int
        Shape (Nbls,).
    lambda_val : float
        Weight of the phase regularization term; must be positive.

    Returns
    -------
    cost : float
        Value of the cost function.
    """
    gains_expanded = (gains[ant1_inds] * np.conj(gains[ant2_inds]))[np.newaxis, :]
    res_vec = model_visibilities - gains_expanded * data_visibilities
    cost = np.sum(visibility_weights * np.abs(res_vec) ** 2)
    if lambda_val > 0:
        regularization_term = lambda_val * np.sum(np.angle(gains)) ** 2.0
        cost += regularization_term
    return cost


def jacobian_skycal(
    gains,
    model_visibilities,
    data_visibilities,
    visibility_weights,
    ant1_inds,
    ant2_inds,
    lambda_val,
):
    """
    Calculate the Jacobian of the cost function.

    Parameters
    ----------
    gains : array of complex
        Shape (Nants,).
    model_visibilities :  array of complex
        Shape (Ntimes, Nbls,).
    data_visibilities : array of complex
        Shape (Ntimes, Nbls,).
    visibility_weights : array of float
        Shape (Ntimes, Nbls,).
    ant1_inds : array of int
        Shape (Nbls,).
    ant2_inds : array of int
        Shape (Nbls,).
    lambda_val : float
        Weight of the phase regularization term; must be positive.

    Returns
    -------
    jac : array of complex
        Jacobian of the chi-squared cost function, shape (Nants,). The real part
        corresponds to derivatives with respect to the real part of the gains;
        the imaginary part corresponds to derivatives with respect to the
        imaginary part of the gains.    
    """

    start_jac = time.time()

    # Convert gains to visibility space
    # Add time axis
    gains_expanded_1 = gains[np.newaxis, ant1_inds]  # shape (1,Nbls)
    gains_expanded_2 = gains[np.newaxis, ant2_inds]  # shape (1,Nbls)

    res_vec = (
        gains_expanded_1 * np.conj(gains_expanded_2) * data_visibilities
        - model_visibilities
    )
    term1 = np.sum(
        visibility_weights * gains_expanded_2 * np.conj(data_visibilities) * res_vec,
        axis=0,
    )
    term1 = utils.bincount_multidim(
        ant1_inds,
        weights=term1,
        minlength=np.max([np.max(ant1_inds), np.max(ant2_inds)]) + 1,
    )
    term2 = np.sum(
        visibility_weights * gains_expanded_1 * data_visibilities * np.conj(res_vec),
        axis=0,
    )
    term2 = utils.bincount_multidim(
        ant2_inds,
        weights=term2,
        minlength=np.max([np.max(ant1_inds), np.max(ant2_inds)]) + 1,
    )

    jac = 2 * (term1 + term2)

    if lambda_val > 0:
        regularization_term = (
            lambda_val * 1j * np.sum(np.angle(gains)) * gains / np.abs(gains) ** 2.0
        )
        jac += 2 * regularization_term

    end_jac = time.time()
    print("***JACOBIAN TIME***", (end_jac - start_jac)/60.)

    return jac


def reformat_baselines_to_antenna_matrix(
    bl_array,
    ant1_inds,
    ant2_inds,
    Nants,
    Nbls,
):
    """
    Reformat an array indexed in baselines into a matrix with antenna indices.

    Parameters
    ----------
    bl_array : array of float or complex
        Shape (Nbls, ...,).
    ant1_inds : array of int
        Shape (Nbls,).
    ant2_inds : array of int
        Shape (Nbls,).
    Nants : int
        Number of antennas.
    Nbls : int
        Number of baselines.

    Returns
    -------
    antenna matrix : array of float or complex
        Shape (Nants, Nants, ...,). Same dtype as bl_array.
    """

    antenna_matrix = np.zeros_like(
        bl_array[0,],
        dtype=bl_array.dtype,
    )
    antenna_matrix = np.repeat(
        np.repeat(antenna_matrix[np.newaxis,], Nants, axis=0)[np.newaxis,],
        Nants,
        axis=0,
    )
    for bl_ind in range(Nbls):
        antenna_matrix[
            ant1_inds[bl_ind],
            ant2_inds[bl_ind],
        ] = bl_array[
            bl_ind,
        ]
    return antenna_matrix


def hessian_skycal(
    gains,
    Nants,
    Nbls,
    model_visibilities,
    data_visibilities,
    visibility_weights,
    ant1_inds,
    ant2_inds,
    lambda_val,
):
    """
    Calculate the Hessian of the cost function.

    Parameters
    ----------
    gains : array of complex
        Shape (Nants,).
    Nants : int
        Number of antennas.
    Nbls : int
        Number of baselines.
    model_visibilities : array of complex
        Shape (Ntimes, Nbls,).
    data_visibilities : array of complex
        Shape (Ntimes, Nbls,).
    visibility_weights : array of float
        Shape (Ntimes, Nbls,).
    ant1_inds : array of int
        Shape (Nbls,).
    ant2_inds : array of int
        Shape (Nbls,).
    lambda_val : float
        Weight of the phase regularization term; must be positive.

    Returns
    -------
    hess_real_real : array of float
        Real-real derivative components of the Hessian of the cost function.
        Shape (Nants, Nants,).
    hess_real_imag : array of float
        Real-imaginary derivative components of the Hessian of the cost
        function. Note that the transpose of this array gives the imaginary-real
        derivative components. Shape (Nants, Nants,).
    hess_imag_imag : array of float
        Imaginary-imaginary derivative components of the Hessian of the cost
        function. Shape (Nants, Nants,).
    """

    start_hess = time.time()

    gains_expanded_1 = gains[ant1_inds]
    gains_expanded_2 = gains[ant2_inds]
    data_squared = np.sum(visibility_weights * np.abs(data_visibilities) ** 2.0, axis=0)
    data_times_model = np.sum(
        visibility_weights * model_visibilities * np.conj(data_visibilities), axis=0
    )

    # Calculate the antenna off-diagonal components
    hess_components = np.zeros((Nbls, 4), dtype=float)
    # Real-real Hessian component:
    hess_components[:, 0] = np.real(
        4 * np.real(gains_expanded_1) * np.real(gains_expanded_2) * data_squared
        - 2 * np.real(data_times_model)
    )
    # Real-imaginary Hessian component, term 1:
    hess_components[:, 1] = np.real(
        4 * np.real(gains_expanded_1) * np.imag(gains_expanded_2) * data_squared
        + 2 * np.imag(data_times_model)
    )
    # Real-imaginary Hessian component, term 2:
    hess_components[:, 2] = np.real(
        4 * np.imag(gains_expanded_1) * np.real(gains_expanded_2) * data_squared
        - 2 * np.imag(data_times_model)
    )
    # Imaginary-imaginary Hessian component:
    hess_components[:, 3] = np.real(
        4 * np.imag(gains_expanded_1) * np.imag(gains_expanded_2) * data_squared
        - 2 * np.real(data_times_model)
    )

    hess_components = reformat_baselines_to_antenna_matrix(
        hess_components,
        ant1_inds,
        ant2_inds,
        Nants,
        Nbls,
    )
    hess_real_real = hess_components[:, :, 0] + hess_components[:, :, 0].T
    hess_real_imag = hess_components[:, :, 1] + hess_components[:, :, 2].T
    hess_imag_imag = hess_components[:, :, 3] + hess_components[:, :, 3].T

    # Calculate the antenna diagonals
    hess_diag = 2 * (
        utils.bincount_multidim(
            ant1_inds,
            weights=np.abs(gains_expanded_2) ** 2.0 * data_squared,
            minlength=Nants,
        )
        + utils.bincount_multidim(
            ant2_inds,
            weights=np.abs(gains_expanded_1) ** 2.0 * data_squared,
            minlength=Nants,
        )
    )
    np.fill_diagonal(hess_real_real, hess_diag)
    np.fill_diagonal(hess_imag_imag, hess_diag)
    np.fill_diagonal(hess_real_imag, 0.0)

    if lambda_val > 0:  # Add regularization term
        gains_weighted = gains / np.abs(gains) ** 2.0
        arg_sum = np.sum(np.angle(gains))
        # Antenna off-diagonals
        hess_real_real += (
            2 * lambda_val * np.outer(np.imag(gains_weighted), np.imag(gains_weighted))
        )
        hess_real_imag -= (
            2 * lambda_val * np.outer(np.imag(gains_weighted), np.real(gains_weighted))
        )
        hess_imag_imag += (
            2 * lambda_val * np.outer(np.real(gains_weighted), np.real(gains_weighted))
        )
        # Antenna diagonals
        hess_real_real += np.diag(
            4 * lambda_val * arg_sum * np.imag(gains_weighted) * np.real(gains_weighted)
        )
        hess_real_imag -= np.diag(
            2
            * lambda_val
            * arg_sum
            * (np.real(gains_weighted) ** 2.0 - np.imag(gains_weighted) ** 2.0)
        )
        hess_imag_imag -= np.diag(
            4 * lambda_val * arg_sum * np.imag(gains_weighted) * np.real(gains_weighted)
        )

    end_hess = time.time()
    print("***HESSIAN TIME***", (end_hess - start_hess)/60.)

    return hess_real_real, hess_real_imag, hess_imag_imag


def set_crosspol_phase(
    gains,
    crosspol_model_visibilities,
    crosspol_data_visibilities,
    crosspol_visibility_weights,
    ant1_inds,
    ant2_inds,
):
    """
    Calculate the cross-polarization phase between the P and Q gains. This
    quantity is not constrained in typical per-polarization calibration but is
    required for polarized imaging. See Byrne et al. 2022 for details of the
    calculation.

    Parameters
    ----------
    gains : array of complex
        Shape (Nants, 2,). gains[:, 0] corresponds to the P-polarized gains and
        gains[:, 1] corresponds to the Q-polarized gains.
    crosspol_model_visibilities :  array of complex
        Shape (Ntimes, Nbls, 2,). Cross-polarized model visibilities.
        model_visilibities[:, :, 0] corresponds to the PQ-polarized visibilities
        and model_visilibities[:, :, 1] corresponds to the QP-polarized
        visibilities.
    crosspol_data_visibilities : array of complex
        Shape (Ntimes, Nbls, 2,). Cross-polarized data visibilities.
        model_visilibities[:, :, 0] corresponds to the PQ-polarized visibilities
        and model_visilibities[:, :, 1] corresponds to the QP-polarized
        visibilities.
    crosspol_visibility_weights : array of float
        Shape (Ntimes, Nbls, 2).
    ant1_inds : array of int
        Shape (Nbls,).
    ant2_inds : array of int
        Shape (Nbls,).

    Returns
    -------
    crosspol_phase : float
        Cross-polarization phase, in radians.
    """

    gains_expanded_1 = gains[np.newaxis, ant1_inds, :]
    gains_expanded_2 = gains[np.newaxis, ant2_inds, :]
    term1 = np.nansum(
        crosspol_visibility_weights[:, :, 0]
        * np.conj(crosspol_model_visibilities[:, :, 0])
        * gains_expanded_1[:, :, 0]
        * np.conj(gains_expanded_2[:, :, 1])
        * crosspol_data_visibilities[:, :, 0]
    )
    term2 = np.nansum(
        crosspol_visibility_weights[:, :, 1]
        * crosspol_model_visibilities[:, :, 1]
        * np.conj(gains_expanded_1[:, :, 1])
        * gains_expanded_2[:, :, 0]
        * np.conj(crosspol_data_visibilities[:, :, 1])
    )
    crosspol_phase = np.angle(term1 + term2)

    return crosspol_phase


def set_crosspol_phase_pseudoV(
    gains,
    crosspol_data_visibilities,
    crosspol_visibility_weights,
    ant1_inds,
    ant2_inds,
):
    """
    Calculate the cross-polarization phase between the P and Q gains. This
    quantity is not constrained in typical per-polarization calibration but is
    required for polarized imaging. See Byrne et al. 2022 for details of the
    calculation.

    Parameters
    ----------
    gains : array of complex
        Shape (Nants, 2,). gains[:, 0] corresponds to the P-polarized gains and
        gains[:, 1] corresponds to the Q-polarized gains.
    crosspol_data_visibilities : array of complex
        Shape (Ntimes, Nbls, 2,). Cross-polarized data visibilities.
        model_visilibities[:, :, 0] corresponds to the PQ-polarized visibilities
        and model_visilibities[:, :, 1] corresponds to the QP-polarized
        visibilities.
    crosspol_visibility_weights : array of float
        Shape (Ntimes, Nbls, 2).
    ant1_inds : array of int
        Shape (Nbls,).
    ant2_inds : array of int
        Shape (Nbls,).

    Returns
    -------
    crosspol_phase : float
        Cross-polarization phase, in radians.
    """

    gains_expanded_1 = gains[np.newaxis, ant1_inds, :]
    gains_expanded_2 = gains[np.newaxis, ant2_inds, :]
    crosspol_data_visibilities_calibrated = crosspol_data_visibilities
    crosspol_data_visibilities_calibrated[:, :, 0] *= gains_expanded_1[
        :, :, 0
    ] * np.conj(
        gains_expanded_2[:, :, 1]
    )  # Apply gains to PQ visibilities
    crosspol_data_visibilities_calibrated[:, :, 1] *= gains_expanded_1[
        :, :, 1
    ] * np.conj(
        gains_expanded_2[:, :, 0]
    )  # Apply gains to QP visibilities
    visibility_weights = np.nanmean(
        crosspol_visibility_weights, axis=2
    )  # Doesn't support different weights for PQ and QP
    sum_term = np.nansum(
        visibility_weights
        * crosspol_data_visibilities_calibrated[:, :, 0]
        * np.conj(crosspol_data_visibilities_calibrated[:, :, 1])
    )
    crosspol_phase = np.angle(sum_term) / 2
    return crosspol_phase


def cost_function_abs_cal(
    amp,
    phase_grad,
    model_visibilities,
    data_visibilities,
    uv_array,
    visibility_weights,
):
    """
    Calculate the cost function (chi-squared) value for absolute calibration.

    Parameters
    ----------
    amp : float
        Overall visibility amplitude.
    phase_grad :  array of float
        Shape (2,). Phase gradient terms, in units of 1/m.
    model_visibilities : array of complex
        Shape (Ntimes, Nbls,).
    data_visibilities : array of complex
        Relatively calibrated data. Shape (Ntimes, Nbls,).
    uv_array : array of float
        Shape(Nbls, 2,)
    visibility_weights : array of float
        Shape (Ntimes, Nbls,).

    Returns
    -------
    cost : float
        Value of the cost function.
    """

    phase_term = np.sum(phase_grad[np.newaxis, :] * uv_array, axis=1)
    res_vec = (amp**2.0 * np.exp(1j * phase_term))[
        np.newaxis, :
    ] * data_visibilities - model_visibilities
    cost = np.sum(visibility_weights * np.abs(res_vec) ** 2)
    return cost


def jacobian_abs_cal(
    amp,
    phase_grad,
    model_visibilities,
    data_visibilities,
    uv_array,
    visibility_weights,
):
    """
    Calculate the Jacobian for absolute calibration.

    Parameters
    ----------
    amp : float
        Overall visibility amplitude.
    phase_grad :  array of float
        Shape (2,). Phase gradient terms, in units of 1/m.
    model_visibilities : array of complex
        Shape (Ntimes, Nbls,).
    data_visibilities : array of complex
        Relatively calibrated data. Shape (Ntimes, Nbls,).
    uv_array : array of float
        Shape(Nbls, 2,)
    visibility_weights : array of float
        Shape (Ntimes, Nbls,).

    Returns
    -------
    amp_jac : float
        Derivative of the cost with respect to the visibility amplitude term.
    phase_jac : array of float
        Derivatives of the cost with respect to the phase gradient terms. Shape (2,).
    """

    phase_term = np.sum(phase_grad[np.newaxis, :] * uv_array, axis=1)
    data_prod = (
        np.exp(1j * phase_term)[np.newaxis, :]
        * data_visibilities
        * np.conj(model_visibilities)
    )

    amp_jac = (
        4
        * amp
        * np.sum(
            visibility_weights
            * (amp**2.0 * np.abs(data_visibilities) ** 2.0 - np.real(data_prod))
        )
    )
    phase_jac = (
        2
        * amp**2.0
        * np.sum(
            visibility_weights[:, :, np.newaxis]
            * uv_array[np.newaxis, :, :]
            * np.imag(data_prod)[:, :, np.newaxis],
            axis=(0, 1),
        )
    )

    return amp_jac, phase_jac


def hess_abs_cal(
    amp,
    phase_grad,
    model_visibilities,
    data_visibilities,
    uv_array,
    visibility_weights,
):
    """
    Calculate the Hessian for absolute calibration.

    Parameters
    ----------
    amp : float
        Overall visibility amplitude.
    phase_grad :  array of float
        Shape (2,). Phase gradient terms, in units of 1/m.
    model_visibilities : array of complex
        Shape (Ntimes, Nbls,).
    data_visibilities : array of complex
        Relatively calibrated data. Shape (Ntimes, Nbls,).
    uv_array : array of float
        Shape(Nbls, 2,)
    visibility_weights : array of float
        Shape (Ntimes, Nbls,).


    Returns
    -------
    hess_amp_amp : float
        Second derivative of the cost with respect to the amplitude term.
    hess_amp_phasex : float
        Second derivative of the cost with respect to the amplitude term and the phase gradient in x.
    hess_amp_phasey : float
        Second derivative of the cost with respect to the amplitude term and the phase gradient in y.
    hess_phasex_phasex : float
        Second derivative of the cost with respect to the phase gradient in x.
    hess_phasey_phasey : float
        Second derivative of the cost with respect to the phase gradient in x.
    hess_phasex_phasey : float
        Second derivative of the cost with respect to the phase gradient in x and y.
    """

    phase_term = np.sum(phase_grad[np.newaxis, :] * uv_array, axis=1)
    data_prod = (
        np.exp(1j * phase_term)[np.newaxis, :]
        * data_visibilities
        * np.conj(model_visibilities)
    )

    hess_amp_amp = np.sum(
        visibility_weights
        * (
            12.0 * amp**2.0 * np.abs(data_visibilities) ** 2.0
            - 4.0 * np.real(data_prod)
        )
    )

    hess_amp_phasex = (
        4.0
        * amp
        * np.sum(visibility_weights * uv_array[np.newaxis, :, 0] * np.imag(data_prod))
    )
    hess_amp_phasey = (
        4.0
        * amp
        * np.sum(visibility_weights * uv_array[np.newaxis, :, 1] * np.imag(data_prod))
    )

    hess_phasex_phasex = (
        2.0
        * amp**2.0
        * np.sum(
            visibility_weights * uv_array[np.newaxis, :, 0] ** 2.0 * np.real(data_prod)
        )
    )

    hess_phasey_phasey = (
        2.0
        * amp**2.0
        * np.sum(
            visibility_weights * uv_array[np.newaxis, :, 1] ** 2.0 * np.real(data_prod)
        )
    )

    hess_phasex_phasey = (
        2.0
        * amp**2.0
        * np.sum(
            visibility_weights
            * uv_array[np.newaxis, :, 0]
            * uv_array[np.newaxis, :, 1]
            * np.real(data_prod)
        )
    )

    return (
        hess_amp_amp,
        hess_amp_phasex,
        hess_amp_phasey,
        hess_phasex_phasex,
        hess_phasey_phasey,
        hess_phasex_phasey,
    )


def cost_function_dw_abscal(
    amp,
    phase_grad,
    model_visibilities,
    data_visibilities,
    uv_array,
    visibility_weights,
    dwcal_inv_covariance,
):
    """
    Calculate the cost function (chi-squared) value for absolute calibration
    with delay weighting.

    Parameters
    ----------
    amp : array of float
        Shape (Nfreqs,). Overall visibility amplitude.
    phase_grad :  array of float
        Shape (2, Nfreqs,). Phase gradient terms, in units of 1/m.
    model_visibilities : array of complex
        Shape (Ntimes, Nbls, Nfreqs,).
    data_visibilities : array of complex
        Relatively calibrated data. Shape (Ntimes, Nbls, Nfreqs,).
    uv_array : array of float
        Shape(Nbls, 2,)
    visibility_weights : array of float
        Shape (Ntimes, Nbls, Nfreqs,).
    dwcal_inv_covariance : array of complex
        Shape (Ntimes, Nbls, Nfreqs, Nfreqs,).

    Returns
    -------
    cost : float
        Value of the cost function.
    """

    phase_term = np.sum(
        phase_grad[np.newaxis, :, :] * uv_array[:, :, np.newaxis], axis=1
    )  # Shape (Nbls, Nfreqs,)
    res_vec = np.sqrt(visibility_weights) * (
        (amp[np.newaxis, :] ** 2.0 * np.exp(1j * phase_term))[np.newaxis, :, :]
        * data_visibilities
        - model_visibilities
    )  # Shape (Ntimes, Nbls, Nfreqs)
    cost = np.real(
        np.sum(
            dwcal_inv_covariance
            * np.conj(res_vec[:, :, :, np.newaxis])
            * res_vec[:, :, np.newaxis, :]
        )
    )
    print(f"DWAbscal cost: {cost}")
    sys.stdout.flush()
    return cost


def cost_function_dw_abscal_toeplitz(
    amp,
    phase_grad,
    model_visibilities,
    data_visibilities,
    uv_array,
    visibility_weights,
    dwcal_inv_covariance,
):
    """
    Calculate the cost function (chi-squared) value for absolute calibration
    with delay weighting.

    Parameters
    ----------
    amp : array of float
        Shape (Nfreqs,). Overall visibility amplitude.
    phase_grad :  array of float
        Shape (2, Nfreqs,). Phase gradient terms, in units of 1/m.
    model_visibilities : array of complex
        Shape (Ntimes, Nbls, Nfreqs,).
    data_visibilities : array of complex
        Relatively calibrated data. Shape (Ntimes, Nbls, Nfreqs,).
    uv_array : array of float
        Shape(Nbls, 2,)
    visibility_weights : array of float
        Shape (Ntimes, Nbls, Nfreqs,).
    dwcal_inv_covariance : array of complex
        Shape (Ntimes, Nbls, Nfreqs,).

    Returns
    -------
    cost : float
        Value of the cost function.
    """

    phase_term = np.sum(
        phase_grad[np.newaxis, :, :] * uv_array[:, :, np.newaxis], axis=1
    )  # Shape (Nbls, Nfreqs,)
    res_vec = np.sqrt(visibility_weights) * (
        (amp[np.newaxis, :] ** 2.0 * np.exp(1j * phase_term))[np.newaxis, :, :]
        * data_visibilities
        - model_visibilities
    )  # Shape (Ntimes, Nbls, Nfreqs)
    cost = np.real(
        np.sum(
            dwcal_inv_covariance
            * np.conj(res_vec[:, :, :, np.newaxis])
            * res_vec[:, :, np.newaxis, :]
        )
    )
    print(f"DWAbscal cost: {cost}")
    sys.stdout.flush()
    return cost


def jacobian_dw_abscal(
    amp,
    phase_grad,
    model_visibilities,
    data_visibilities,
    uv_array,
    visibility_weights,
    dwcal_inv_covariance,
):
    """
    Calculate the Jacobian for absolute calibration with delay weighting.

    Parameters
    ----------
    amp : array of float
        Shape (Nfreqs,). Overall visibility amplitude.
    phase_grad :  array of float
        Shape (2, Nfreqs,). Phase gradient terms, in units of 1/m.
    model_visibilities : array of complex
        Shape (Ntimes, Nbls, Nfreqs,).
    data_visibilities : array of complex
        Relatively calibrated data. Shape (Ntimes, Nbls, Nfreqs,).
    uv_array : array of float
        Shape(Nbls, 2,)
    visibility_weights : array of float
        Shape (Ntimes, Nbls, Nfreqs,).
    dwcal_inv_covariance : array of complex
        Shape (Ntimes, Nbls, Nfreqs, Nfreqs,).

    Returns
    -------
    amp_jac : array of float
        Derivative of the cost with respect to the visibility amplitude terms. Shape (Nfreqs,).
    phase_jac : array of float
        Derivatives of the cost with respect to the phase gradient terms. Shape (2, Nfreqs,).

    """

    phase_term = np.sum(
        phase_grad[np.newaxis, :, :] * uv_array[:, :, np.newaxis], axis=1
    )  # Shape (Nbls, Nfreqs,)
    res_vec = np.sqrt(visibility_weights) * (
        (amp[np.newaxis, :] ** 2.0 * np.exp(1j * phase_term))[np.newaxis, :, :]
        * data_visibilities
        - model_visibilities
    )  # Shape (Ntimes, Nbls, Nfreqs,)
    derivative_term = (
        np.sqrt(visibility_weights)
        * np.exp(-1j * phase_term)[np.newaxis, :, :]
        * np.conj(data_visibilities)
    )
    amp_jac = (
        4
        * amp
        * np.real(
            np.sum(
                dwcal_inv_covariance
                * derivative_term[:, :, :, np.newaxis]
                * res_vec[:, :, np.newaxis, :],
                axis=(0, 1, 3),
            )
        )
    )
    phase_jac = (
        2
        * amp[:, np.newaxis] ** 2.0
        * np.real(
            np.sum(
                dwcal_inv_covariance[:, :, :, :, np.newaxis]
                * (-1j)
                * uv_array[np.newaxis, :, np.newaxis, np.newaxis, :]
                * derivative_term[:, :, :, np.newaxis, np.newaxis]
                * res_vec[:, :, np.newaxis, :, np.newaxis],
                axis=(0, 1, 3),
            )
        )
    ).T
    return amp_jac, phase_jac


def hess_dw_abscal(
    amp,
    phase_grad,
    model_visibilities,
    data_visibilities,
    uv_array,
    visibility_weights,
    dwcal_inv_covariance,
):
    """
    Calculate the Hessian for absolute calibration with delay weighting.

    Parameters
    ----------
    amp : array of float
        Shape (Nfreqs,). Overall visibility amplitude.
    phase_grad :  array of float
        Shape (2, Nfreqs,). Phase gradient terms, in units of 1/m.
    model_visibilities : array of complex
        Shape (Ntimes, Nbls, Nfreqs,).
    data_visibilities : array of complex
        Relatively calibrated data. Shape (Ntimes, Nbls, Nfreqs,).
    uv_array : array of float
        Shape(Nbls, 2,)
    visibility_weights : array of float
        Shape (Ntimes, Nbls, Nfreqs,).
    dwcal_inv_covariance : array of complex
        Shape (Ntimes, Nbls, Nfreqs, Nfreqs,).

    Returns
    hess_amp_amp : array of float
        Shape (Nfreqs, Nfreqs,). Second derivative of the cost with respect to the
        amplitude term.
    hess_amp_phasex : array of float
        Shape (Nfreqs, Nfreqs,). Second derivative of the cost with respect to the
        amplitude term and the phase gradient in x.
    hess_amp_phasey : array of float
        Shape (Nfreqs, Nfreqs,). Second derivative of the cost with respect to the
        amplitude term and the phase gradient in y.
    hess_phasex_phasex : array of float
        Shape (Nfreqs, Nfreqs,). Second derivative of the cost with respect to the
        phase gradient in x.
    hess_phasey_phasey : array of float
        Shape (Nfreqs, Nfreqs,). Second derivative of the cost with respect to the
        phase gradient in x.
    hess_phasex_phasey : array of float
        Shape (Nfreqs, Nfreqs,). Second derivative of the cost with respect to the
        phase gradient in x and y.
    -------

    """

    phase_term = np.sum(
        phase_grad[np.newaxis, :, :] * uv_array[:, :, np.newaxis], axis=1
    )  # Shape (Nbls, Nfreqs,)
    res_vec = np.sqrt(visibility_weights) * (
        (amp[np.newaxis, :] ** 2.0 * np.exp(1j * phase_term))[np.newaxis, :, :]
        * data_visibilities
        - model_visibilities
    )  # Shape (Ntimes, Nbls, Nfreqs,)
    derivative_term = (
        np.sqrt(visibility_weights)
        * np.exp(-1j * phase_term)[np.newaxis, :, :]
        * np.conj(data_visibilities)
    )  # Shape (Ntimes, Nbls, Nfreqs,)

    hess_amp_amp_diagonal_term = 4 * np.real(
        np.sum(
            dwcal_inv_covariance
            * derivative_term[:, :, :, np.newaxis]
            * res_vec[:, :, np.newaxis, :],
            axis=(0, 1, 3),
        )
    )
    hess_amp_amp = (
        8
        * amp[:, np.newaxis]
        * amp[np.newaxis, :]
        * np.real(
            np.sum(
                dwcal_inv_covariance
                * derivative_term[:, :, :, np.newaxis]
                * np.conj(derivative_term[:, :, np.newaxis, :]),
                axis=(0, 1),
            )
        )
    ) + np.diag(hess_amp_amp_diagonal_term)

    hess_amp_phase_diagonal_term = (
        4
        * amp[:, np.newaxis]
        * np.real(
            np.sum(
                dwcal_inv_covariance[:, :, :, :, np.newaxis]
                * (-1j)
                * uv_array[np.newaxis, :, np.newaxis, np.newaxis, :]
                * derivative_term[:, :, :, np.newaxis, np.newaxis]
                * res_vec[:, :, np.newaxis, :, np.newaxis],
                axis=(0, 1, 3),
            )
        )
    )
    hess_amp_phase = (
        4
        * amp[:, np.newaxis, np.newaxis] ** 2.0
        * amp[np.newaxis, :, np.newaxis]
        * np.real(
            np.sum(
                dwcal_inv_covariance[:, :, :, :, np.newaxis]
                * (-1j)
                * uv_array[np.newaxis, :, np.newaxis, np.newaxis, :]
                * derivative_term[:, :, np.newaxis, :, np.newaxis]
                * np.conj(derivative_term[:, :, :, np.newaxis, np.newaxis]),
                axis=(0, 1),
            )
        )
    )
    hess_amp_phasex = hess_amp_phase[:, :, 0] + np.diag(
        hess_amp_phase_diagonal_term[:, 0]
    )
    hess_amp_phasey = hess_amp_phase[:, :, 1] + np.diag(
        hess_amp_phase_diagonal_term[:, 1]
    )

    hess_phasex_phasex = (
        2
        * amp[:, np.newaxis] ** 2.0
        * amp[np.newaxis, :] ** 2.0
        * np.real(
            np.sum(
                dwcal_inv_covariance
                * uv_array[np.newaxis, :, np.newaxis, np.newaxis, 0] ** 2.0
                * derivative_term[:, :, np.newaxis, :]
                * np.conj(derivative_term[:, :, :, np.newaxis]),
                axis=(0, 1),
            )
        )
    )
    hess_phasey_phasey = (
        2
        * amp[:, np.newaxis] ** 2.0
        * amp[np.newaxis, :] ** 2.0
        * np.real(
            np.sum(
                dwcal_inv_covariance
                * uv_array[np.newaxis, :, np.newaxis, np.newaxis, 1] ** 2.0
                * derivative_term[:, :, np.newaxis, :]
                * np.conj(derivative_term[:, :, :, np.newaxis]),
                axis=(0, 1),
            )
        )
    )
    hess_phasex_phasey = (
        2
        * amp[:, np.newaxis] ** 2.0
        * amp[np.newaxis, :] ** 2.0
        * np.real(
            np.sum(
                dwcal_inv_covariance
                * uv_array[np.newaxis, :, np.newaxis, np.newaxis, 0]
                * uv_array[np.newaxis, :, np.newaxis, np.newaxis, 1]
                * derivative_term[:, :, np.newaxis, :]
                * np.conj(derivative_term[:, :, :, np.newaxis]),
                axis=(0, 1),
            )
        )
    )
    hess_phasex_phasex_diagonal_term = (
        -2
        * amp**2.0
        * np.real(
            np.sum(
                dwcal_inv_covariance
                * uv_array[np.newaxis, :, np.newaxis, np.newaxis, 0] ** 2.0
                * derivative_term[:, :, :, np.newaxis]
                * res_vec[:, :, np.newaxis, :],
                axis=(0, 1, 3),
            )
        )
    )
    hess_phasey_phasey_diagonal_term = (
        -2
        * amp**2.0
        * np.real(
            np.sum(
                dwcal_inv_covariance
                * uv_array[np.newaxis, :, np.newaxis, np.newaxis, 1] ** 2.0
                * derivative_term[:, :, :, np.newaxis]
                * res_vec[:, :, np.newaxis, :],
                axis=(0, 1, 3),
            )
        )
    )
    hess_phasex_phasey_diagonal_term = (
        -2
        * amp**2.0
        * np.real(
            np.sum(
                dwcal_inv_covariance
                * uv_array[np.newaxis, :, np.newaxis, np.newaxis, 0]
                * uv_array[np.newaxis, :, np.newaxis, np.newaxis, 1]
                * derivative_term[:, :, :, np.newaxis]
                * res_vec[:, :, np.newaxis, :],
                axis=(0, 1, 3),
            )
        )
    )
    hess_phasex_phasex += np.diag(hess_phasex_phasex_diagonal_term)
    hess_phasey_phasey += np.diag(hess_phasey_phasey_diagonal_term)
    hess_phasex_phasey += np.diag(hess_phasex_phasey_diagonal_term)

    return (
        hess_amp_amp,
        hess_amp_phasex,
        hess_amp_phasey,
        hess_phasex_phasex,
        hess_phasey_phasey,
        hess_phasex_phasey,
    )


def reformat_to_matrix(
    input_array,
    ant1_inds,
    ant2_inds,
    Nants,
    Nbls,
):
    """
    Reformat an array indexed in baselines into a matrix with antenna indices.

    Parameters
    ----------
    input_array : array of float or complex
        Shape (Nbls, ...,).
    ant1_inds : array of int
        Shape (Nbls,).
    ant2_inds : array of int
        Shape (Nbls,).
    Nants : int
        Number of antennas.
    Nbls : int
        Number of baselines.
    Ntimes : int
        Number of obs times

    Returns
    -------
    antenna matrix : array of float or complex
        Shape (Nants, Nbls, ...,). Same dtype as input_array.
    """

    rect_matrix = np.zeros_like(
        input_array[0,],
        dtype=input_array.dtype,
    )
    rect_matrix = np.repeat(
        np.repeat(rect_matrix[np.newaxis,], Nants, axis=0)[np.newaxis,],
        Nbls,
        axis=0,
    )
    # what does this do?
    for bl_ind in range(Nbls):
        rect_matrix[
            ant1_inds[bl_ind],
            ant2_inds[bl_ind],
        ] = input_array[
            bl_ind,
        ]
    return rect_matrix

def cost_unical(
    gains         : np.ndarray[complex],
    fit_vis       : np.ndarray[complex],
    data_vis      : np.ndarray[complex],
    model_vis     : np.ndarray[complex],
    vis_weights   : np.ndarray[float],
    model_weights : np.ndarray[float],
    ant1_inds     : np.ndarray[int],
    ant2_inds     : np.ndarray[int],
    lambda_val    : float,
    force_skycal  : bool = False,
    gmm           : bool = True,
) -> float:
    """
    Calculate the cost function (chi-squared) value.
    Friendly to Scipy (real variables) optimization.

    Parameters
    ----------
    gains : array of complex
        Shape (Nants,).
    fit_vis : array of complex
        Shape (Nbls,).
    model_vis :  array of complex
        Shape (Ntimes, Nbls,).
    data_vis: array of complex
        Shape (Ntimes, Nbls,).
    vis_weights : array of float
        Shape (Ntimes, Nbls,).
    model_weights : array of float
        Shape (Ntimes, Nbls,).
    ant1_inds : array of int
        Shape (Nbls,).
    ant2_inds : array of int
        Shape (Nbls,).
    lambda_val : float
        Weight of the phase regularization term; must be positive.

    Returns
    -------
    cost : float
        Value of the cost function.
    """
    gains_expanded = (gains[ant1_inds] * np.conj(gains[ant2_inds]))[np.newaxis, :]
    if not force_skycal:
        if gmm:
            res_vec_1 = data_vis - gains_expanded * fit_vis
        else:
            res_vec_1 = fit_vis - gains_expanded * data_vis
        res_vec_2 = fit_vis - model_vis
        cost = np.sum(vis_weights * np.abs(res_vec_1) ** 2) + np.sum(model_weights * np.abs(res_vec_2)**2)
    else:
        res_vec_1 = data_vis - gains_expanded * model_vis
        cost = np.sum(vis_weights * np.abs(res_vec_1) ** 2)
    if lambda_val > 0:
        regularization_term = lambda_val * np.sum(np.angle(gains)) ** 2.0
        cost += regularization_term

    return cost

def cost_unical_torch(
    params        : torch.Tensor,
    data_vis      : torch.Tensor,
    model_vis     : torch.Tensor,
    vis_weights   : torch.Tensor,
    model_weights : torch.Tensor,
    ant_inds      : torch.Tensor,
    ant1_inds     : torch.Tensor,
    ant2_inds     : torch.Tensor,
    num_ants      : int,
    lambda_val    : float,
) -> float:
    """
    Calculate the cost function (chi-squared) value.
    Friendly to PyTorch (complex variables) optimization.

    Parameters
    ----------
    params : tensor of complex
        Shape (Nants + Nbls,).
    model_vis :  tensor of complex
        Shape (Ntimes, Nbls,).
    data_vis: tensor of complex
        Shape (Ntimes, Nbls,).
    vis_weights : tensor of float
        Shape (Ntimes, Nbls,).
    model_weights : tensor of float
        Shape (Ntimes, Nbls,).
    ant_inds : tensor of int
        Shape (Nants_unflagged,).
    ant1_inds : tensor of int
        Shape (Nbls,).
    ant2_inds : tensor of int
        Shape (Nbls,).
    num_ants : int
        number of total antennas, flagged or unflagged
    lambda_val : float
        Weight of the phase regularization term; must be positive.

    Returns
    -------
    cost : float
        Value of the cost function.
    """
    # gains_reshaped = params[:len(ant_inds)]
    # gains = torch.ones((num_ants), dtype=torch.complex64)
    # gains[ant_inds] = gains_reshaped
    gains = params[:len(ant_inds)]
    fit_vis = params[len(ant_inds):]

    gains_expanded = (gains[ant1_inds] * torch.conj((gains[ant2_inds])))[None, :]
    res_vec_1 = data_vis - gains_expanded * fit_vis
    res_vec_2 = fit_vis - model_vis
    cost_1 = vis_weights * (res_vec_1.real ** 2 + res_vec_1.imag ** 2)
    cost_2 = model_weights * (res_vec_2.real ** 2 + res_vec_2.imag ** 2)
    cost = torch.sum(cost_1) + torch.sum(cost_2)

    if lambda_val > 0:
<<<<<<< HEAD
        regularization_term = lambda_val * np.sum(np.angle(gains)) ** 2.0
=======
        regularization_term = lambda_val * torch.sum(torch.angle(gains)) ** 2.0
>>>>>>> unical2
        cost += regularization_term

    return cost

def jacobian_unical(
    gains         : np.ndarray[complex],
    fit_vis       : np.ndarray[complex],
    data_vis      : np.ndarray[complex],
    model_vis     : np.ndarray[complex],
    vis_weights   : np.ndarray[float],
    model_weights : np.ndarray[float],
    ant1_inds     : np.ndarray[int],
    ant2_inds     : np.ndarray[int],
    lambda_val    : float,
) -> np.ndarray[complex]:
    """
    Calculate the Jacobian of the cost function.

    Parameters
    ----------
    gains : array of complex
        Shape (Nants,).
    fit_vis : array of complex
        Shape (Ntimes, Nbls,).
    model_vis : array of complex
        Shape (Ntimes, Nbls,).
    data_vis : array of complex
        Shape (Ntimes, Nbls,).
    u_params : array of complex
        Shape (Ntimes, Nbls,).
    vis_weights : array of float
        Shape (Ntimes, Nbls,).
    model_weights : array of float
        Shape (Ntimes, Nbls,).
    ant1_inds : array of int
        Shape (Nbls,).
    ant2_inds : array of int
        Shape (Nbls,).
    lambda_val : float
        Weight of the phase regularization term; must be positive.

    Returns
    -------
    jac : array of complex
        Jacobian of the chi-squared cost function, shape (Nants,). The real part
        corresponds to derivatives with respect to the real part of the gains;
        the imaginary part corresponds to derivatives with respect to the
        imaginary part of the gains.
    """

    start_jac = time.time()

    # Convert gains to row vector with unit
    # dimension extended along the time axis
    gains_exp_1 = gains[np.newaxis, ant1_inds]                   # shape (1,Nbls)
    gains_exp_2 = gains[np.newaxis, ant2_inds]                   # shape (1,Nbls)

    # calculate real terms in baseline space
    gains_term1_bls = np.mean(                                   # shape (Nbls)
        2 * vis_weights * (gains_exp_1 * np.abs(gains_exp_2)**2.0 * np.abs(fit_vis)**2.0 - (
            data_vis * gains_exp_2 * np.conj(fit_vis))
        ),
        axis=0,
    )
    # convert to antenna space
    gains_term1_ants = utils.bincount_multidim(                  # shape (Nants)
        ant1_inds,
        weights=gains_term1_bls,
        minlength=np.max([np.max(ant1_inds), np.max(ant2_inds)]) + 1,
    )
    # must add antenna permutations
    gains_term2_bls = np.mean(                                   # shape (Nbls)
        2 * vis_weights * (gains_exp_2 * np.abs(gains_exp_1)**2.0 * np.abs(fit_vis)**2.0 - (
            np.conj(data_vis) * gains_exp_1 * fit_vis)
        ),
        axis=0,
    )
    # convert to antenna space
    gains_term2_ants = utils.bincount_multidim(                  # shape (Nants)
        ant2_inds,
        weights=gains_term2_bls,
        minlength=np.max([np.max(ant1_inds), np.max(ant2_inds)]) + 1,
    )
    jac_gains = gains_term1_ants + gains_term2_ants;             # shape (Nants)

    # model vis params terms
    vis_term = np.mean(                                    # shape (Nbls)
        2 * vis_weights * (np.abs(gains_exp_1)**2.0 * np.abs(gains_exp_2)**2.0 * fit_vis
                           - gains_exp_1 * np.conj(gains_exp_2) * data_vis),
        axis=0,
    )
    model_term = np.mean(                                    # shape (Nbls)
        2 * model_weights * (fit_vis - model_vis),
        axis=0,
    )

    jac_vis = vis_term + model_term    # shape (Nbls)

    jac = np.hstack((jac_gains, jac_vis))

    print(f"Jacobian time - {(time.time()-start_jac)/60} minutes")

    return jac


def hessian_unical(
    gains         : np.ndarray[complex],
    fit_vis       : np.ndarray[complex],
    Nants         : int,
    Nbls          : int,
    Ntimes        : int,
    data_vis      : np.ndarray[complex],
    model_vis     : np.ndarray[complex],
    vis_weights   : np.ndarray[float],
    model_weights : np.ndarray[float],
    ant1_inds     : np.ndarray[int],
    ant2_inds     : np.ndarray[int],
    bl_inds       : np.ndarray[int],
    lambda_val    : float,
) -> tuple[np.ndarray[float], ...]:
    """
    Calculate the Hessian of the cost function.

    Parameters
    ----------
    gains : array of complex
        Shape (Nants,).
    fit_vis : array of complex
        Shape (Nbls,).
    Nants : int
        Number of antennas.
    Nbls : int
        Number of baselines.
    model_vis : array of complex
        Shape (Ntimes, Nbls,).
    data_vis : array of complex
        Shape (Ntimes, Nbls,).
    vis_weights : array of float
        Shape (Ntimes, Nbls,).
    ant1_inds : array of int
        Shape (Nbls,).
    ant2_inds : array of int
        Shape (Nbls,).
    lambda_val : float
        Weight of the phase regularization term; must be positive.

    Returns
    -------
    hess_real_real : array of float
        Real-real derivative components of the Hessian of the cost function.
        Shape (Nants, Nants,).
    hess_real_imag : array of float
        Real-imaginary derivative components of the Hessian of the cost
        function. Note that the transpose of this array gives the imaginary-real
        derivative components. Shape (Nants, Nants,).
    hess_imag_imag : array of float
        Imaginary-imaginary derivative components of the Hessian of the cost
        function. Shape (Nants, Nants,).
    """

    start_hess = time.time()

    gains_exp_1 = gains[ant1_inds]                                      # shape (Nbls)
    gains_exp_2 = gains[ant2_inds]                                      # shape (Nbls)
    fit_squared = np.mean(
        vis_weights * np.abs(fit_vis) ** 2.0,   
        axis=0,
    )                                                      
    data_times_u = np.mean(
        vis_weights * np.conj(data_vis) * fit_vis,
        axis=0,
    )

    """Gains params only"""
    # Calculate the antenna off-diagonal components
    gain_hess_components = np.zeros((Nbls, 4), dtype=float)             # shape (Nbls, 4)
    # Real-real Hessian component for gains:
    gain_hess_components[:, 0] = (
        4 * gains_exp_1.real * gains_exp_2.real * fit_squared
        - 2 * data_times_u.real
    ).real
    # Real-imaginary Hessian component for gains, term 1:
    gain_hess_components[:, 1] = (
        4 * gains_exp_1.real * gains_exp_2.imag * fit_squared
        + 2 * data_times_u.imag
    ).real
    # Real-imaginary Hessian component for gains, term 2:
    gain_hess_components[:, 2] = (
        4 * gains_exp_1.imag * gains_exp_2.real * fit_squared
        - 2 * data_times_u.imag
    ).real
    # Imaginary-imaginary Hessian component for gains:
    gain_hess_components[:, 3] = (
        4 * gains_exp_1.imag * gains_exp_2.imag * fit_squared
        - 2 * data_times_u.real
    ).real

    gain_hess_components = reformat_baselines_to_antenna_matrix(
        gain_hess_components,
        ant1_inds,
        ant2_inds,
        Nants,
        Nbls,
    )
    gain_hess_real_real = gain_hess_components[:, :, 0] + gain_hess_components[:, :, 0].T
    gain_hess_real_imag = gain_hess_components[:, :, 1] + gain_hess_components[:, :, 2].T
    gain_hess_imag_imag = gain_hess_components[:, :, 3] + gain_hess_components[:, :, 3].T

    # Calculate the antenna diagonals
    gain_hess_diag = 2 * (
        utils.bincount_multidim(
            ant1_inds,
            weights = 2 * np.abs(gains_exp_2) ** 2.0 * fit_squared,
            minlength=Nants,
        )
        + utils.bincount_multidim(
            ant2_inds,
            weights= 2 * np.abs(gains_exp_1) ** 2.0 * fit_squared,
            minlength=Nants,
        )
    )
    np.fill_diagonal(gain_hess_real_real, gain_hess_diag)
    np.fill_diagonal(gain_hess_imag_imag, gain_hess_diag)
    np.fill_diagonal(gain_hess_real_imag, 0.0)

    if lambda_val > 0:  # Add regularization term
        gains_weighted = gains / np.abs(gains) ** 2.0
        arg_sum = np.sum(np.angle(gains))
        # Antenna off-diagonals
        gain_hess_real_real += (
            2 * lambda_val * np.outer(gains_weighted.imag, gains_weighted.imag)
        )
        gain_hess_real_imag -= (
            2 * lambda_val * np.outer(gains_weighted.imag, gains_weighted.real)
        )
        gain_hess_imag_imag += (
            2 * lambda_val * np.outer(gains_weighted.real, gains_weighted.real)
        )
        # Antenna diagonals
        gain_hess_real_real += np.diag(
            4 * lambda_val * arg_sum * gains_weighted.imag * gains_weighted.real
        )
        gain_hess_real_imag -= np.diag(
            2
            * lambda_val
            * arg_sum
            * (gains_weighted.real ** 2.0 - gains_weighted.imag ** 2.0)
        )
        gain_hess_imag_imag -= np.diag(
            4 * lambda_val * arg_sum * gains_weighted.imag * gains_weighted.real
        )

    """Fit vis params only"""
    # Fill the fitted visibility only matrix with zeros; all off-diagnoal
    # entries will remain zero (NOTE: Should be Nbls not 2*NBls yes?)
    fit_hess_real_real = np.zeros((Nbls, Nbls), dtype=float)
    fit_hess_imag_imag = np.zeros((Nbls, Nbls), dtype=float)
    fit_hess_real_imag = np.zeros((Nbls, Nbls), dtype=float)

    # Calculate the fitted visibilities diagonals
    fit_hess_diag = (
        2 * vis_weights * np.abs(gains_exp_1) ** 2.0 * np.abs(gains_exp_2)**2.0
         + model_weights
    )
    np.fill_diagonal(fit_hess_real_real, fit_hess_diag)
    np.fill_diagonal(fit_hess_imag_imag, fit_hess_diag)

    """Fit vis/gains mix"""
    # Calculate the antenna off-diagonal components 
    # for both antennas in baseline
    fit_gain_hess_vectors = np.zeros((Nbls, 4), dtype=float)               # shape: (Nbls, 4)
    fit_gain_hess_components = np.zeros((Nbls, Nants, 4), dtype=float)     # shape: (Nbls, Nants, 4)

    """Fit vis/antenna 1 gains Hessian components"""
    # Real-real
    fit_gain_hess_vectors[:, 0] = (
        4 * np.mean(vis_weights, axis=0) * fit_vis.real * gains_exp_1.real * np.abs(gains_exp_2)**2.0
        - 2 * (np.mean(vis_weights * np.conj(data_vis), axis=0) * np.conj(gains_exp_2)).real
        )
    # Real gain, imaginary fit vis:
    fit_gain_hess_vectors[:, 1] = (
        4 * np.mean(vis_weights, axis=0) * fit_vis.imag * gains_exp_1.real * np.abs(gains_exp_2)**2.0
        + 2 * (np.mean(vis_weights * np.conj(data_vis), axis=0) * np.conj(gains_exp_2)).imag
        )
    # Imaginary gain, real fit vis:
    fit_gain_hess_vectors[:, 2] = (
        4 * np.mean(vis_weights, axis=0) * fit_vis.real * gains_exp_1.imag * np.abs(gains_exp_2)**2.0
        + 2 * (np.mean(vis_weights * np.conj(data_vis), axis=0) * np.conj(gains_exp_2)).imag
        )
    # Imaginary-imaginary
    fit_gain_hess_vectors[:, 3] = (
        4 * np.mean(vis_weights, axis=0) * fit_vis.imag * gains_exp_1.imag * np.abs(gains_exp_2)**2.0
        + 2 * (np.mean(vis_weights * np.conj(data_vis), axis=0) * np.conj(gains_exp_2)).real
        )

    # # Update hessian block with second derivatives w.r.t. antenna 1 gains
    # for bl_ind in range(Nbls):
    #     fit_gain_hess_components[
    #         bl_ind,
    #         ant1_inds[bl_ind],
    #     ] = fit_gain_hess_vectors[bl_ind, :]

    """U params/antenna 2 gains Hessian components"""
    # Real-real
    fit_gain_hess_vectors[:, 0] += (
        4 * np.mean(vis_weights, axis=0) * fit_vis.real * gains_exp_2.real * np.abs(gains_exp_1)**2.0
        - 2 * (np.mean(vis_weights * np.conj(data_vis), axis=0) * np.conj(gains_exp_1)).real
        )
    # Real gain, imaginary fit vis:
    fit_gain_hess_vectors[:, 1] += (
        4 * np.mean(vis_weights, axis=0) * fit_vis.imag * gains_exp_2.real * np.abs(gains_exp_1)**2.0
        + 2 * (np.mean(vis_weights * np.conj(data_vis), axis=0) * np.conj(gains_exp_1)).imag
        )
    # Imaginary gain, real fit vis:
    fit_gain_hess_vectors[:, 2] += (
        4 * np.mean(vis_weights, axis=0) * fit_vis.real * gains_exp_2.imag * np.abs(gains_exp_1)**2.0
        + 2 * (np.mean(vis_weights * np.conj(data_vis), axis=0) * np.conj(gains_exp_1)).imag
        )
    # Imaginary-imaginary
    fit_gain_hess_vectors[:, 3] += (
        4 * np.mean(vis_weights, axis=0) * fit_vis.imag * gains_exp_2.imag * np.abs(gains_exp_1)**2.0
        + 2 * (np.mean(vis_weights * np.conj(data_vis), axis=0) * np.conj(gains_exp_1)).real
        )

    # Update hessian block with second derivatives w.r.t. antenna 1 and 2 gains
    for bl_ind in range(Nbls):
        fit_gain_hess_components[
            bl_ind,
            ant2_inds[bl_ind],
        ] = fit_gain_hess_vectors[bl_ind,:]

    fit_gain_hess_realu_realg = fit_gain_hess_components[:, :, 0]
    fit_gain_hess_imagu_realg = fit_gain_hess_components[:, :, 1]
    fit_gain_hess_realu_imagg = fit_gain_hess_components[:, :, 2]
    fit_gain_hess_imagu_imagg = fit_gain_hess_components[:, :, 3]

    print(f"Hessian time - {(time.time()-start_hess)/60} minutes")

    return gain_hess_real_real, gain_hess_real_imag, gain_hess_imag_imag, \
        fit_hess_real_real, fit_hess_real_imag, fit_hess_imag_imag, \
        fit_gain_hess_realu_realg, fit_gain_hess_imagu_realg, fit_gain_hess_realu_imagg, fit_gain_hess_imagu_imagg