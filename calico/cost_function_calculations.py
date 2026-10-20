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
    print("***JACOBIAN TIME***", (end_jac - start_jac) / 60.0)

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
        ] = bl_array[bl_ind,]
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
    print("***HESSIAN TIME***", (end_hess - start_hess) / 60.0)

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
    ] * np.conj(gains_expanded_2[:, :, 1])  # Apply gains to PQ visibilities
    crosspol_data_visibilities_calibrated[:, :, 1] *= gains_expanded_1[
        :, :, 1
    ] * np.conj(gains_expanded_2[:, :, 0])  # Apply gains to QP visibilities
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
        ] = input_array[bl_ind,]
    return rect_matrix


def cost_unical(
    gains: np.ndarray[complex],
    fit_vis: np.ndarray[complex],
    data_vis: np.ndarray[complex],
    model_vis: np.ndarray[complex],
    vis_weights: np.ndarray[float],
    model_weights: np.ndarray[float],
    ant1_inds: np.ndarray[int],
    ant2_inds: np.ndarray[int],
    lambda_val: float,
    force_skycal: bool = False,
    gmm: bool = True,
) -> float:
    """
    Calculate the cost function (chi-squared) value.
    Friendly to Scipy (real variables) optimization.

    Parameters
    ----------
    gains : array of complex
        Shape (Nants,).
    fit_vis : array of complex
        Shape (Ntimes, Nbls,).
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
        cost = np.sum(vis_weights * np.abs(res_vec_1) ** 2) + np.sum(
            model_weights * np.abs(res_vec_2) ** 2
        )
    else:
        res_vec_1 = data_vis - gains_expanded * model_vis
        cost = np.sum(vis_weights * np.abs(res_vec_1) ** 2)
    if lambda_val > 0:
        regularization_term = lambda_val * np.sum(np.angle(gains)) ** 2.0
        cost += regularization_term

    return cost


def cost_unical_torch(
    gains: torch.Tensor,
    fit_vis: torch.Tensor,
    data_vis: torch.Tensor,
    model_vis: torch.Tensor,
    vis_weights: torch.Tensor,
    model_weights: torch.Tensor,
    ant_inds: torch.Tensor,
    ant1_inds: torch.Tensor,
    ant2_inds: torch.Tensor,
    num_ants: int,
    lambda_val: float,
) -> float:
    """
    Calculate the cost function (chi-squared) value.
    Friendly to PyTorch (complex variables) optimization.

    Parameters
    ----------
    gains : tensor of complex
        Shape (Nants,).
    fit_vis : tensor of complex
        Shape (Ntimes, Nbls,).
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

    gains_expanded = (gains[ant1_inds] * torch.conj((gains[ant2_inds])))[None, :]
    res_vec_1 = data_vis - gains_expanded * fit_vis
    res_vec_2 = fit_vis - model_vis
    cost_1 = vis_weights * (res_vec_1.real**2 + res_vec_1.imag**2)
    cost_2 = model_weights * (res_vec_2.real**2 + res_vec_2.imag**2)
    cost = torch.sum(cost_1) + torch.sum(cost_2)

    if lambda_val > 0:
        regularization_term = lambda_val * torch.sum(torch.angle(gains)) ** 2.0
        cost += regularization_term

    return cost


def _unical_local_derivatives(
    gains, fit_vis, data_vis, vis_weights, ant1_inds, ant2_inds, ant_inds
):
    """
    Per-visibility pieces shared by jacobian_unical and hessian_unical.

    Each visibility (time t, baseline b) has residual
    r = d - g1 * conj(g2) * u and depends on six real parameters, ordered
    [Re g1, Im g1, Re g2, Im g2, Re u, Im u]. Returns the residuals, the first
    and second derivatives of r with respect to those parameters, and each
    parameter's position in the flattened parameter vector (-1 for gains of
    flagged antennas, which are held fixed).
    """
    n_times, n_bls = fit_vis.shape
    n_ants_unflagged = len(ant_inds)
    g1 = np.broadcast_to(gains[ant1_inds], (n_times, n_bls)).ravel()
    g2c = np.broadcast_to(np.conj(gains[ant2_inds]), (n_times, n_bls)).ravel()
    u = fit_vis.ravel()
    w = vis_weights.ravel()
    resid = data_vis.ravel() - g1 * g2c * u

    # dr/dp, shape (Ntimes*Nbls, 6)
    first = np.stack(
        (
            -g2c * u,
            -1j * g2c * u,
            -g1 * u,
            1j * g1 * u,
            -g1 * g2c,
            -1j * g1 * g2c,
        ),
        axis=1,
    )
    # d2r/dp_i dp_j, shape (Ntimes*Nbls, 6, 6); r is bilinear in (g1, g2, u),
    # so only the cross blocks between different parameters are nonzero
    # each block is [[d/dRe d/dRe, d/dRe d/dIm], [d/dIm d/dRe, d/dIm d/dIm]]
    second = np.zeros((len(u), 6, 6), dtype=complex)
    second[:, 0:2, 2:4] = np.moveaxis(
        np.array([[-u, 1j * u], [-1j * u, -u]]), -1, 0
    )  # g1, g2
    second[:, 0:2, 4:6] = np.moveaxis(
        np.array([[-g2c, -1j * g2c], [-1j * g2c, g2c]]), -1, 0
    )  # g1, u
    second[:, 2:4, 4:6] = np.moveaxis(
        np.array([[-g1, -1j * g1], [1j * g1, -g1]]), -1, 0
    )  # g2, u
    second += second.transpose(0, 2, 1)

    # positions in the flattened parameter vector
    ant_pos = np.full(len(gains), -1)
    ant_pos[ant_inds] = 2 * np.arange(n_ants_unflagged)
    pos1 = np.broadcast_to(ant_pos[ant1_inds], (n_times, n_bls)).ravel()
    pos2 = np.broadcast_to(ant_pos[ant2_inds], (n_times, n_bls)).ravel()
    pos_u = 2 * n_ants_unflagged + 2 * np.arange(n_times * n_bls)
    positions = np.stack((pos1, pos1 + 1, pos2, pos2 + 1, pos_u, pos_u + 1), axis=1)
    positions[:, 0:2][pos1 < 0] = -1
    positions[:, 2:4][pos2 < 0] = -1

    return resid, w, first, second, positions


def _regularization_angle_derivatives(gains, ant_inds):
    """
    First and second derivatives of angle(g) with respect to (Re g, Im g)
    for the unflagged antennas.
    """
    g = gains[ant_inds]
    x, y = g.real, g.imag
    rho2 = x**2 + y**2
    d_angle = np.stack((-y / rho2, x / rho2), axis=1)  # (Nants_unflagged, 2)
    d2_angle = np.stack(
        (
            np.stack((2 * x * y, y**2 - x**2), axis=-1),
            np.stack((y**2 - x**2, -2 * x * y), axis=-1),
        ),
        axis=1,
    ) / (rho2**2)[:, np.newaxis, np.newaxis]  # (Nants_unflagged, 2, 2)
    return d_angle, d2_angle


def jacobian_unical(
    gains: np.ndarray[complex],
    fit_vis: np.ndarray[complex],
    data_vis: np.ndarray[complex],
    model_vis: np.ndarray[complex],
    vis_weights: np.ndarray[float],
    model_weights: np.ndarray[float],
    ant1_inds: np.ndarray[int],
    ant2_inds: np.ndarray[int],
    ant_inds: np.ndarray[int],
    lambda_val: float,
) -> np.ndarray[float]:
    """
    Calculate the Jacobian (gradient) of the unified calibration cost function
    cost_unical (with gains multiplying the fitted visibilities).

    Parameters
    ----------
    gains : array of complex
        Shape (Nants,).
    fit_vis : array of complex
        Shape (Ntimes, Nbls,).
    data_vis : array of complex
        Shape (Ntimes, Nbls,).
    model_vis : array of complex
        Shape (Ntimes, Nbls,).
    vis_weights : array of float
        Shape (Ntimes, Nbls,).
    model_weights : array of float
        Shape (Ntimes, Nbls,).
    ant1_inds : array of int
        Shape (Nbls,).
    ant2_inds : array of int
        Shape (Nbls,).
    ant_inds : array of int
        Shape (Nants_unflagged,). Antennas whose gains are fit parameters.
    lambda_val : float
        Weight of the phase regularization term; must be positive.

    Returns
    -------
    jac : array of float
        Shape (2*Nants_unflagged + 2*Ntimes*Nbls,), in the same order as the
        flattened parameters: real and imaginary parts of each unflagged gain,
        then of each fitted visibility in (time, baseline) order.
    """
    n_params = 2 * len(ant_inds) + 2 * fit_vis.size
    resid, w, first, _, positions = _unical_local_derivatives(
        gains, fit_vis, data_vis, vis_weights, ant1_inds, ant2_inds, ant_inds
    )
    # d(w|r|^2)/dp = 2 w Re(conj(r) dr/dp)
    local_grad = 2 * w[:, np.newaxis] * np.real(np.conj(resid)[:, np.newaxis] * first)
    valid = positions >= 0
    jac = np.zeros(n_params, dtype=float)
    np.add.at(jac, positions[valid], local_grad[valid])

    # model term: m |u - model|^2
    model_resid = (fit_vis - model_vis).ravel()
    m = model_weights.ravel()
    jac[2 * len(ant_inds) :: 2] += 2 * m * model_resid.real
    jac[2 * len(ant_inds) + 1 :: 2] += 2 * m * model_resid.imag

    # regularization: lambda * (sum of gain phases)^2
    if lambda_val > 0:
        angle_sum = np.sum(np.angle(gains))
        d_angle, _ = _regularization_angle_derivatives(gains, ant_inds)
        jac[: 2 * len(ant_inds)] += 2 * lambda_val * angle_sum * d_angle.ravel()

    return jac


def hessian_unical(
    gains: np.ndarray[complex],
    fit_vis: np.ndarray[complex],
    data_vis: np.ndarray[complex],
    model_vis: np.ndarray[complex],
    vis_weights: np.ndarray[float],
    model_weights: np.ndarray[float],
    ant1_inds: np.ndarray[int],
    ant2_inds: np.ndarray[int],
    ant_inds: np.ndarray[int],
    lambda_val: float,
) -> np.ndarray[float]:
    """
    Calculate the dense Hessian of the unified calibration cost function
    cost_unical (with gains multiplying the fitted visibilities).

    Parameters are as for jacobian_unical. model_vis is unused (the model
    term's second derivative doesn't depend on it) but kept so the two
    functions take the same arguments.

    Returns
    -------
    hess : array of float
        Shape (N, N,) with N = 2*Nants_unflagged + 2*Ntimes*Nbls, in the same
        parameter order as jacobian_unical. Memory grows as N**2.
    """
    n_params = 2 * len(ant_inds) + 2 * fit_vis.size
    resid, w, first, second, positions = _unical_local_derivatives(
        gains, fit_vis, data_vis, vis_weights, ant1_inds, ant2_inds, ant_inds
    )
    # d2(w|r|^2)/dp_i dp_j = 2 w Re(dr/dp_i conj(dr/dp_j) + conj(r) d2r/dp_i dp_j)
    local_hess = (
        2
        * w[:, np.newaxis, np.newaxis]
        * np.real(
            first[:, :, np.newaxis] * np.conj(first[:, np.newaxis, :])
            + np.conj(resid)[:, np.newaxis, np.newaxis] * second
        )
    )
    # parameters of flagged antennas are fixed: drop their rows and columns
    valid = positions >= 0
    local_hess[~(valid[:, :, np.newaxis] & valid[:, np.newaxis, :])] = 0.0
    safe_positions = np.where(valid, positions, 0)
    hess = np.zeros((n_params, n_params), dtype=float)
    np.add.at(
        hess,
        (safe_positions[:, :, np.newaxis], safe_positions[:, np.newaxis, :]),
        local_hess,
    )

    # model term: m |u - model|^2
    u_diag = 2 * len(ant_inds) + np.arange(2 * fit_vis.size)
    hess[u_diag, u_diag] += 2 * np.repeat(model_weights.ravel(), 2)

    # regularization: lambda * S^2 with S the sum of gain phases
    if lambda_val > 0:
        angle_sum = np.sum(np.angle(gains))
        d_angle, d2_angle = _regularization_angle_derivatives(gains, ant_inds)
        d_angle_flat = d_angle.ravel()
        n_gain_params = 2 * len(ant_inds)
        hess[:n_gain_params, :n_gain_params] += (
            2 * lambda_val * np.outer(d_angle_flat, d_angle_flat)
        )
        for k in range(len(ant_inds)):
            hess[2 * k : 2 * k + 2, 2 * k : 2 * k + 2] += (
                2 * lambda_val * angle_sum * d2_angle[k]
            )

    return hess
