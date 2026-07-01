import pyuvdata as uv
import caldata
import cost_function_calculations as calcs
import numpy as np

def calibration_wrapper(caldata_obj, freq_ind, delta_real, delta_imag, alpha):
    for feed_pol_ind, feed_pol in enumerate(caldata_obj.feed_polarization_array):
        vis_pol_ind = np.where(caldata_obj.vis_polarization_array == feed_pol)[0]
        caldata_obj.set_ant_inds(freq_ind, feed_pol_ind)
        caldata_obj.set_bl_inds(freq_ind, feed_pol_ind)
        # prep parameters to fit
        gains = np.ones((caldata_obj.Nants), dtype=complex)
        gains[caldata_obj.ant_inds] = caldata_obj.gains[caldata_obj.ant_inds, freq_ind, feed_pol_ind]
        fit_vis = caldata_obj.fit_vis[0, caldata_obj.bl_inds, freq_ind, feed_pol_ind]
        # init data and model
        data_vis = np.reshape(
            caldata_obj.data_visibilities[:, :, freq_ind, vis_pol_ind],
            (caldata_obj.Ntimes, caldata_obj.Nbls),
        )[0,:]
        model_vis = np.reshape(
            caldata_obj.model_visibilities[:, :, freq_ind, vis_pol_ind],
            (caldata_obj.Ntimes, caldata_obj.Nbls),
        )[0,:]
        # init weights
        vis_weights = np.reshape(
            caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind],
            (caldata_obj.Ntimes, caldata_obj.Nbls),
        )[0,:]
        model_weights = np.reshape(
            caldata_obj.model_weights[:, :, freq_ind, vis_pol_ind],
            (caldata_obj.Ntimes, caldata_obj.Nbls),
        )[0,:]
        lambda_val = 100
        
        # iterate through values of cost function to test it
        x = []
        costs = []
        slopes = []
        for i in range(10):
            cost = calcs.cost_unical(
                gains,
                fit_vis,
                data_vis,
                model_vis,
                vis_weights,
                model_weights,
                caldata_obj.ant1_inds,
                caldata_obj.ant2_inds,
                lambda_val,
            )
            slope = calcs.jacobian_unical(
                gains,
                fit_vis,
                data_vis,
                model_vis,
                vis_weights,
                model_weights,
                caldata_obj.ant1_inds,
                caldata_obj.ant2_inds,
                lambda_val,
            )
            curvature = calcs.hessian_unical(
                gains,
                fit_vis,
                data_vis,
                model_vis,
                vis_weights,
                model_weights,
                caldata_obj.ant1_inds,
                caldata_obj.ant2_inds,
                lambda_val,
            )
            print("i",i)
            print("Cost",cost)
            print("Diff", slope)
            
            gains -= alpha * slope * (1.0 + 1.0j)
            x.append(i)
            costs.append(cost)
            slopes.append(slope+1)
    return gains, fit_vis

def test(path):
    # read in data and model
    data = uv.UVData()
    data.read_uvfits(path)
    model = uv.UVData()
    model.read_uvfits(path)
    # Ensure data and model are phased the same
    data.phase_to_time(np.mean(data.time_array))
    model.phase_to_time(np.mean(data.time_array))
    # create caldata obj and load data
    caldata_obj = caldata.CalData()
    caldata_obj.load_data(
        data,
        model,
        gain_init_stddev=0.1,
        fit_vis_init_stddev=0.0,
    )

    # calibration per freq (copy/paste from main code)
    for freq_ind in range(caldata_obj.Nfreqs):
        # before_arr_gains = caldata_obj.gains[:, [freq_ind], :]
        # before_arr_u = caldata_obj.fit_vis[:, :1, [freq_ind], :]
        gains_fit, fit_vis_fit = calibration_wrapper(caldata_obj, freq_ind, delta_imag=0.01, delta_real=0.01, alpha=0.01)
        caldata_obj.gains[:, [freq_ind], :] = gains_fit[:, np.newaxis, np.newaxis]
        caldata_obj.fit_vis[:1, :, [freq_ind], :] = fit_vis_fit[:, np.newaxis, np.newaxis]
        # caldata_obj.temp_test(before_arr_gains, before_arr_u)