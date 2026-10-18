import numpy as np

from calico.dev import dev_tools as dev, gain_error_offset_analysis as grid_search

filenames = ["data_1", "data_14", "data_28", "data_42", "tutorial_medium"]

e_vals = None
n_vals = None
truth_grids = []
skycal_grids = []

for filename in filenames:
    e_vals, n_vals, g_offsets_truth, g_offsets_skycal = grid_search.main(
        calibrate=True,
        optim_type="pytorch",
        gains_multiply_model=True,
        data_name=filename,
        simulate_visibilities=True,
        same_sky_all_times=False,
    )
    truth_grids.append(g_offsets_truth)
    skycal_grids.append(g_offsets_skycal)

avg_truth_grids = np.mean(np.asarray(truth_grids))
avg_skycal_grids = np.mean(np.asarray(skycal_grids))

dev.plot_3d_data_as_2d_hist(
    x_array=e_vals,
    y_array=n_vals,
    z_array=avg_truth_grids,
    num_y_vals=len(e_vals),
    num_x_vals=len(n_vals),
    plot_title="Average of Gain Offsets over Different Sky Realizations (Truth)",
    plot_xlabel="$Re(v_T - m)$",
    plot_ylabel="$\\sigma_t (Re)$",
    plot_vmax=min([np.abs(max(avg_truth_grids)), np.abs(min(avg_truth_grids))]),
    plot_vmin=min([np.abs(max(avg_truth_grids)), np.abs(min(avg_truth_grids))]),
    plot_xlim_h=max(e_vals),
    plot_xlim_l=min(e_vals),
    plot_ylim_h=max(n_vals),
    plot_ylim_l=min(n_vals),
    filename="avg_grid_search_truth.png",
    plot_cmap="viridis",
    cmap_label="Re(g-1)",
    box_text="Optimizer: pytorch",
)

dev.plot_3d_data_as_2d_hist(
    x_array=e_vals,
    y_array=n_vals,
    z_array=avg_skycal_grids,
    num_y_vals=len(e_vals),
    num_x_vals=len(n_vals),
    plot_title="Average of Gain Offsets over Different Sky Realizations (Skycal)",
    plot_xlabel="$Re(v_T - m)$",
    plot_ylabel="$\\sigma_t (Re)$",
    plot_vmax=min([np.abs(max(avg_skycal_grids)), np.abs(min(avg_skycal_grids))]),
    plot_vmin=min([np.abs(max(avg_skycal_grids)), np.abs(min(avg_skycal_grids))]),
    plot_xlim_h=max(e_vals),
    plot_xlim_l=min(e_vals),
    plot_ylim_h=max(n_vals),
    plot_ylim_l=min(n_vals),
    filename="avg_grid_search_skycal.png",
    plot_cmap="viridis",
    cmap_label="Re(g-1)",
    box_text="Optimizer: pytorch",
)
