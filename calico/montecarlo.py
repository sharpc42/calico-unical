import numpy as np
import matplotlib.pyplot as plt
import time

n_samples = 1e7
abs_avg_sum_vals = []  # |<|v|^2 + n*e + vn* + v*e>|
sum_abs_mag_vals = []  # |<|v|^2>| + |<n*e>| + |<vn*>| + |<v*e>|
subtract_off_v   = []  # |<v^2 + n*e + vn* + v*e>| - |<|v|^2>|
n_and_e_terms    = []  # |<n*e + v*e + vn*>|
avg_sum          = []  # <|v|^2 + n*e + v* + v*e>
avg_v_squared    = []  # <|v|^2>
avg_ne_arr       = []  # |<n*e>|
avg_vn_arr       = []  # |<vn*>|
avg_ve_arr       = []  # |<v*e>|
e_arr            = []

# roll data vis once
org_true_data_visibilities = np.random.normal(
    loc=0,
    scale=14,
    size=(1,6,1,1),
) + 1.0j * np.random.normal(
    loc=0,
    scale=14,
    size=(1,6,1,1),
)

for i in range(int(n_samples)):
    np.random.seed(i)
    model_error = np.random.normal(
        loc=0, 
        scale=5,
        size=(1,6,1,1),
    ) + 1.0j * np.random.normal(
        loc=0,
        scale=5,
        size=(1,6,1,1),
    )
    thermal_noise = np.random.normal(
        loc=0,
        scale=5,
        size=(1,6,1,1),
    ) + 1.0j * np.random.normal(
        loc=0,
        scale=5,
        size=(1,6,1,1),
    )
    # quantities
    ne = np.conj(thermal_noise) * model_error
    vn = org_true_data_visibilities * np.conj(thermal_noise)
    ve = np.conj(org_true_data_visibilities) * model_error
    abs_v_squared = np.abs(org_true_data_visibilities)**2
    avg_ne = np.mean(ne)
    avg_vn = np.mean(vn)
    avg_ve = np.mean(ve)
    avg_abs_v_squared = np.mean(abs_v_squared)
    # expressions
    abs_avg_sum = np.abs(
        np.mean(abs_v_squared + ne + ve + vn)
    )
    sum_abs_mag = np.abs(avg_abs_v_squared) + np.abs(avg_ne) + \
                  np.abs(avg_vn) + np.abs(avg_ve)
    denominator = avg_abs_v_squared + np.mean(np.abs(model_error)**2)
    # fill arrays
    abs_avg_sum_vals.append(
        abs_avg_sum / avg_abs_v_squared
    )
    sum_abs_mag_vals.append(
        sum_abs_mag / 1
    )
    subtract_off_v.append(
        abs_avg_sum - avg_abs_v_squared
    )
    n_and_e_terms.append(
        np.abs(np.mean(ne + vn + ve))
    )
    avg_sum.append(
        np.mean(avg_abs_v_squared + ve + vn + ne)
    )
    avg_v_squared.append(
        avg_abs_v_squared
    )
    avg_ne_arr.append(
        np.abs(avg_ne)
    )
    avg_ve_arr.append(
        np.abs(avg_ve)
    )
    avg_vn_arr.append(
        np.abs(avg_vn)
    )
    e_arr.append(
        model_error
    )

abs_avg_sum_mean = np.mean(abs_avg_sum_vals)
sum_abs_mag_mean = np.mean(sum_abs_mag_vals)
subtract_off_v_mean = np.mean(subtract_off_v)
n_and_e_terms_mean = np.mean(n_and_e_terms)
avg_sum_mean = np.mean(avg_sum)
avg_ne_mean = np.mean(avg_ne_arr)
avg_ve_mean = np.mean(avg_ve_arr)
avg_vn_mean = np.mean(avg_vn_arr)
avg_v_squared_mean = np.mean(avg_v_squared)
model_error_mean = np.mean(e_arr)

print(f"\n\n***RESULTS***\n{abs_avg_sum_mean=:.3f}\t{sum_abs_mag_mean=:.3f}")
print(f"{subtract_off_v_mean=:.3f}\t{n_and_e_terms_mean=:.3f}")
print(f"{avg_v_squared_mean=:.3f}\t{avg_ne_mean=:.3f}")
print(f"{avg_ve_mean=:.3f}\t{avg_vn_mean=:.3f}")

new_model_error = np.random.normal(
    loc=0,
    scale=5,
    size=(1,6,1,1),
) + 1.0j * np.random.normal(
    loc=0,
    scale=5,
    size=(1,6,1,1),
)
avg_model_error_squared = np.mean(np.abs(new_model_error)**2)
org_model_visibilities = org_true_data_visibilities.copy()

print(f"\n***MODEL VISIBILITIES - CASE m = vT + e***")
new_model_visibilities = org_true_data_visibilities + new_model_error
avg_abs_m_squared = np.mean(np.abs(new_model_visibilities)**2)
avg_abs_vT_squared = np.mean(np.abs(org_true_data_visibilities)**2)
print(f"$<|m|^2>$ {avg_abs_m_squared:.3f}")
print(f"$<|v_T|^2>$ {avg_abs_vT_squared:.3f}\t$<|e|^2>$ {avg_model_error_squared:.3f}")
print(f"$<|v_T|^2> + <|e|^2>$ {avg_abs_vT_squared + avg_model_error_squared:.3f}")
print(f"$<|m|^2> - <|v_T|^2> - <|e|^2>$ {avg_abs_m_squared - avg_abs_vT_squared - avg_model_error_squared:.3f}")
print(f"$<|v_T|^2> - <|m|^2> - <|e|^2>$ {avg_abs_vT_squared - avg_abs_m_squared - avg_model_error_squared:.3f}")

print(f"\n***MODEL VISIBILITIES - CASE vT = m + e***")
new_true_data_visibilities = org_model_visibilities + new_model_error
avg_abs_m_squared = np.mean(np.abs(org_model_visibilities)**2)
avg_abs_vT_squared = np.mean(np.abs(new_true_data_visibilities)**2)
print(f"$<|m|^2>$ {avg_abs_m_squared:.3f}")
print(f"$<|v_T|^2>$ {avg_abs_vT_squared:.3f}\t$<|e|^2>$ {avg_model_error_squared:.3f}")
print(f"$<|v_T|^2> + <|e|^2>$ {avg_abs_vT_squared + avg_model_error_squared:.3f}")
print(f"$<|m|^2> - <|v_T|^2> - <|e|^2>$ {avg_abs_m_squared - avg_abs_vT_squared - avg_model_error_squared:.3f}")
print(f"$<|v_T|^2> - <|m|^2> - <|e|^2>$ {avg_abs_vT_squared - avg_abs_m_squared - avg_model_error_squared:.3f}")

fig, ax = plt.subplots()
plt.hist(abs_avg_sum_vals, bins=50)
plt.title("$\\frac{|< |v_T|^2 + v_T^* e + n^* v_T + n^* e >|}{<|v_T|^2>}$")
plt.xlabel("Realizations")
props = dict(boxstyle='round', color="wheat", alpha=0.7)
plt.text(x=0.85, 
         y=0.95, 
         s=f"Mean: {np.mean(abs_avg_sum_vals):.3f}\nStd: {np.std(abs_avg_sum_vals):.3f}", 
         fontsize=12,
         verticalalignment='top', 
         bbox=props, 
         ha="center", 
         va="top",
         transform=ax.transAxes,
)
# plt.tight_layout()
plt.show()