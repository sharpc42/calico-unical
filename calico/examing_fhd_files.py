from pyuvdata import UVData
import calibration_wrappers as calwrap
import numpy as np
import matplotlib.pyplot as plt

# get v and m arrays back from unical
data = UVData.from_file('calico/data/fhd_data_one_freq.uvfits')
model = UVData.from_file('calico/data/fhd_model_one_freq_01.uvfits')
v = data.data_array[:data.Nbls,0,0]
m = model.data_array[:model.Nbls,0,0]

if data.Nbls > model.Nbls:
    print("Data larger than model")
    difference = data.Nbls - model.Nbls
    model_mean = np.mean(m)
    for i in range(difference):
        m = np.append(m, model_mean)
elif model.Nbls > data.Nbls:
    print("Model larger than data")
    difference = model.Nbls - data.Nbls
    data_mean = np.mean(v)
    for i in range(difference):
        v = np.append(v, data_mean)

print("\n\n\n*** v-m stats***\n\n")
print(f"Avg|v|\n\t", np.mean(np.abs(v)))
print(f"Avg|m|\n\t", np.mean(np.abs(m)))
print(f"Std(v)\n\t", np.std(v))
print(f"Std(m)\n\t", np.std(m))
print(f"Average |v-m|\n\t{np.mean(np.abs(v-m))}")
print(f"Average |Re(v-m)|\n\t{np.mean(np.abs((v-m).real))}")
print(f"Std Re(v-m)\n\t", np.std(v-m))
print(f"Std Re(v-m))\n\t", np.std((v-m).real))
print(f'Max(v)\n\t', np.max(v.real))
print(f'Min(v)\n\t', np.min(v.real))
print(f'Max(m)\n\t', np.max(m.real))
print(f'Min(m)\n\t', np.min(m.real))
print(f'Max(|v-m|)\n\t', np.max(np.abs(v-m)))
print("\n\n\n")

# plt.hist(v.real, bins=range(-40,40,1), label='|v|', alpha=0.5, histtype='step', color='blue')
# plt.hist(v.real, bins=range(-40,40,1), label='|m|', alpha=0.5, color='orange')
plt.hist(v.real-m.real, bins=range(-40,40,1), label='Re(v)-Re(m)', alpha=0.8, histtype='step', color='green')
plt.hist(v.imag-m.imag, bins=range(-40,40,1), label='Im(v)-Im(m)', alpha=0.8, histtype='step', color='blue')
plt.title("v-m hist for FHD runs (coarse)\nCutoff Threshold: 0.1 Jy")
plt.legend()
# plt.yscale('log')
plt.xlabel("Jy")
plt.savefig('calico/images/examining_fhd_runs_coarse_01.png')
plt.close()

# plt.hist(v.real, bins=np.arange(-10,10,0.15), label='|v|', alpha=0.5, histtype='step', color='blue')
# plt.hist(m.real, bins=np.arange(-10,10,0.15), label='|m|', alpha=0.5, color='orange')
plt.hist(v.real-m.real, bins=np.arange(-5,5,0.15), label='Re(v)-Re(m)', alpha=0.8, histtype='step', color='green')
plt.hist(v.imag-m.imag, bins=np.arange(-5,5,0.15), label='Im(v)-Im(m)', alpha=0.8, histtype='step', color='blue')
plt.title("v-m hist for FHD runs (fine)\nCutoff Threshold: 0.1 Jy")
plt.legend()
# plt.yscale('log')
plt.xlabel("Jy")
plt.savefig('calico/images/examining_fhd_runs_fine_01.png')
plt.close()