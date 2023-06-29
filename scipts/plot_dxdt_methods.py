import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz

# Create a time vector from 0 to 1 second
T = 1.0
num_points = 500
t = np.linspace(0, T, num_points)
# Create a continuous sinusoidal signal
x = np.sin(2 * np.pi * 1 * t)

# Sample the signal
N = num_samples = 10

ts = np.linspace(0, T, num_samples)
xs = np.sin(2 * np.pi * 1 * ts)
dxs = np.cos(2 * np.pi * 1 * ts)

# Use ZOH to reconstruct the signal
zoh = interp1d(ts, xs, kind='zero', bounds_error=False, fill_value="extrapolate")
# foh = interp1d(ts, xs, kind='first', bounds_error=False, fill_value="extrapolate")
xr = zoh(t)

# Approximate the integral of the signal using the trapezoidal rule
# Output signal
xt = np.zeros_like(ts)
yt = np.zeros_like(ts)

# Discrete-time simulation using the trapezoidal rule
for k in range(1, len(ts)):
    xt[k] = xt[k-1] + (T/N)/2 * (dxs[k] + dxs[k-1])
    xt[k] = xt[k-1] + (T/N)/2 * (dxs[k] + dxs[k-1])

# Plot the original signal, the sampled signal, the reconstructed signal and the integral
fig, ax = plt.subplots(1,1,dpi=150)

ax.plot(t, x, label='Signal',color='k')
ax.plot(ts, xt, label='Trapezoidal rule',ls='-')
ax.stem(ts, xs, linefmt='g-', markerfmt='gx', basefmt='g-', label='Sampled')
ax.step(t, xr, 'C5', label='Zero-order hold',zorder=3)

ax.legend(loc='upper right')
ax.set_xlim(0,1)
ax.set_xlabel('Time [s]')

w_path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Special course - System identification of black-box dynamical systems\img'
plt.savefig(f'{w_path}\\dxdt_methods.pdf')


# This script creates a sinusoidal waveform, samples it, then uses ZOH to reconstruct the signal from the samples. The integral of
