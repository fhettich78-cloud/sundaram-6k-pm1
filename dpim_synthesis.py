"""
Top-N FFT-Rekonstruktion des Primzahl-Indikators
gegen explizites Wheel-Sieb als Vergleichs-Baseline.

Test-Logik:
  1. FFT der zentrierten Indikator-Folge
  2. Top-N Modi behalten, IFFT -> synthetische Welle
  3. Pearson-Korrelation und Prim-Dichte an Extrempunkten
  4. Wheel-Sieb mit identischen Primzahlen als Baseline
     Wenn beide aehnliche Lifts liefern, sind die FFT-Modi
     im Wesentlichen die Wheel-Harmonischen.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import pearsonr

N_IDX = 100_000

# Sieb-Aufbau (identisch zu dpim_residual_v2.py)
m_plus  = np.zeros(N_IDX + 1, dtype=np.int32)
m_minus = np.zeros(N_IDX + 1, dtype=np.int32)

a = 1
while 6*a*a - 2*a <= N_IDX:
    b = a; n = 6*a*b - a - b
    while n <= N_IDX:
        m_plus[n] += 1; b += 1; n = 6*a*b - a - b
    a += 1

a = 1
while 6*a*a + 2*a <= N_IDX:
    b = a; n = 6*a*b + a + b
    while n <= N_IDX:
        m_plus[n] += 1; b += 1; n = 6*a*b + a + b
    a += 1

a = 1
while 7*a - 1 <= N_IDX:
    b = 1; n = 6*a*b + a - b
    while n <= N_IDX:
        m_minus[n] += 1; b += 1; n = 6*a*b + a - b
    a += 1

is_prime_plus  = (m_plus[1:]  == 0)
is_prime_minus = (m_minus[1:] == 0)
indicator      = is_prime_plus.astype(float) + is_prime_minus.astype(float)
ind_centered   = indicator - indicator.mean()

# FFT
fft   = np.fft.rfft(ind_centered)
power = np.abs(fft) ** 2
freqs = np.fft.rfftfreq(len(ind_centered))

print("=" * 78)
print("REKONSTRUKTION AUS TOP-N FFT-PEAKS")
print("=" * 78)
print(f"Indikator-Laenge: {len(indicator)}")
print(f"Anteil Prim-Indizes (Zufalls-Baseline): {(indicator > 0).mean():.4f}")
print()
print(f"  {'N':>4} | {'Var.erkl.':>10} | {'Pearson r':>10} | "
      f"{'Max->Prim':>10} | {'Min->Prim':>10} | {'Lift':>6}")
print("  " + "-" * 70)

baseline = (indicator > 0).mean()

for N in [5, 10, 20, 50, 100, 200, 500]:
    top_idx = np.argsort(power[1:])[-N:][::-1] + 1
    fft_filtered = np.zeros_like(fft)
    fft_filtered[top_idx] = fft[top_idx]
    synthetic = np.fft.irfft(fft_filtered, n=len(ind_centered))

    r, _ = pearsonr(synthetic, indicator)
    var_explained = np.var(synthetic) / np.var(ind_centered)

    maxima_idx, _ = find_peaks(synthetic)
    minima_idx, _ = find_peaks(-synthetic)
    max_to_prime = (indicator[maxima_idx] > 0).mean() if len(maxima_idx) > 0 else 0
    min_to_prime = (indicator[minima_idx] > 0).mean() if len(minima_idx) > 0 else 0
    lift = max_to_prime / baseline if baseline > 0 else 0

    print(f"  {N:>4} | {var_explained:>9.3f}  | {r:>10.4f} | "
          f"{max_to_prime:>9.2%} | {min_to_prime:>9.2%} | {lift:>5.2f}x")

# Wheel-Baseline
print()
print("=" * 78)
print("BASELINE: EXPLIZITES WHEEL-SIEB")
print("=" * 78)

# Primzahlen aus Top-20-Peaks bestimmen
top20 = np.argsort(power[1:])[-20:][::-1] + 1
small_primes = set()
for i in top20:
    f = freqs[i]
    for p in [5, 7, 11, 13, 17, 19, 23, 29, 31]:
        for k in range(1, p):
            if abs(f - k/p) < 0.001:
                small_primes.add(p)
                break
small_primes = sorted(small_primes)
print(f"Primzahlen in Top-20-Peaks: {small_primes}")

n_arr = np.arange(1, N_IDX + 1)
wheel_survives = np.ones(N_IDX, dtype=bool)
for p in small_primes:
    wheel_survives &= ~(((6*n_arr + 1) % p == 0) & (6*n_arr + 1 != p))
    wheel_survives &= ~(((6*n_arr - 1) % p == 0) & (6*n_arr - 1 != p))

wheel_filter = wheel_survives.astype(float)
r_wheel, _ = pearsonr(wheel_filter, indicator)
prime_density_in  = (indicator[wheel_survives].astype(bool)).mean()
prime_density_out = (indicator[~wheel_survives].astype(bool)).mean() if (~wheel_survives).any() else 0
lift_wheel = prime_density_in / baseline

print(f"Pearson r (Wheel vs. Indikator):     {r_wheel:.4f}")
print(f"Prim-Dichte innerhalb Wheel-Survivor: {prime_density_in:.4f}")
print(f"Prim-Dichte ausserhalb:               {prime_density_out:.4f}")
print(f"Lift durch Wheel:                     {lift_wheel:.2f}x")
print()
print("Interpretation: aehnliche Lifts zwischen FFT-Synthese und Wheel-Sieb")
print("zeigen, dass die Top-Peaks die modularen Harmonischen kleiner Primzahlen sind.")

# Plot
top_N_plot = 50
top_idx = np.argsort(power[1:])[-top_N_plot:][::-1] + 1
fft_filtered = np.zeros_like(fft)
fft_filtered[top_idx] = fft[top_idx]
synthetic = np.fft.irfft(fft_filtered, n=len(ind_centered))

show = 2000
fig, axes = plt.subplots(2, 1, figsize=(13, 8))
axes[0].plot(synthetic[:show], color='teal', lw=0.7, label=f'Top-{top_N_plot} Synthese')
axes[0].plot(ind_centered[:show], color='black', lw=0.4, alpha=0.5, label='Indikator')
axes[0].set_title(f"Synthetische Welle aus Top-{top_N_plot} FFT-Peaks (n = 1..{show})")
axes[0].set_xlabel("Index n"); axes[0].set_ylabel("Amplitude"); axes[0].legend()

axes[1].plot(wheel_filter[:show] - wheel_filter.mean(), color='crimson', lw=0.7,
             label='Wheel-Survivor')
axes[1].plot(ind_centered[:show], color='black', lw=0.4, alpha=0.5, label='Indikator')
axes[1].set_title(f"Wheel-Sieb-Filter (Primzahlen {small_primes}) vs. Indikator")
axes[1].set_xlabel("Index n"); axes[1].set_ylabel("Amplitude"); axes[1].legend()

plt.tight_layout()
plt.savefig('dpim_synthesis.png', dpi=120)
print("\nPlot: dpim_synthesis.png")
