"""
Residuen-Analyse: FFT des Signals nach Abzug der Wheel-Harmonischen (p <= 31)
Franz' Code, vollständig ausgeführt.
"""

import numpy as np
import matplotlib.pyplot as plt

N_IDX = 200_000

# 1. DPIM-Basisdaten generieren
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

# 2. Wheel-Sieb (p <= 31): nur Indizes wo BEIDE Seiten Wheel-Survivor sind
small_primes = [5, 7, 11, 13, 17, 19, 23, 29, 31]
n_arr = np.arange(1, N_IDX + 1)
wheel_mask = np.ones(N_IDX, dtype=bool)
for p in small_primes:
    wheel_mask &= ~(((6*n_arr + 1) % p == 0) & (6*n_arr + 1 != p))
    wheel_mask &= ~(((6*n_arr - 1) % p == 0) & (6*n_arr - 1 != p))

# 3. Primzahl-Indikatoren
is_prime_plus  = (m_plus[1:]  == 0)
is_prime_minus = (m_minus[1:] == 0)
actual_primes  = is_prime_plus | is_prime_minus

# 4. Residuen-Signal R(n)
residuell = np.zeros(N_IDX)
residuell[wheel_mask] = actual_primes[wheel_mask].astype(float)
res_centered = residuell - residuell.mean()

# Statistik
print(f"N_IDX = {N_IDX}")
print(f"Wheel-Survivor-Anteil:        {wheel_mask.mean():.4f}")
print(f"Primzahl-Anteil bei Survivors: {actual_primes[wheel_mask].mean():.4f}")
print(f"Primzahl-Anteil insgesamt:    {actual_primes.mean():.4f}")
print(f"Lift durch Wheel: {actual_primes[wheel_mask].mean() / actual_primes.mean():.2f}x")

# 5. Spektralanalyse
fft_res   = np.fft.rfft(res_centered)
power_res = np.abs(fft_res) ** 2
freqs     = np.fft.rfftfreq(N_IDX)

# 6. Zwillings-Signal
twins = is_prime_plus & is_prime_minus
twins_centered = twins.astype(float) - twins.mean()
fft_twins   = np.fft.rfft(twins_centered)
power_twins = np.abs(fft_twins) ** 2

print(f"\nAnzahl Zwillings-Paare: {twins.sum()}")

# Top-Peaks
def label_peak(f):
    for p in [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]:
        for k in range(1, p):
            if abs(f - k/p) < 0.0003:
                return f"{k}/{p}"
    return ""

print("\nTop 10 Peaks im Residuen-Spektrum (Wheel-Maskierung):")
top = np.argsort(power_res[1:])[-10:][::-1] + 1
for i in top:
    f = freqs[i]; T = 1/f if f > 0 else np.inf
    print(f"  f={f:.5f}  T={T:7.2f}  Power={power_res[i]:11.1f}  → {label_peak(f)}")

print("\nTop 10 Peaks im Zwillings-Spektrum:")
top = np.argsort(power_twins[1:])[-10:][::-1] + 1
for i in top:
    f = freqs[i]; T = 1/f if f > 0 else np.inf
    print(f"  f={f:.5f}  T={T:7.2f}  Power={power_twins[i]:11.1f}  → {label_peak(f)}")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(13, 10))
axes[0].semilogy(freqs[1:], power_res[1:], color='teal', lw=0.5)
axes[0].set_title("Spektrum des Restsignals (Wheel-Maskierung, p≤31)")
axes[0].set_xlabel("Frequenz"); axes[0].set_ylabel("Power"); axes[0].set_xlim(0, 0.5)

axes[1].semilogy(freqs[1:], power_twins[1:], color='crimson', lw=0.5)
axes[1].set_title("Spektrum der Primzahl-Zwillinge (Twin-Indicator FFT)")
axes[1].set_xlabel("Frequenz"); axes[1].set_ylabel("Power"); axes[1].set_xlim(0, 0.5)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/dpim_residual_v2.png', dpi=120)
print("\nPlot: dpim_residual_v2.png")
