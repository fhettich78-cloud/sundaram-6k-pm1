"""
6k±1-Sundaram-Sieb: vollstaendige Implementierung mit Verifikation gegen
Eratosthenes und Benchmark.

Mathematischer Kern:
    f1(a,b) = 6ab - a - b   ->  (6a-1)(6b-1) = 6*f1 + 1
    f2(a,b) = 6ab + a + b   ->  (6a+1)(6b+1) = 6*f2 + 1
    f3(a,b) = 6ab + a - b   ->  (6a-1)(6b+1) = 6*f3 - 1

Index n ist prim in 6k+1 gdw. n nicht in image(f1) U image(f2).
Index n ist prim in 6k-1 gdw. n nicht in image(f3).
Zwei separate Siebe - nicht zusammenfassen.
"""
import time
import numpy as np


def dpim_sieve(N):
    """Liefert alle Primzahlen <= N via 6k+-1-Sundaram-Sieb."""
    if N < 2:
        return []

    # n_max so, dass 6*n+1 und 6*n-1 noch ueber N hinausreichen koennen
    N_IDX = N // 6 + 1

    m_plus  = np.zeros(N_IDX + 2, dtype=np.int32)
    m_minus = np.zeros(N_IDX + 2, dtype=np.int32)

    # f1: (6a-1)(6b-1) = 6*f1 + 1, a <= b
    a = 1
    while 6*a*a - 2*a <= N_IDX:
        b = a; n = 6*a*b - a - b
        while n <= N_IDX:
            m_plus[n] += 1
            b += 1; n = 6*a*b - a - b
        a += 1

    # f2: (6a+1)(6b+1) = 6*f2 + 1, a <= b
    a = 1
    while 6*a*a + 2*a <= N_IDX:
        b = a; n = 6*a*b + a + b
        while n <= N_IDX:
            m_plus[n] += 1
            b += 1; n = 6*a*b + a + b
        a += 1

    # f3: (6a-1)(6b+1) = 6*f3 - 1, a,b >= 1 (nicht symmetrisch)
    a = 1
    while 7*a - 1 <= N_IDX:
        b = 1; n = 6*a*b + a - b
        while n <= N_IDX:
            m_minus[n] += 1
            b += 1; n = 6*a*b + a - b
        a += 1

    # Primzahlen einsammeln
    primes = []
    if N >= 2: primes.append(2)
    if N >= 3: primes.append(3)

    for n in range(1, N_IDX + 1):
        if m_minus[n] == 0:
            val = 6*n - 1
            if 5 <= val <= N:
                primes.append(val)
        if m_plus[n] == 0:
            val = 6*n + 1
            if val <= N:
                primes.append(val)

    primes.sort()
    return primes


def eratosthenes(N):
    """Sieb des Eratosthenes als Referenzimplementierung."""
    if N < 2:
        return []
    sieve = bytearray(b'\x01') * (N + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(N**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, N + 1) if sieve[i]]


def twin_prime_count(primes):
    """Zaehlt Primzahlpaare (p, p+2)."""
    s = set(primes)
    return sum(1 for p in primes if (p + 2) in s)


if __name__ == "__main__":
    print("=" * 64)
    print("VERIFIKATION (DPIM vs. Eratosthenes)")
    print("=" * 64)
    for N in [100, 1_000, 10_000, 100_000, 1_000_000]:
        a = dpim_sieve(N)
        b = eratosthenes(N)
        match = "OK" if a == b else "FEHLER"
        print(f"  N = {N:>9}: {len(a):>7} Primzahlen   identisch? {match}")

    print()
    print("=" * 64)
    print("ZWILLINGSPRIMZAHLEN bei x <= 1.2e6")
    print("=" * 64)
    primes = dpim_sieve(1_200_000)
    tw = twin_prime_count(primes)
    # Hardy-Littlewood-Asymptotik (Integralform): 2*C2 * Li2(x) = 2*C2 * int_2^x dt/(ln t)^2
    import math
    from scipy.integrate import quad
    C2 = 0.6601618158468696
    Li2, _ = quad(lambda t: 1 / math.log(t)**2, 2, 1_200_000)
    hl = 2 * C2 * Li2
    err = abs(tw - hl) / hl * 100
    print(f"  Anzahl Zwillinge: {tw}")
    print(f"  Hardy-Littlewood: {hl:.0f}  (Integralform 2*C2*Li2(x))")
    print(f"  Abweichung:       {err:.2f}%")

    print()
    print("=" * 64)
    print("BENCHMARK (reines Python, single-thread)")
    print("=" * 64)
    print(f"  {'N':>10} | {'DPIM (s)':>10} | {'Eratos. (s)':>12} | {'Verhaeltnis':>11}")
    print("  " + "-" * 56)
    for N in [100_000, 1_000_000]:
        t0 = time.perf_counter(); dpim_sieve(N);   t_dpim = time.perf_counter() - t0
        t0 = time.perf_counter(); eratosthenes(N); t_era  = time.perf_counter() - t0
        print(f"  {N:>10} | {t_dpim:>10.3f} | {t_era:>12.3f} | {t_dpim/t_era:>10.2f}x")
