# Sundaram Sieve on the 6k±1 Lattice — Spectral Validation

A branch-free implementation of the Sundaram sieve restricted to the 6k±1 residue
classes, plus an FFT-based analysis of the residual signal after wheel factorization.

## Method

Composite indices in 6n+1 / 6n−1 are characterized by three Diophantine forms:

f₁(a,b) = 6ab − a − b   →  (6a−1)(6b−1) = 6·f₁ + 1
f₂(a,b) = 6ab + a + b   →  (6a+1)(6b+1) = 6·f₂ + 1
f₃(a,b) = 6ab + a − b   →  (6a−1)(6b+1) = 6·f₃ − 1

An index `n` is prime in 6k+1 iff `n ∉ image(f₁) ∪ image(f₂)`,
prime in 6k−1 iff `n ∉ image(f₃)`. Two independent sieves, no cross-contamination.

This is a 2,3-wheel variant of Sundaram (1934) in the framework of
Pritchard's wheel factorization (1981).

## Spectral analysis

After wheel factorization with primes p ≤ 31, the residual prime indicator
was analyzed via FFT. Dominant spectral peaks at frequencies k/p reproduce
the modular sieving structure of small primes.

## Results

- Verified against Sieve of Eratosthenes for x ≤ 10⁶ (78,498 primes, exact match).
- Twin-prime count at x ≤ 1.2·10⁶: 9,598 — within 7% of the Hardy-Littlewood
  asymptotic (8,950), consistent with known convergence at this scale.
- Twin-prime spectrum is a scaled copy of the residual spectrum — both
  inherit the same modular harmonics from small primes.

## Performance

Serial Python, ~3× slower than a tight Eratosthenes implementation.
Same complexity class O(N log log N). Optimization (C/SIMD/GPU) not pursued.

## References

- Sundaram, S. P. (1934). The Sundaram sieve.
- Pritchard, P. (1981). A sublinear additive sieve for finding prime numbers.
- Hardy, G. H. & Littlewood, J. E. (1923). Some problems of "Partitio Numerorum" III.

## Files

- `dpim_full.py` — sieve + verification + benchmark
- `dpim_synthesis.py` — top-N FFT reconstruction vs. wheel baseline
- `dpim_residual_v2.py` — residual and twin-prime spectrum

## License

MIT
