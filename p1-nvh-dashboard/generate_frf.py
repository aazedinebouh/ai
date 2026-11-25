 %%
import numpy as np
import pandas as pd
from scipy.linalg import eigh


def build_matrices(m, k):
    """Mass and stiffness matrices for the chain ground-k1-m1-k2-m2-k3-m3."""
    m1, m2, m3 = m
    k1, k2, k3 = k
    M = np.diag([m1, m2, m3])
    K = np.array([[k1 + k2, -k2,      0.0],
                  [-k2,      k2 + k3, -k3],
                  [0.0,     -k3,       k3]])
    return M, K


def frf(M, K, C, freqs):
    """Receptance at mass 3 (m/N) for a unit force at mass 1."""
    F = np.array([1.0, 0.0, 0.0])
    H3 = np.empty(len(freqs), dtype=complex)
    for i, f in enumerate(freqs):
        w = 2 * np.pi * f
        x = np.linalg.solve(K - w**2 * M + 1j * w * C, F)
        H3[i] = x[2]
    return H3


# %% Generate 50 random designs (random DoE)
rng = np.random.default_rng(42)
freqs = np.arange(1, 201)          # 1 to 200 Hz, 1 Hz steps
rows = []

for v in range(50):
    m = rng.uniform(1, 5, 3)
    k = rng.uniform(1e4, 1e6, 3)
    beta = rng.uniform(1e-5, 1e-4)

    M, K = build_matrices(m, k)
    C = beta * K
    H3 = frf(M, K, C, freqs)
    amp_db = 20 * np.log10(np.abs(H3))

    for f, a in zip(freqs, amp_db):
        rows.append({"variant": v,
                     "m1": m[0], "m2": m[1], "m3": m[2],
                     "k1": k[0], "k2": k[1], "k3": k[2],
                     "beta": beta, "freq_hz": f, "amp_db": a})

df = pd.DataFrame(rows)
df.to_csv("frf_variants.csv", index=False)
print(df.shape)      # expected: (10000, 10)
print(df.head())

# %% Sanity check: FRF peaks vs natural frequencies for variant 0
d0 = df[df.variant == 0]
m0 = d0[["m1", "m2", "m3"]].iloc[0].values
k0 = d0[["k1", "k2", "k3"]].iloc[0].values
M0, K0 = build_matrices(m0, k0)
fn = np.sqrt(eigh(K0, M0, eigvals_only=True)) / (2 * np.pi)
print("Natural frequencies (Hz):", fn.round(1))