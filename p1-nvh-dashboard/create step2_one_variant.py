# %%
import numpy as np
from scipy.linalg import eigh

# One design: masses (kg) and springs (N/m)
m1, m2, m3 = 2.0, 1.5, 1.0
k1, k2, k3 = 5e5, 3e5, 2e5

M = np.diag([m1, m2, m3])
K = np.array([[k1 + k2, -k2,      0],
              [-k2,      k2 + k3, -k3],
              [0,       -k3,       k3]])

# Natural frequencies, like a modal analysis
eigvals, modes = eigh(K, M)
print("Natural frequencies (Hz):", np.sqrt(eigvals) / (2 * np.pi))