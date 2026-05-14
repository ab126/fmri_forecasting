import numpy as np
from utils.measures import compute_di_from_log_prob_api

def generate_coupled_gaussian(N=20000, M=3, a=0.8, sigma=0.5, seed=0):
    rng = np.random.default_rng(seed)

    # Need M history points plus 1 future target
    x = rng.normal(size=N + M)
    eps = rng.normal(scale=sigma, size=N + M)

    y = np.zeros_like(x)
    y[1:] = a * x[:-1] + eps[1:]

    X = np.zeros((N, M, 2), dtype=np.float32)
    Y = np.zeros((N, 1, 2), dtype=np.float32)

    for n in range(N):
        t = n + M
        X[n, :, 0] = x[t-M:t]   # source process history
        X[n, :, 1] = y[t-M:t]   # target process history
        Y[n, 0, 0] = x[t]       # next X
        Y[n, 0, 1] = y[t]       # next Y

    return X, Y


class OracleGaussianAPI:
    def __init__(self, a=0.8, sigma=0.5):
        self.a = a
        self.sigma = sigma

    def predict_proba(self, X, Y=None, batch_size=512):
        N = X.shape[0]
        mean = np.zeros((N, 1, 2), dtype=np.float32)
        std = np.ones((N, 1, 2), dtype=np.float32)

        # ROI 0: X_t ~ N(0, 1)
        mean[:, 0, 0] = 0.0
        std[:, 0, 0] = 1.0

        # ROI 1: Y_t = a X_{t-1} + eps
        source_history_removed = np.all(np.isclose(X[:, :, 0], 0.0), axis=1)

        # Full conditional: p(Y_t | X_{t-1})
        mean[:, 0, 1] = self.a * X[:, -1, 0]
        std[:, 0, 1] = self.sigma

        # Reduced/marginal conditional: p(Y_t) = N(0, a^2 + sigma^2)
        mean[source_history_removed, 0, 1] = 0.0
        std[source_history_removed, 0, 1] = np.sqrt(self.a**2 + self.sigma**2)

        return {"mean": mean, "std": std}


a = 0.8
sigma = 0.5

X, Y = generate_coupled_gaussian(N=50000, M=3, a=a, sigma=sigma)
model = OracleGaussianAPI(a=a, sigma=sigma)

DI = compute_di_from_log_prob_api(
    model,
    X,
    Y,
    horizon_idx=0,
    reduction_mode="zero",
    batch_size=2048,
)

analytic = 0.5 * np.log(1 + a**2 / sigma**2)

print("Estimated DI matrix:")
print(DI)
print("Analytic X -> Y:", analytic)
print("Estimated X -> Y:", DI[0, 1])

