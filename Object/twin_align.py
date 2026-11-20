"""Digital twin of the off-axis holography bench: the object arm.

Shared by the notebooks in this folder. Notation follows the manuscript and the Supplementary Information (SI):

    X_O = x * exp(j theta)              SLM field of the object pattern (x = exp(j phi) carries the data)   Eq. (1)
    O   = T_O{X_O},  I = |O + R|^2      object field on the camera and the off-axis hologram                Eq. (1)
    W   = g * O_hat,  g = dL/d|O_hat|^2  error field, formed from the MEASURED field O_hat                  Eq. (2)
    dL/dtheta = 2 Im[X_O^* T_O^H{W}]    gradient of the object arm                                          Eq. (3)
    T_O = P_zs diag(D) P_-zs F          thin-screen model of the diffuser                                   Eq. (4)
    D_hat = argmin sum_k l(O_hat_k, T_O[D]{X_k})    screen fitted to K probe fields                         Eq. (5), SI Eq. (S9)

The twin hides the true screen and the lens aberrations from every method and adds the noise of the bench
(per-pulse wavefront jitter, laser phase and energy jitter, shot and read noise, 12-bit camera).

twin_align.py: the same twin with the object of align.py, 512 x 512 SLM pixels of 8 um (a 4.096 mm square)
instead of 32 x 32 cells of 40 um. Everything else (iris B, distances, screen, camera, noise, carrier) is unchanged.
The 20 um field grid still holds the field, since the iris keeps only |f| <= B = 12 lp/mm: the SLM is carried onto
the grid by its exact spectrum on the grid frequencies (pixel sinc included), and its adjoint is exact too. The iris
passes about 2 B x 4.096 mm ~ 98 resolvable samples per axis, so the 512 x 512 pattern reaches the camera
low-passed (the quality loss is accepted). Use it as `import twin_align as tw`; NC is now 512 SLM pixels, and every
pattern X is (..., 512, 512). The transmission-matrix benchmark (N = 512^2 inputs) is not practical at this size.
"""
import warnings
import numpy as np
import scipy.fft as sfft
import scipy.optimize
import matplotlib.pyplot as plt
from PIL import Image

# Apple's Accelerate BLAS (used by NumPy 2 on macOS) raises false "divide by zero / overflow / invalid value
# encountered in matmul" warnings; the products are finite (fit_tm checks it), so these warnings are hidden.
warnings.filterwarnings("ignore", message=r".*encountered in matmul", category=RuntimeWarning)

# ----------------------------------------------------------------------------------------------------------------------
# Geometry (bench values). Fields live on a 20 um grid; the object is NC x NC SLM pixels of PITCH (8 um), as in align.py.
# ----------------------------------------------------------------------------------------------------------------------
LAM = 1.03e-6        # wavelength (m)
DX = 20e-6           # field grid pitch (m)
PITCH = 8e-6         # SLM pixel pitch (m); one object sample is one SLM pixel
B = 12e3             # iris radius in the Fourier plane of the relay = camera band (cycles/m)
Z1 = 13.1e-3         # SLM image -> diffuser screen (m)
ZS = 53.7e-3         # diffuser screen -> camera, z_s of Eq. (4) (m)
MARGIN_M = 1.4e-3    # field around the SLM aperture (light spreads < 1.4 mm before the camera) (m)
ABERRATION = dict(defocus=0.6, astig=-0.45, coma=0.6, spherical=-0.25)   # true lens aberrations (rad), hidden


def fft2(a):
    return sfft.fft2(a, norm="ortho", workers=-1)


def ifft2(a):
    return sfft.ifft2(a, norm="ortho", workers=-1)


def configure(nc=512):
    """Set the grid for nc x nc SLM pixels of PITCH. Iris, distances and noise do not depend on nc (SI Note 6)."""
    global NC, N, F2, FX, FY, BAND, BAND2, AP, UP, NU, FCB, KB, CARRIER, CHUNK, WIN, DISK, E_SLM
    NC = nc
    width = nc * PITCH                         # SLM aperture (m): 4.096 mm for 512 pixels
    N = 2 * int(np.ceil((width + 2 * MARGIN_M) / DX / 2))
    N = N if N % 8 == 0 else N + 8 - N % 8     # 360 for 512 pixels (7.2 mm)
    f = sfft.fftfreq(N, DX)
    FX, FY = np.meshgrid(f, f)
    F2 = FX**2 + FY**2
    BAND = F2 <= B**2                          # band of the object field on the camera
    BAND2 = F2 <= (2 * B)**2                   # band of the screen D (it can couple 2B into the camera band)
    x0 = (N * DX - width) / 2                  # SLM aperture centred on the grid
    AP = slice(int(np.floor(x0 / DX)), int(np.ceil((x0 + width) / DX)))      # SLM aperture on the grid
    # SLM pixels -> grid spectrum: E[m, k] = PITCH sinc(f_m PITCH) exp(-j 2 pi f_m x_k), x_k the pixel centres
    xk = x0 + (np.arange(nc) + 0.5) * PITCH
    E_SLM = PITCH * np.sinc(f * PITCH)[:, None] * np.exp(-2j * np.pi * f[:, None] * xk[None, :])
    # camera: 10 um pixels (2 x the grid) and a diagonal carrier of 28.1 lp/mm at every size
    UP = 2
    NU = N * UP
    FCB = 108 * N // 192                       # carrier bin per axis
    KB = int(np.ceil(B * N * DX)) + 1          # half-size of the sideband block (bins)
    yc, xc = np.mgrid[0:NU, 0:NU]
    CARRIER = np.exp(2j * np.pi * FCB * (xc + yc) / NU).astype(np.complex64)
    # camera frames per batch (memory): about 2 GB, counting the SLM patterns (48 B/px) and camera frames (64 B/px)
    CHUNK = max(1, min(int(320 * (384 / NU)**2), int(2e9 / (48 * nc**2 + 64 * NU**2))))
    # image window (43 x 43 samples = 0.86 mm) and focus target (one speckle grain), at the centre of the field
    yy, xx = np.mgrid[0:N, 0:N] - N // 2
    WIN = (np.abs(yy) <= 21) & (np.abs(xx) <= 21)
    DISK = yy**2 + xx**2 <= 2.6**2


configure(512)


# ----------------------------------------------------------------------------------------------------------------------
# Free space and the object arm, Eq. (4)
# ----------------------------------------------------------------------------------------------------------------------
def transfer(z):
    """Angular-spectrum transfer function of free space over z (paraxial), P_z in Fourier space."""
    return np.exp(-1j * np.pi * LAM * z * F2)


def pupil(aberration=None):
    """Iris of radius B with low-order Zernike aberrations (rad at the iris edge)."""
    a = aberration or {}
    u, v, r2 = FX / B, FY / B, F2 / B**2
    phase = (a.get("defocus", 0) * (2 * r2 - 1) + a.get("astig", 0) * 2 * u * v
             + a.get("coma", 0) * (3 * r2 - 2) * u + a.get("spherical", 0) * (6 * r2**2 - 6 * r2 + 1))
    return BAND * np.exp(1j * phase)


def cells_to_grid(X):
    """SLM pixels (..., NC, NC) -> field on the grid: the exact spectrum of the pixelated SLM field on the grid
    frequencies (|f| <= 25 lp/mm; the iris keeps |f| <= B), back to the 20 um grid. A lit pixel of value 1 gives 1."""
    return sfft.ifft2(E_SLM @ X @ E_SLM.T, workers=-1) / DX**2


def grid_to_cells(u):
    """Adjoint of cells_to_grid."""
    return np.conj(E_SLM).T @ sfft.fft2(u, workers=-1) @ np.conj(E_SLM) / (N**2 * DX**2)


class ObjectArm:
    """T_O = P_zs diag(D) P_-zs F and its adjoint T_O^H = F^H P_zs diag(D^*) P_-zs (SI Eqs. S8, S9).

    F carries the SLM cells through the relay (iris and aberrations as a pupil) to the camera plane; P_-zs takes the
    field back to the screen, D multiplies it, and P_zs brings it to the camera. Since the propagators commute,
    P_-zs F is applied as one transfer function (the relay to the screen plane, Z1).
    D = 1 gives free space.
    """

    def __init__(self, D=None, aberration=None):
        self.D = np.ones((N, N), complex) if D is None else D
        self.H_in = pupil(aberration) * transfer(Z1)     # P_-zs F
        self.H_out = transfer(ZS) * BAND                 # P_zs, then the camera band kept by the demodulation

    def at_screen(self, X):
        """V = P_-zs F X: the field that illuminates the screen."""
        return ifft2(fft2(cells_to_grid(X)) * self.H_in)

    def forward(self, X):
        return ifft2(fft2(self.at_screen(X) * self.D) * self.H_out)

    def adjoint(self, W):
        v = ifft2(fft2(W) * np.conj(self.H_out)) * np.conj(self.D)
        return grid_to_cells(ifft2(fft2(v) * np.conj(self.H_in)))


def random_screen(rng, corr_len=60e-6, sigma=2.0):
    """Ground-glass surface: Gaussian-correlated phase (60 um, 2 rad rms), band-limited to 2B."""
    phase = np.real(ifft2(fft2(rng.standard_normal((N, N))) * np.exp(-F2 * (np.pi * corr_len)**2)))
    return ifft2(fft2(np.exp(1j * sigma * phase / phase.std())) * BAND2)


def shift_screen(D, dy):
    """Move a screen sideways by dy metres along the first axis (Fourier shift, so any fraction of a grid step)."""
    return ifft2(fft2(D) * np.exp(-2j * np.pi * FY * dy))


def random_patterns(rng, n):
    """n random unit-amplitude phase patterns on the SLM pixels."""
    return np.exp(2j * np.pi * rng.random((n, NC, NC)))


def amplitude_image(img):
    """align.py's "image" object: a picture resized to NC x NC as AMPLITUDE (max 1), phase = 1 rad everywhere."""
    img = np.asarray(img, float)
    img = img.mean(axis=2) if img.ndim == 3 else img
    amp = np.asarray(Image.fromarray(img).resize((NC, NC), Image.BOX))
    return amp / (amp.max() + 1e-12) * np.exp(1j)


# ----------------------------------------------------------------------------------------------------------------------
# Off-axis holography: one frame gives the complex field (SI Note 1B)
# ----------------------------------------------------------------------------------------------------------------------
def to_camera(O, up=None):
    """Band-limited interpolation of a field onto the camera pixels (up x finer than the grid; default UP)."""
    up = up or UP
    S = fft2(O)
    big = np.zeros(O.shape[:-2] + (N * up, N * up), np.complex64)
    for a in (slice(0, KB), slice(-KB, None)):
        for b in (slice(0, KB), slice(-KB, None)):
            big[..., a, b] = S[..., a, b]
    return ifft2(big) * up


def hologram(O, r0):
    """I = |O + R|^2 with a tilted plane-wave reference R = r0 exp(j 2 pi f_c . x)."""
    return np.abs(to_camera(O) + r0 * CARRIER)**2


def demodulate(I, r0):
    """Fourier filter: crop the sideband around the carrier, shift it to zero, divide by the reference -> O_hat."""
    S = sfft.rfft2(I, norm="ortho", workers=-1)
    side = np.zeros(I.shape[:-2] + (N, N), np.complex64)
    c = N // 2
    side[..., c - KB:c + KB, c - KB:c + KB] = S[..., FCB - KB:FCB + KB, FCB - KB:FCB + KB]
    d = ifft2(sfft.ifftshift(side * sfft.fftshift(BAND), axes=(-2, -1))) / UP
    return np.conj(d) / r0


# ----------------------------------------------------------------------------------------------------------------------
# The twin bench
# ----------------------------------------------------------------------------------------------------------------------
# Three light levels, as in the experiments: a dim run for focusing (the focus must not saturate the 12-bit camera),
# a brighter run for images, and long exposures (about 40 pulses, so the per-pulse jitter averages down) for
# classification, where each image is read once.
PRESETS = {
    "focus":          dict(sig_cell=0.30, sig_low=0.22, energy_jitter=0.16, ref_counts=200, obj_counts=2),
    "imaging":        dict(sig_cell=0.20, sig_low=0.15, energy_jitter=0.16, ref_counts=1000, obj_counts=10),
    "classification": dict(sig_cell=0.05, sig_low=0.04, energy_jitter=0.05, ref_counts=1000, obj_counts=10),
}


class Bench:
    """The bench seen by the methods: write SLM patterns, get camera frames. The true arm is hidden in `_true`.

    sig_cell, sig_low  per-pulse wavefront jitter per cell and in five low-order modes (rad)
    energy_jitter      pulse energy jitter; the pulse phase is uniform (one unknown complex factor per frame)
    ref_counts, obj_counts   reference and mean object level per camera pixel (counts); 3-count read noise, 12 bit
    """

    def __init__(self, preset="imaging", seed=0, read_noise=3.0):
        p = PRESETS[preset]
        g = np.random.default_rng(seed)
        self.rng = np.random.default_rng(seed + 1)
        self._true = ObjectArm(random_screen(g), ABERRATION)
        self.sig_cell, self.sig_low, self.ej = p["sig_cell"], p["sig_low"], p["energy_jitter"]
        self.read_noise, self.r0 = read_noise, np.sqrt(p["ref_counts"])
        x = np.exp(2j * np.pi * g.random((NC, NC)))
        self.scale = np.sqrt(p["obj_counts"] / np.mean(np.abs(self._true.forward(x)[AP, AP])**2))
        u = (np.arange(NC) - (NC - 1) / 2) / (NC / 2)
        U, V = np.meshgrid(u, u)
        modes = np.stack([U, V, 2 * (U**2 + V**2) - 1, U**2 - V**2, 2 * U * V])
        self.modes = modes / np.sqrt((modes**2).mean(axis=(1, 2), keepdims=True))
        self.patterns = 0        # SLM patterns displayed (the cost unit of the manuscript)
        self.exposures = 0       # camera exposures

    def frame(self, X):
        """One raw camera frame (counts) per pattern in X (..., NC, NC)."""
        a = self.rng.standard_normal(X.shape[:-2] + (5, 1, 1)) * self.sig_low / np.sqrt(5)
        jitter = np.sum(a * self.modes, axis=-3) + self.sig_cell * self.rng.standard_normal(X.shape)
        O = self._true.forward(X * np.exp(1j * jitter)) * self.scale
        energy = np.maximum(1 + self.ej * self.rng.standard_normal(X.shape[:-2] + (1, 1)), 0.05)
        pulse = np.sqrt(energy) * np.exp(2j * np.pi * self.rng.random(X.shape[:-2] + (1, 1)))
        I = hologram(O * pulse, self.r0)
        I = I + np.sqrt(I + self.read_noise**2) * self.rng.standard_normal(I.shape)      # shot + read noise
        self.exposures += int(np.prod(X.shape[:-2]))
        return np.clip(np.round(I), 0, 4095)

    def read(self, X, K=1):
        """Measured field O_hat of each pattern: K exposures, each demodulated, aligned in global phase, averaged.

        X is one pattern (NC, NC) or a stack (n, NC, NC). One hologram records one SLM pattern (SI Note 1A).
        """
        single = X.ndim == 2
        X = X[None] if single else X
        out = []
        n = max(1, CHUNK // K)
        for i in range(0, len(X), n):
            d = demodulate(self.frame(np.repeat(X[i:i + n, None], K, axis=1)), self.r0)        # (n, K, N, N)
            phase = np.exp(-1j * np.angle(np.sum(np.conj(d[:, :1]) * d, axis=(-2, -1))))
            out.append(np.mean(d * phase[..., None, None], axis=1))
        self.patterns += len(X)
        out = np.concatenate(out)
        return out[0] if single else out


# ----------------------------------------------------------------------------------------------------------------------
# Comparing fields (SI Note 1B)
# ----------------------------------------------------------------------------------------------------------------------
def corr(a, b):
    """Gauge-invariant correlation |<a, b>| / (|a| |b|): ignores the global phase and amplitude of a frame."""
    return np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))


def model_accuracy(arm, X, Y, mask=None):
    """Mean correlation between predicted and measured fields of held-out patterns, inside `mask` (default: all)."""
    P = arm.forward(X)
    m = np.ones((N, N), bool) if mask is None else mask
    return float(np.mean([corr(p[m], y[m]) for p, y in zip(P, Y)]))


def to_model_gauge(O_hat, pred, mask=None):
    """Rotate a measured field into the model's gauge: one phase per frame from the forward prediction (SI Eq. S12)."""
    m = np.ones((N, N), bool) if mask is None else mask
    zeta = np.vdot(pred[m], O_hat[m])
    return O_hat * np.conj(zeta) / abs(zeta)


# ----------------------------------------------------------------------------------------------------------------------
# Fitting the screen, Eq. (5) and SI Eq. (S9)
# ----------------------------------------------------------------------------------------------------------------------
def fit_screen(X, Y, arm=None, iters=10, ridge=1e-3, mask=None):
    """Fit the screen D of the thin-screen model to K measured probe fields Y_k = O_hat_k written as patterns X_k.

    Gauge-invariant misfit l(a, b) = 1 - |<a, b>|^2 / (|a|^2 |b|^2) per probe, plus a small ridge towards the starting
    screen, minimised with L-BFGS and exact gradients. The prediction P_zs(V_k D) is linear in D, with V_k = P_-zs F X_k.
    A fixed, small number of L-BFGS iterations is the early stopping: with few probes, a longer fit overfits the noise.
    `arm` gives the free-space part and the starting screen (default: free space, D = 1); a new arm is returned.
    `mask` restricts the misfit to the camera pixels where the fields were measured (default: everywhere).
    """
    arm = arm or ObjectArm()
    V = arm.at_screen(X)
    D0 = arm.D.copy()
    m = np.ones((N, N)) if mask is None else mask.astype(float)
    Y = np.asarray(Y) * m
    yy = np.sum(np.abs(Y)**2, axis=(1, 2))
    K = len(X)

    def misfit(p):
        D = (p[:N * N] + 1j * p[N * N:]).reshape(N, N)
        M = ifft2(fft2(V * D) * arm.H_out) * m                     # predicted fields
        c = np.sum(np.conj(Y) * M, axis=(1, 2))                    # <Y_k, M_k>
        mm = np.sum(np.abs(M)**2, axis=(1, 2))
        loss = np.mean(1 - np.abs(c)**2 / (yy * mm)) + ridge * np.mean(np.abs(D - D0)**2)
        dM = -(c / (yy * mm))[:, None, None] * (Y - (np.conj(c) / mm)[:, None, None] * M) / K     # dl/dM^*
        gD = np.sum(np.conj(V) * ifft2(fft2(dM * m) * np.conj(arm.H_out)), axis=0) + ridge * (D - D0) / N**2
        gD = ifft2(fft2(gD) * BAND2)                               # keep D band-limited to 2B
        return loss, np.concatenate([2 * gD.real.ravel(), 2 * gD.imag.ravel()])

    p0 = np.concatenate([D0.real.ravel(), D0.imag.ravel()])
    res = scipy.optimize.minimize(misfit, p0, jac=True, method="L-BFGS-B",
                                  options=dict(maxiter=iters, gtol=0, ftol=0))
    fitted = ObjectArm()
    fitted.D = (res.x[:N * N] + 1j * res.x[N * N:]).reshape(N, N)
    return fitted


# ----------------------------------------------------------------------------------------------------------------------
# Transmission-matrix benchmark (SI Note 3C)
# ----------------------------------------------------------------------------------------------------------------------
def fit_tm(X, Z, Z_seed, ridge=1e-3, rounds=8):
    """TM from patterns X (M, NC, NC) to measured window fields Z (M, n_out).

    Each measured field carries an unknown factor, so T is found by ridge regression alternated with the phases of
    these factors, initialised from the physical model's predictions Z_seed.
    """
    Xf = X.reshape(len(X), -1)
    G = Xf.conj().T @ Xf
    G_inv = np.linalg.inv(G + ridge * np.trace(G).real / len(G) * np.eye(len(G)))
    c = np.exp(1j * np.angle(np.sum(np.conj(Z_seed) * Z, 1)))
    for _ in range(rounds):
        T = (G_inv @ (Xf.conj().T @ (np.conj(c)[:, None] * Z))).T
        c = np.exp(1j * np.angle(np.sum(np.conj(Xf @ T.T) * Z, 1)))
    assert np.all(np.isfinite(T))
    return T


class TMArm:
    """A measured TM on the window, with the same forward/adjoint interface as ObjectArm."""

    def __init__(self, T, mask=None):
        self.T, self.mask = T, WIN if mask is None else mask

    def forward(self, X):
        O = np.zeros(X.shape[:-2] + (N, N), complex)
        O[..., self.mask] = X.reshape(X.shape[:-2] + (-1,)) @ self.T.T
        return O

    def adjoint(self, W):
        return (W[..., self.mask] @ np.conj(self.T)).reshape(W.shape[:-2] + (NC, NC))


def measure_tm(bench, rng, M, K=5, seed_arm=None, mask=None):
    """Measure a TM from M random patterns (K exposures each), keeping only the window. Returns (TMArm, X, Z)."""
    m = WIN if mask is None else mask
    X = random_patterns(rng, M)
    Z = np.concatenate([bench.read(X[i:i + 256], K)[:, m] for i in range(0, M, 256)])
    return TMArm(fit_tm(X, Z, window_fields(seed_arm or ObjectArm(), X, m)), m), X, Z


def window_fields(arm, X, mask=None):
    """arm.forward(X) inside the window, in blocks (memory)."""
    m = WIN if mask is None else mask
    return np.concatenate([arm.forward(X[i:i + 256])[:, m] for i in range(0, len(X), 256)])


# ----------------------------------------------------------------------------------------------------------------------
# Losses (each returns L and g = dL/dI), metrics and the holographic gradient
# ----------------------------------------------------------------------------------------------------------------------
def focus_loss(I):
    """L = -<I>_disk: maximise the intensity in one speckle grain (SI Note 3B)."""
    return -I[DISK].mean(), np.where(DISK, -1.0 / DISK.sum(), 0.0)


def image_loss(I, target, mask=None):
    """Energy-normalised image error L = <(s I - T)^2>, s = sum T / sum I, on the window (SI Eq. S13)."""
    m = WIN if mask is None else mask
    t, i = target.ravel(), I[m]
    s = t.sum() / i.sum()
    r = s * i - t
    g = np.zeros((N, N))
    g[m] = 2 * s / t.size * (r - np.sum(r * i) / i.sum())
    return np.mean(r**2), g


def grating_target(n=43, periods=8):
    """Sinusoidal grating on the n x n window (8 periods over 0.86 mm)."""
    j = (np.arange(n) + 0.5) / n
    return np.tile(0.5 * (1 + np.sin(2 * np.pi * periods * j)), (n, 1))


def resample_target(img, n=43):
    """Low-pass a bench target (128 x 128 camera pixels) to the n x n window: keep its lowest n x n frequencies."""
    F = np.fft.fftshift(np.fft.fft2(img))
    c = img.shape[0] // 2
    out = np.real(np.fft.ifft2(np.fft.ifftshift(F[c - n // 2:c + n // 2 + 1, c - n // 2:c + n // 2 + 1])))
    out = np.clip(out, 0, None)
    return out / out.max()


def psnr(I, target):
    """PSNR = 10 log10(1 / MSE) of the energy-normalised intensity against the target (peak 1)."""
    I, t = np.ravel(I), np.ravel(target)
    return 10 * np.log10(1 / np.mean((t.sum() / I.sum() * I - t)**2))


def object_gradient(X, W, arm):
    """Eq. (3): dL/dtheta = 2 Im[X_O^* T_O^H{W}], with the adjoint of `arm` (model or TM)."""
    return 2 * np.imag(np.conj(X) * arm.adjoint(W))


def grad_cosine(arm, true_arm, X, loss):
    """Cosine between the gradient through `arm` and the true gradient, for a noise-free output (SI Note 6)."""
    O = true_arm.forward(X)
    g_true = object_gradient(X, loss(np.abs(O)**2)[1] * O, true_arm)
    O = to_model_gauge(O, arm.forward(X), WIN)
    g = object_gradient(X, loss(np.abs(O)**2)[1] * O, arm)
    return float(np.sum(g * g_true) / (np.linalg.norm(g) * np.linalg.norm(g_true)))


class Adam:
    """Adam on the phases; the learning rate decays geometrically from lr to lr_final over `decay` steps."""

    def __init__(self, shape, lr=0.1, lr_final=0.03, decay=100):
        self.m, self.v, self.t = np.zeros(shape), np.zeros(shape), 0
        self.lr, self.lr_final, self.decay = lr, lr_final, decay

    def step(self, theta, grad):
        self.t += 1
        lr = self.lr * (self.lr_final / self.lr) ** (min(self.t - 1, self.decay) / self.decay)
        self.m = 0.9 * self.m + 0.1 * grad
        self.v = 0.999 * self.v + 0.001 * grad**2
        return theta - lr * (self.m / (1 - 0.9**self.t)) / (np.sqrt(self.v / (1 - 0.999**self.t)) + 1e-8)


def train_in_situ(bench, arm, loss, metric, iters=150, K=5, final_reads=10, crop=None, lr_final=0.03, update=None):
    """In situ training of the object arm (Methods; main Fig. 2a).

    Every iteration: display X_O = exp(j theta), measure O_hat (K exposures), rotate it into the model's gauge,
    form the error field W = g * O_hat from the MEASURED field, carry it back with the adjoint of `arm` and update
    theta with Adam. Only the adjoint comes from `arm`. The final pattern is read `final_reads` times (one exposure).
    The learning rate decays from 0.1 to lr_final over the first 100 iterations, or over the whole run if shorter.

    update(it, X, O_hat, arm), if given, runs after every step and may return a new arm: it moves a drifting medium
    and refits the model (see Drift).

    Returns theta, the metric per iteration, the model accuracy per iteration (correlation of the measured training
    field with the model's prediction on the window), the measured intensity per iteration (cropped, for the GIF) and
    the final single-exposure intensities.
    """
    theta, opt = np.zeros((NC, NC)), Adam((NC, NC), lr_final=lr_final, decay=min(100, iters))
    curve, accuracy, snaps = [], [], []
    for it in range(iters + 1):
        X = np.exp(1j * theta)
        O_hat = bench.read(X, K)
        pred = arm.forward(X)
        accuracy.append(corr(pred[WIN], O_hat[WIN]))
        O = to_model_gauge(O_hat, pred, WIN)
        I = np.abs(O)**2
        curve.append(metric(I))
        snaps.append(crop_to(I, WIN if crop is None else crop))
        if it == iters:
            break
        g = loss(I)[1]
        theta = opt.step(theta, object_gradient(X, g * O, arm))
        if update:
            arm = update(it, X, O_hat, arm)
    finals = [np.abs(bench.read(np.exp(1j * theta)))**2 for _ in range(final_reads)]
    return dict(theta=theta, curve=np.array(curve), accuracy=np.array(accuracy), snaps=np.array(snaps),
                finals=np.array(finals))


# ----------------------------------------------------------------------------------------------------------------------
# A medium that drifts during training, and refits of the model (Methods; main Fig. 4a-d; SI Note 5A)
# ----------------------------------------------------------------------------------------------------------------------
def decorrelation_shift(D, target=0.5):
    """Sideways shift (m) of the screen after which it keeps a correlation `target` with its starting state."""
    shifts = np.arange(0, 200e-6, 0.5e-6)
    c = np.array([corr(D, shift_screen(D, d)) for d in shifts])
    return shifts[np.argmax(c <= target)]


class Drift:
    """The hidden screen moves sideways, by an amount that grows linearly with the iteration, to `total` at the end.

    On the bench, a hidden phase ramp whose frequency grows with the iteration walks the illumination across the
    diffuser; its output tilt and shift are calibrated and removed, so only the change of the medium seen by the beam
    remains. Here that change is the screen moving under the beam.

    With refit=True the model is refitted every `every` iterations: `new` random probes (five exposures each) and the
    latest training frame, which is already measured, join the most recent fields (`keep` in all), and the screen is
    fitted again, warm-started from the current one (SI Note 5A: two probes and the training frame every five
    iterations). Use as train_in_situ(..., update=Drift(...)); call reset() afterwards.
    """

    def __init__(self, bench, total, iters, refit=False, probes=None, rng=None, every=5, new=2, keep=10):
        self.bench, self.D0, self.total, self.iters = bench, bench._true.D.copy(), total, iters
        self.refit, self.rng, self.every, self.new, self.keep = refit, rng, every, new, keep
        self.X, self.Y = (list(probes[0]), list(probes[1])) if probes else ([], [])
        self.refits = []                                    # iterations at which the model was refitted

    def __call__(self, it, X, O_hat, arm):
        self.bench._true.D = shift_screen(self.D0, self.total * (it + 1) / self.iters)
        if not self.refit or (it + 1) % self.every:
            return arm
        X_new = random_patterns(self.rng, self.new)
        self.X = (self.X + list(X_new) + [X])[-self.keep:]
        self.Y = (self.Y + list(self.bench.read(X_new, 5)) + [O_hat])[-self.keep:]
        self.refits.append(it + 1)
        return fit_screen(np.array(self.X), np.array(self.Y), arm=arm)

    def reset(self):
        self.bench._true.D = self.D0


def crop_to(I, mask):
    """The bounding box of a boolean mask, as a 2-D array."""
    r, c = np.nonzero(mask.any(1))[0], np.nonzero(mask.any(0))[0]
    return I[..., r[0]:r[-1] + 1, c[0]:c[-1] + 1]


# ----------------------------------------------------------------------------------------------------------------------
# GIF of a training run: measured intensity per iteration next to the metric curve
# ----------------------------------------------------------------------------------------------------------------------
def save_gif(path, images, curve, ylabel, title="", xlabel="iteration", every=1, fps=15, own_scale=True,
             overlay=None, captions=None, extra_curves=None):
    """Write an animated GIF: left, images[i] (measured |O_hat|^2); right, curve[:i+1] growing.

    own_scale   each frame on its own colour scale (True) or on the scale of the final frame (False)
    overlay     function(ax) drawing fixed marks on the image axis (detectors, focus disk)
    captions    optional text per frame under the image title
    extra_curves  dict {label: full curve} drawn in grey for comparison (e.g. the other adjoint)
    """
    images, curve = np.asarray(images), np.asarray(curve)
    fig, (ax_i, ax_c) = plt.subplots(1, 2, figsize=(6.4, 2.8), dpi=80, gridspec_kw=dict(width_ratios=[1, 1.35]))
    im = ax_i.imshow(images[0], cmap="inferno", vmin=0, vmax=images[-1].max())
    ax_i.set_xticks([]); ax_i.set_yticks([])
    if overlay:
        overlay(ax_i)
    for label, c in (extra_curves or {}).items():
        ax_c.plot(np.arange(len(c)), c, color="0.7", lw=1, label=label)
    line, = ax_c.plot([], [], color="C0", lw=1.5, label="this run")
    dot, = ax_c.plot([], [], "o", color="C0", ms=4)
    ax_c.set_xlim(0, len(curve) - 1)
    lo, hi = np.nanmin(curve), np.nanmax(curve)
    for c in (extra_curves or {}).values():
        lo, hi = min(lo, np.nanmin(c)), max(hi, np.nanmax(c))
    ax_c.set_ylim(lo - 0.05 * (hi - lo), hi + 0.08 * (hi - lo))
    ax_c.set_xlabel(xlabel); ax_c.set_ylabel(ylabel); ax_c.grid(alpha=0.3)
    if extra_curves:
        ax_c.legend(fontsize=7, loc="lower right")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    frames = []
    for i in list(range(0, len(images), every)) + [len(images) - 1] * 8:     # hold the last frame
        im.set_data(images[i])
        if own_scale:
            im.set_clim(0, images[i].max())
        line.set_data(np.arange(i + 1), curve[:i + 1])
        dot.set_data([i], [curve[i]])
        ax_i.set_title(f"{xlabel} {i}" + (f"\n{captions[i]}" if captions else ""), fontsize=8)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).quantize(128))
    plt.close(fig)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return path


# ----------------------------------------------------------------------------------------------------------------------
# Scaling with the number of SLM cells (SI Note 6, Fig. S9): one size, one seed, one task
# ----------------------------------------------------------------------------------------------------------------------
def scaling_run(nc, seed, task, tm=True, iters=150, gif_dir=None):
    """Repeat focusing ("focus", dim bench) or the grating ("grating", imaging bench) for nc x nc SLM cells.

    Adjoints: the physical model from 10 probes, and (if tm) TMs from N and 3N patterns, N = nc^2, where the TM from N
    uses the first N patterns of the same acquisition. Returns a dict of numbers; saves a GIF of the model run.
    """
    configure(nc)
    n_cells = nc * nc
    bench = Bench("focus" if task == "focus" else "imaging", seed=seed)
    rng = np.random.default_rng(7 + 100 * seed)
    X_held = random_patterns(rng, 40)
    Y_held = bench.read(X_held, 10)
    X_probe = random_patterns(rng, 10)
    arms = {"model": fit_screen(X_probe, bench.read(X_probe, 5))}
    if tm:
        arms["tm_3N"], X_tm, Z_tm = measure_tm(bench, rng, 3 * n_cells, seed_arm=arms["model"])
        Z_seed = window_fields(arms["model"], X_tm[:n_cells])
        arms["tm_N"] = TMArm(fit_tm(X_tm[:n_cells], Z_tm[:n_cells], Z_seed))
        del X_tm, Z_tm
    if task == "focus":
        loss = focus_loss
        I0 = np.mean(np.abs(bench.read(random_patterns(rng, 320))[:, DISK])**2)
        metric = lambda I: I[DISK].mean() / I0
    else:
        target = grating_target()
        loss = lambda I: image_loss(I, target)
        metric = lambda I: psnr(I[WIN], target)
    rc = np.random.default_rng(1000 + seed)
    X_cos = random_patterns(rc, 20)
    out = dict(nc=nc, n_cells=n_cells, seed=seed, task=task, patterns={"model": 10, "tm_N": n_cells, "tm_3N": 3 * n_cells})
    for name, arm in arms.items():
        r = train_in_situ(bench, arm, loss, metric, iters=iters)
        final = [metric(I) for I in r["finals"]]
        out[name] = dict(accuracy=model_accuracy(arm, X_held, Y_held, WIN),
                         cos=float(np.mean([grad_cosine(arm, bench._true, x, loss) for x in X_cos])),
                         final=float(np.mean(final)), final_sem=float(np.std(final) / np.sqrt(len(final))),
                         curve=r["curve"].tolist())
        if name == "model" and gif_dir:
            label = "enhancement η" if task == "focus" else "PSNR (dB)"
            save_gif(f"{gif_dir}/scaling_{task}_{nc}x{nc}.gif", r["snaps"], r["curve"], label,
                     f"{task}, {nc}x{nc} SLM cells, model from 10 probes", own_scale=(task == "focus"))
    return out
