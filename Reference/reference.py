"""Digital twin of the bench: the reference arm. Shared by the notebooks in this folder; needs twin.py (Object folder).

Notation follows the manuscript and the Supplementary Information (SI):

    R = T_R{E}                    reference field on the camera from the 100 x 100 reference pattern E      SI Eq. (S14)
    I = |O + R|^2                 off-axis hologram; the reference arrives tilted, on the carrier f_c        Eq. (1)
    s_c = Re[e^{-j phi} sum_x M_c(x) e^{j 2 pi f_c . x} I(x)]  ~  Re[e^{-j phi} sum_{x in M_c} O R^*]          Eq. (7)
    dL/dE^* = T_R^H{W_R},  W_R = 1/2 O_hat * sum_c g_c M_c,  g_c = dL/ds_c                                  Eq. (3), SI Eq. (S7)

The object field does not depend on E, so the fields O_hat of a data set are measured once with a flat reference and
E is trained offline on them (SI Note 4A). The twin hides the diffuser and the exact reference arm from the methods.
"""
import numpy as np
import scipy.fft as sfft
import twin as tw
from twin import fft2, ifft2

tw.configure(32)

# ----------------------------------------------------------------------------------------------------------------------
# Geometry of the reference arm and of the camera
# ----------------------------------------------------------------------------------------------------------------------
NE = 100                 # reference pattern: 100 x 100 SLM cells of 40 um (4 mm)
NRG = 224                # grid of the reference arm (20 um pixels)
BR = 11e3                # reference iris (cycles/m)
C0 = 0.415               # SLM amplitude response: written field = C0 exp(j arg E) + (1 - C0) E (|E| <= 1)
UP = 3                   # camera pixels per grid pixel (6.7 um): a structured reference needs more pixels (SI Note 1B)
NU = tw.N * UP           # camera pixels per side (576)
FC = 168                 # carrier bin per axis on the camera (43.7 lp/mm per axis)
SB = 24e3                # radius of the demodulated sideband O R^* (cycles/m): object band + reference band
W90 = slice(tw.N // 2 - 45, tw.N // 2 + 45)          # 90 x 90 readout window (1.8 mm) on the field grid
WINDOW = np.zeros((tw.N, tw.N), bool)
WINDOW[W90, W90] = True

_f = sfft.fftfreq(NRG, tw.DX)
_FXR, _FYR = np.meshgrid(_f, _f)
_F2R = _FXR**2 + _FYR**2
_S0, _C0 = (NRG - NE * tw.CELL) // 2, (NRG - tw.N) // 2       # reference cells on the reference grid; camera crop


def zernike(FX, FY, radius, a):
    u, v = FX / radius, FY / radius
    r2 = u**2 + v**2
    return (a.get("defocus", 0) * (2 * r2 - 1) + a.get("astig", 0) * 2 * u * v + a.get("coma", 0) * (3 * r2 - 2) * u
            + a.get("spherical", 0) * (6 * r2**2 - 6 * r2 + 1))


class ReferenceArm:
    """R = T_R{E}: illumination (Gaussian beam with a phase), the SLM cells, a pupil (iris, defocus, aberrations) and a
    shift onto the camera grid (SI Eq. S14). The model and the hidden true arm differ in these few parameters."""

    def __init__(self, z=40e-3, aberration=None, shift=(0.0, 0.0), scale=1.0):
        c = (np.arange(NE) - (NE - 1) / 2) * tw.CELL * tw.DX
        CY, CX = np.meshgrid(c, c, indexing="ij")
        h = NE * tw.CELL * tw.DX / 2
        self.beam = scale * np.exp(-((CY + 0.8e-3)**2 + (CX + 0.7e-3)**2) / 4.1e-3**2) \
            * np.exp(1.5j * ((CY / h)**2 + 0.6 * (CX / h)**2))
        self.P = (_F2R <= BR**2) * np.exp(-1j * np.pi * tw.LAM * z * _F2R) \
            * np.exp(1j * zernike(_FXR, _FYR, BR, aberration or {})) \
            * np.exp(-2j * np.pi * (_FYR * shift[0] + _FXR * shift[1]) * tw.DX)

    def forward(self, E):
        u = np.zeros(E.shape[:-2] + (NRG, NRG), complex)
        u[..., _S0:_S0 + NE * tw.CELL, _S0:_S0 + NE * tw.CELL] = np.repeat(np.repeat(self.beam * E, tw.CELL, -1), tw.CELL, -2)
        return ifft2(fft2(u) * self.P)[..., _C0:_C0 + tw.N, _C0:_C0 + tw.N]

    def adjoint(self, R):
        v = np.zeros(R.shape[:-2] + (NRG, NRG), complex)
        v[..., _C0:_C0 + tw.N, _C0:_C0 + tw.N] = R
        v = ifft2(fft2(v) * np.conj(self.P))[..., _S0:_S0 + NE * tw.CELL, _S0:_S0 + NE * tw.CELL]
        return np.conj(self.beam) * v.reshape(v.shape[:-2] + (NE, tw.CELL, NE, tw.CELL)).sum(axis=(-3, -1))


def slm_response(E):
    """The field the SLM writes for a requested complex value E (|E| <= 1): a cell cannot be switched off."""
    return C0 * np.exp(1j * np.angle(E)) + (1 - C0) * E


FLAT = np.ones((NE, NE), complex)                     # the untrained, flat reference
REF_COUNTS = 300                                      # flat reference on the readout window (counts), as on the bench
REF_SCALE = np.sqrt(REF_COUNTS / np.mean(np.abs(ReferenceArm().forward(FLAT)[W90, W90])**2))
MODEL = ReferenceArm(scale=REF_SCALE)                 # the calibrated model of the reference arm (what the methods know)
# The hidden true reference arm differs from the model by a few geometry errors, sized so that the detector offsets they
# leave (about 0.25 score spreads) are those of the bench's Basic session. The model then predicts measured interference
# with an accuracy of about 0.97; the bench's check probes read 0.84-0.86, where measurement noise takes a larger part.
TRUE_REFERENCE = dict(z=40.75e-3, aberration=dict(astig=0.126, coma=0.105, spherical=0.066), shift=(0.15, -0.105))


# ----------------------------------------------------------------------------------------------------------------------
# Detectors: square windows on the readout window, and their fringe-matched masks on the raw frame, Eq. (7)
# ----------------------------------------------------------------------------------------------------------------------
class Detectors:
    """Detector windows M_c on the 90 x 90 readout window (corners of size x size boxes, field pixels).

    soft=False: hard 0/1 boxes. soft=True: the bench's super-Gaussian detectors centred on the same boxes,
    M_c = exp(-((dy/s)^(2 order) + (dx/s)^(2 order))) with s = (size/2)(1 + softness), softness 0.35, order 4.
    pixels  the window pixels where any detector weighs more than 1e-4, and to_box (n_pixels, n_detectors) their weights
    masks   M_c(x) exp(j 2 pi f_c . x) on the raw camera frame: the fringe-matched read of Eq. (7)
    """

    def __init__(self, corners, size=30, soft=False, softness=0.35, order=4):
        self.corners, self.size = corners, size
        self.window = np.zeros((len(corners), 90, 90))
        yy, xx = np.mgrid[0:90, 0:90].astype(float)
        sig = 0.5 * size * (1 + softness)
        for k, (r, c) in enumerate(corners):
            if soft:
                cy, cx = r + (size - 1) / 2, c + (size - 1) / 2
                self.window[k] = np.exp(-(((yy - cy) / sig)**(2 * order) + ((xx - cx) / sig)**(2 * order)))
            else:
                self.window[k, r:r + size, c:c + size] = 1
        flat = self.window.reshape(len(corners), -1)
        self.pixels = np.nonzero(flat.max(0) > 1e-4)[0]
        self.to_box = flat[:, self.pixels].T
        on_grid = np.zeros((len(corners), tw.N, tw.N))
        on_grid[:, W90, W90] = self.window
        on_camera = np.repeat(np.repeat(on_grid, UP, -1), UP, -2)
        y, x = np.mgrid[0:NU, 0:NU]
        self.masks = (on_camera * np.exp(2j * np.pi * FC * (x + y) / NU)).reshape(len(corners), -1).astype(np.complex64)

    def on_pixels(self, field):
        """Values of fields (..., N, N) at the detector pixels -> (..., n_pixels)."""
        return field[..., W90, W90].reshape(field.shape[:-2] + (-1,))[..., self.pixels]

    def scores(self, O_pix, R_pix):
        """Model scores s_c = sum_{x in M_c} Re(O R^*), from fields at the detector pixels."""
        return np.real(O_pix * np.conj(R_pix)) @ self.to_box


# ----------------------------------------------------------------------------------------------------------------------
# The twin bench with both arms
# ----------------------------------------------------------------------------------------------------------------------
class Bench:
    """Write data phases on the object arm and a pattern E on the reference arm; get raw camera frames.

    Set like the bench (Defend the Center run of 2026-10-25): flat reference about 300 counts and object about 14 counts
    per pixel, an interference visibility of 0.25 (only part of the light interferes), and each read the mean of five
    exposures (sig_cell adds a wavefront jitter per object cell and read; off by default). Hidden from the methods: the diffuser and lens aberrations of the object arm (as in twin.py) and the exact
    reference arm (TRUE_REFERENCE). Per exposure: the phase between the arms after pilot tracking (5 degrees rms) and the
    amplitude (5 % rms); shot and read noise, 12-bit camera.
    """

    def __init__(self, seed=0, obj_counts=14, visibility=0.25, exposures=5, phase_deg=5.0, amp=0.05, read_noise=3.0,
                 sig_cell=0.0):
        g = np.random.default_rng(seed)
        self.rng = np.random.default_rng(seed + 1)
        self._obj = tw.ObjectArm(tw.random_screen(g), tw.ABERRATION)
        self._ref = ReferenceArm(scale=REF_SCALE, **TRUE_REFERENCE)
        x = np.exp(2j * np.pi * g.random((5, tw.NC, tw.NC)))
        self.so = np.sqrt(obj_counts / np.mean(np.abs(self._obj.forward(x)[:, W90, W90])**2))
        self.visibility, self.exposures, self.sig_cell = visibility, exposures, sig_cell
        self.phase, self.amp, self.read_noise = np.deg2rad(phase_deg), amp, read_noise
        y, xx = np.mgrid[0:NU, 0:NU]
        self.carrier = np.exp(2j * np.pi * FC * (xx + y) / NU).astype(np.complex64)
        self.R0_model = MODEL.forward(FLAT)
        self.reads = 0

    def frame(self, phases, E):
        """One read per input: the mean of `exposures` raw camera frames (counts, NU x NU) for data phases
        (n, NC, NC) and one reference pattern E. A fraction `visibility` of the object light interferes."""
        n = len(phases)
        jitter = self.sig_cell * self.rng.standard_normal(phases.shape)              # wavefront jitter of this read
        O = tw.to_camera(self._obj.forward(np.exp(1j * (phases + jitter))) * self.so, UP)
        R = tw.to_camera(self._ref.forward(slm_response(E)), UP) * self.carrier
        I = 0
        for _ in range(self.exposures):
            k = ((1 + self.amp * self.rng.standard_normal(n)) * np.exp(1j * self.phase * self.rng.standard_normal(n)))[:, None, None]
            Ie = np.abs(np.sqrt(self.visibility) * k * O + R)**2 + (1 - self.visibility) * np.abs(k * O)**2
            Ie = Ie + np.sqrt(Ie + self.read_noise**2) * self.rng.standard_normal(Ie.shape)
            I = I + np.clip(np.round(Ie), 0, 4095) / self.exposures
        self.reads += n
        return I.astype(np.float32)

    def read(self, phases, E, detectors, chunk=32):
        """One read per input: the fringe-matched scores on the raw frame (Eq. 7) and, from the same frames, the scores
        of the demodulated field d = O R^*, summed over the detector windows."""
        raw, demod = [], []
        for i in range(0, len(phases), chunk):
            I = self.frame(phases[i:i + chunk], E)
            raw.append(np.real(I.reshape(len(I), -1) @ detectors.masks.T) / UP**2)
            d = demodulate(I)[..., W90, W90].reshape(len(I), -1)
            demod.append(np.real(d) @ detectors.window.reshape(len(detectors.corners), -1).T)
        return np.concatenate(raw), np.concatenate(demod)

    def acquire(self, phases, chunk=32, detectors=None):
        """Object fields O_hat measured with the flat reference: d R0 / |R0|^2 with the MODEL's flat reference R0.
        Returns full fields (n, N, N), or only the detector pixels (n, n_pixels) if detectors are given."""
        R0 = self.R0_model
        out = []
        for i in range(0, len(phases), chunk):
            d = demodulate(self.frame(phases[i:i + chunk], FLAT))
            O = d * R0 / (np.abs(R0)**2 + 1e-2 * np.max(np.abs(R0)**2))
            out.append(O if detectors is None else detectors.on_pixels(O).astype(np.complex64))
        return np.concatenate(out)


def demodulate(I):
    """Off-axis demodulation: crop the sideband O R^* around -f_c, shift it to zero -> d = O R^* on the field grid."""
    S = sfft.fftshift(fft2(I), axes=(-2, -1))
    c = NU // 2 - FC
    side = S[..., c - tw.N // 2:c + tw.N // 2, c - tw.N // 2:c + tw.N // 2] * sfft.fftshift(tw.F2 <= SB**2)
    return ifft2(sfft.ifftshift(side, axes=(-2, -1))) / UP


# ----------------------------------------------------------------------------------------------------------------------
# Offline training of the reference (SI Note 4A)
# ----------------------------------------------------------------------------------------------------------------------
def weighted_ce(s, y, w):
    """Class-weighted cross-entropy of softmax(s): loss and dL/ds."""
    p = np.exp(s - s.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    wn = w[y] / w[y].sum()
    return -np.sum(wn * np.log(p[np.arange(len(y)), y])), wn[:, None] * (p - np.eye(s.shape[1])[y])


def tv_grad(E, eps=1e-6):
    """Gradient (with respect to Re E and Im E) of the total variation of E."""
    g = np.zeros_like(E)
    for ax in (0, 1):
        d = np.diff(E, axis=ax)
        u = d / np.sqrt(np.abs(d)**2 + eps)
        if ax == 0:
            g[1:] += u; g[:-1] -= u
        else:
            g[:, 1:] += u; g[:, :-1] -= u
    return g


def train_reference(O, y, O_val, y_val, det, epochs=30, batch=50, lr=0.05, tv=0.02, seed=0, E0=None, sigma=None):
    """Train E offline on measured fields O (n, n_pixels) at the detector pixels.

    Loss: class-weighted cross-entropy of the scores (divided by their spread at E = FLAT) plus a total-variation
    penalty on E. Gradient: dL/dE^* = T_R^H{W_R} with W_R = 1/2 sum_n g_n O_n, through the reference MODEL only.
    E starts flat; Adam on Re E and Im E with a cosine learning-rate schedule (as on the bench), then the projection
    |E| <= 1. The epoch with the best validation accuracy predicted by the model is kept. Returns E and the training
    history (loss, validation accuracy per epoch). E0 and sigma (the score scale) continue a training from a given
    reference, as in the joint training of both arms.
    """
    n_cls = det.to_box.shape[1]
    w = len(y) / (n_cls * np.bincount(y, minlength=n_cls))
    sigma = sigma or np.std(det.scores(O, det.on_pixels(MODEL.forward(FLAT))))
    re, im = tw.Adam((NE, NE), lr, lr), tw.Adam((NE, NE), lr, lr)
    E, rng = (FLAT if E0 is None else E0).copy(), np.random.default_rng(seed)
    history, best = [], (-1, None)
    tv_weight = None
    for epoch in range(epochs):
        re.lr = re.lr_final = im.lr = im.lr_final = 0.5 * lr * (1 + np.cos(np.pi * epoch / epochs))
        order, losses = rng.permutation(len(y)), []
        for b in range(0, len(y), batch):
            ids = order[b:b + batch]
            R = det.on_pixels(MODEL.forward(E))
            L, ds = weighted_ce(det.scores(O[ids], R) / sigma, y[ids], w)
            w_pix = np.zeros(90 * 90, complex)
            w_pix[det.pixels] = 0.5 * np.sum((ds / sigma @ det.to_box.T) * O[ids], axis=0)          # dL/dR^*
            W = np.zeros((tw.N, tw.N), complex)
            W[W90, W90] = w_pix.reshape(90, 90)
            g = 2 * MODEL.adjoint(W)                                                                 # dL/dRe E + j dL/dIm E
            tv_weight = tv_weight or tv * np.sqrt(np.mean(np.abs(g)**2))
            g = g + tv_weight * tv_grad(E)
            E = re.step(E.real, g.real) + 1j * im.step(E.imag, g.imag)
            E = E / np.maximum(1, np.abs(E))                                                         # |E| <= 1
            losses.append(L)
        val = np.mean(det.scores(O_val, det.on_pixels(MODEL.forward(E))).argmax(1) == y_val)
        history.append((np.mean(losses), val))
        if val > best[0]:
            best = (val, E.copy())
    return best[1], np.array(history)


def balanced_accuracy(pred, y):
    """Mean recall over the classes."""
    return np.mean([np.mean(pred[y == k] == k) for k in np.unique(y)])
