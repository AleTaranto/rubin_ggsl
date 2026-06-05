"""Generate perturbed Einstein ring plots (analytic SIS + point subhalos).

This script produces a 2x3 figure similar to the earlier test_perturbed_ring
visualizations for quick inspection and debugging.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os


def deflection_sis(x, y, theta_E, x0=0.0, y0=0.0):
    dx = x - x0
    dy = y - y0
    r = np.hypot(dx, dy)
    r_safe = np.where(r == 0, 1e-8, r)
    ax = theta_E * (dx / r_safe)
    ay = theta_E * (dy / r_safe)
    return ax, ay


def deflection_point(x, y, x0, y0, alpha0):
    dx = x - x0
    dy = y - y0
    rsq = dx * dx + dy * dy
    rsq_safe = np.where(rsq == 0, 1e-8, rsq)
    ax = alpha0 * dx / rsq_safe
    ay = alpha0 * dy / rsq_safe
    return ax, ay


def sersic_profile(X, Y, x0, y0, re, n=2.0, Ie=1.0):
    # approximate Sersic using exponential for n=1 or general n via b_n approximation
    dx = X - x0
    dy = Y - y0
    R = np.hypot(dx, dy)
    b_n = 2 * n - 0.324
    profile = Ie * np.exp(-b_n * ((R / re)**(1.0 / n) - 1.0))
    return profile


def make_perturbed_ring_plot(outpath='perturbed_ring_example.png'):
    # Grid
    npix = 512
    fov = 10.0  # arcsec
    pixel_scale = (2 * fov) / npix
    coords = np.linspace(-fov, fov, npix)
    X, Y = np.meshgrid(coords, coords)

    # Main SIS
    theta_E = 1.0  # arcsec
    ax_sis, ay_sis = deflection_sis(X, Y, theta_E)

    # Subhalos: random positions and strengths
    rng = np.random.RandomState(42)
    n_sub = 12
    rmax = 4.0
    thetas = rng.uniform(0, 2*np.pi, n_sub)
    rs = rmax * np.sqrt(rng.uniform(size=n_sub))
    xs = rs * np.cos(thetas)
    ys = rs * np.sin(thetas)
    masses = 10**(rng.uniform(9.0, 11.0, n_sub))

    ax_sub = np.zeros_like(X)
    ay_sub = np.zeros_like(Y)
    for i in range(n_sub):
        # alpha0 scaled with mass
        alpha0 = 0.05 * (masses[i] / 1e10)**(1/3)
        ax_i, ay_i = deflection_point(X, Y, xs[i], ys[i], alpha0)
        ax_sub += ax_i
        ay_sub += ay_i

    ax_total = ax_sis + ax_sub
    ay_total = ay_sis + ay_sub

    # Source: small Sersic offset to produce arc
    src_x, src_y = 0.5, 0.0
    re = 0.2
    source = sersic_profile(X, Y, src_x, src_y, re, n=2.0, Ie=1.0)

    # Perfect image: sample source at beta = theta - alpha
    beta_x = X - ax_total
    beta_y = Y - ay_total
    perfect = sersic_profile(beta_x, beta_y, src_x, src_y, re, n=2.0, Ie=1.0)

    # Arcs only: threshold perfect image
    arcs_only = np.copy(perfect)
    arcs_only[arcs_only < 0.1 * arcs_only.max()] = 0

    # Convergence (approx): SIS kappa ~ theta_E/(2r); add gaussian blobs for subhalos
    r = np.hypot(X, Y)
    kappa = np.where(r == 0, theta_E / (2 * 1e-3), theta_E / (2 * r))
    for i in range(n_sub):
        kappa += gaussian_filter(np.exp(-((X - xs[i])**2 + (Y - ys[i])**2) / (2 * (0.3)**2)), sigma=1)

    # Noisy image (no PSF)
    noise = rng.normal(0, 0.02 * perfect.std(), perfect.shape)
    noisy = perfect + noise

    # PSF-convolved + noise
    psf = gaussian_filter(perfect, sigma=2)
    psf_noisy = psf + rng.normal(0, 0.02 * psf.std(), psf.shape)

    # DM subhalo positions map
    submap = np.zeros_like(X)
    for i in range(n_sub):
        rr = np.hypot(X - xs[i], Y - ys[i])
        submap += np.exp(-0.5 * (rr / 0.15)**2)

    # Create figure 2x3
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    im0 = axes[0, 0].imshow(perfect, origin='lower', cmap='inferno', extent=[-fov, fov, -fov, fov])
    axes[0, 0].set_title('Lensed image (noiseless)')
    fig.colorbar(im0, ax=axes[0,0], fraction=0.046)

    im1 = axes[0, 1].imshow(arcs_only, origin='lower', cmap='inferno', extent=[-fov, fov, -fov, fov])
    axes[0, 1].set_title('Lensed arcs only')
    fig.colorbar(im1, ax=axes[0,1], fraction=0.046)

    im2 = axes[0, 2].imshow(kappa, origin='lower', cmap='viridis', extent=[-fov, fov, -fov, fov])
    axes[0, 2].set_title('Convergence (kappa)')
    fig.colorbar(im2, ax=axes[0,2], fraction=0.046)

    im3 = axes[1, 0].imshow(noisy, origin='lower', cmap='inferno', extent=[-fov, fov, -fov, fov])
    axes[1, 0].set_title('Final image (noise, no PSF)')
    fig.colorbar(im3, ax=axes[1,0], fraction=0.046)

    im4 = axes[1, 1].imshow(psf_noisy, origin='lower', cmap='inferno', extent=[-fov, fov, -fov, fov])
    axes[1, 1].set_title('PSF-convolved + noise')
    fig.colorbar(im4, ax=axes[1,1], fraction=0.046)

    im5 = axes[1, 2].imshow(submap, origin='lower', cmap='plasma', extent=[-fov, fov, -fov, fov])
    axes[1, 2].scatter(xs, ys, c='white', marker='+')
    axes[1, 2].set_title('DM subhalo positions')
    fig.colorbar(im5, ax=axes[1,2], fraction=0.046)

    for ax in axes.ravel():
        ax.set_xlabel('x [arcsec]')
        ax.set_ylabel('y [arcsec]')

    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()


if __name__ == '__main__':
    make_perturbed_ring_plot(outpath='pylenslib_outputs/perturbed_ring_example.png')
