import numpy as np
from astropy.cosmology import Planck18 as cosmo
from pyLensLib.compositeModel import compositeModel
from pyLensLib.sie import sie
from pyLensLib.nfwell import nfwell


def compute_mu_for_model(model, x_range=(-5, 5), y_range=(-5, 5), npix=200):
    x = np.linspace(x_range[0], x_range[1], npix)
    y = np.linspace(y_range[0], y_range[1], npix)
    X, Y = np.meshgrid(x, y)
    kappa = model.kappa(X, Y)
    g1, g2 = model.gamma(X, Y)
    denom = (1.0 - kappa) ** 2 - (g1 ** 2 + g2 ** 2)
    mu = np.where(denom != 0, 1.0 / denom, np.nan)
    return mu


def test_sie_produces_high_magnification():
    # Create an isothermal ellipsoid (SIE) with reasonable sigma
    zl = 0.3
    zs = 1.0
    model = sie(co=cosmo, sigma0=220.0, q=1.0, pa=0.0, x1=0.0, x2=0.0, zl=zl, zs=zs)
    comp = compositeModel(co=cosmo, models=[model])
    comp.zl = zl
    comp.zs = zs
    mu = compute_mu_for_model(comp, x_range=(-3, 3), y_range=(-3, 3), npix=200)
    mu_max = np.nanmax(mu)
    mu_mean = np.nanmean(mu)
    assert mu_max > 10.0, f"Expected high magnification, got mu_max={mu_max}"
    assert mu_mean > 0.5, f"Unexpectedly low mean magnification mu_mean={mu_mean}"


def test_nfwell_subhalo_magnification():
    # Small concentrated NFW should produce localized magnification peaks
    zl = 0.3
    zs = 1.0
    model = nfwell(co=cosmo, mass=1e11, conc=15.0, x1=0.2, x2=0.0, zl=zl, zs=zs)
    comp = compositeModel(co=cosmo, models=[model])
    comp.zl = zl
    comp.zs = zs
    mu = compute_mu_for_model(comp, x_range=(-1, 1), y_range=(-1, 1), npix=200)
    mu_max = np.nanmax(mu)
    assert mu_max > 2.0, f"Expected some magnification from subhalo, got mu_max={mu_max}"
