#!/usr/bin/env python
"""Solve the two-point boundary value problem for Gamma_2(tau) in a dense wind.

    (1/3) d^2 Gamma_2 / dtau^2 = (kappa_ff*(3.4 eV) / kappa_T) (Gamma_2 - Gamma_2*)
    Gamma_2'(0) = sqrt(3) Gamma_2(0),   Gamma_2(tau_in) = Gamma_2*(tau_in)

Assumptions of this first step: n_2 sigma_2 is neglected in d tau, the gas is
fully ionized (n_e = rho / m_p), and rho = rho16 (r / 1e16 cm)^-alpha.  Then
tau(r) = kappa_T rho r / (alpha - 1) is the Thomson depth measured inward from
infinity, so tau = 0 is the outer surface and tau_in = tau(r_in).
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.integrate import solve_bvp

from dense_spectrum import (
    BOLTZMANN_ERG_K,
    CM_PER_REFERENCE_RADIUS,
    HYDROGEN_MASS,
    PLANCK_ERG_S,
    STEFAN_BOLTZMANN,
)

C_LIGHT = 2.99792458e10
EV_ERG = 1.602176634e-12
EPS_2_ERG = 3.4 * EV_ERG
DEFAULT_KAPPA_T = 0.34
DEFAULT_SIGMA_2 = 1.38e-17


def tau_of_radius(r, alpha, rho16, kappa_t):
    """Thomson depth from infinity to r."""
    rho = rho16 * (r / CM_PER_REFERENCE_RADIUS) ** (-alpha)
    return kappa_t * rho * r / (alpha - 1.0)


def radius_of_tau(tau, alpha, rho16, kappa_t):
    """Invert tau(r); tau = 0 maps to r = infinity."""
    tau = np.asarray(tau, dtype=float)
    with np.errstate(divide="ignore"):
        x = tau * (alpha - 1.0) / (kappa_t * rho16 * CM_PER_REFERENCE_RADIUS)
        return CM_PER_REFERENCE_RADIUS * x ** (-1.0 / (alpha - 1.0))


def local_state(tau, luminosity, alpha, rho16, kappa_t):
    """Return r, rho, T_e for each tau; T_e = 0 and rho = 0 at tau = 0."""
    tau = np.asarray(tau, dtype=float)
    pos = tau > 0.0
    r = np.full(tau.shape, np.inf)
    rho = np.zeros(tau.shape)
    temp = np.zeros(tau.shape)
    r[pos] = radius_of_tau(tau[pos], alpha, rho16, kappa_t)
    rho[pos] = rho16 * (r[pos] / CM_PER_REFERENCE_RADIUS) ** (-alpha)
    pref = 0.75 * (alpha - 1.0) / (alpha + 1.0)
    temp[pos] = (luminosity * tau[pos] / (4.0 * np.pi * r[pos] ** 2 * STEFAN_BOLTZMANN) * pref) ** 0.25
    return r, rho, temp


def kappa_ff_over_kappa_t(rho, temp, kappa_t):
    """kappa_ff*(3.4 eV) / (kappa_T rho), with n_e = rho / m_p."""
    nu = EPS_2_ERG / PLANCK_ERG_S
    ratio = np.zeros(np.shape(rho))
    ok = (temp > 0.0) & (rho > 0.0)
    ne = rho[ok] / HYDROGEN_MASS
    x = EPS_2_ERG / (BOLTZMANN_ERG_K * temp[ok])
    alpha_ff = 3.7e8 * temp[ok] ** -0.5 * ne**2 * nu**-3 * -np.expm1(-x)
    ratio[ok] = alpha_ff / (kappa_t * rho[ok])
    return ratio


def gamma_star_prefactor(sigma_2):
    """8 pi sigma_2 (3.4 eV)^3 / (c^2 h^3); Gamma_2* = this * (kT / 3.4 eV) * exp(-3.4 eV / kT)."""
    return 8.0 * np.pi * sigma_2 * EPS_2_ERG**3 / (C_LIGHT**2 * PLANCK_ERG_S**3)


def solve_gamma2(
    luminosity, rho16, alpha, rin, kappa_t=DEFAULT_KAPPA_T, sigma_2=DEFAULT_SIGMA_2, nmesh=200, tol=1.0e-6,
    tau_match=None, gamma_match=None,
):
    """Return a dict of arrays.

    By default the surface condition at tau = 0 is used.  If tau_match and gamma_match are
    given, the problem is solved on [tau_match, tau_in] with Gamma_2(tau_match) = gamma_match.
    """
    if alpha <= 1.0:
        raise ValueError("alpha must be > 1")
    tau_in = tau_of_radius(rin, alpha, rho16, kappa_t)
    if (tau_match is None) != (gamma_match is None):
        raise ValueError("tau_match and gamma_match must be given together")
    if tau_match is not None and not 0.0 < tau_match < tau_in:
        raise ValueError(f"tau_match must lie in (0, tau_in = {tau_in:.4g})")
    _, _, t_in = local_state(np.array([tau_in]), luminosity, alpha, rho16, kappa_t)
    t_in = float(t_in[0])
    x_in = EPS_2_ERG / (BOLTZMANN_ERG_K * t_in)
    gstar_in = gamma_star_prefactor(sigma_2) * np.exp(-x_in) / x_in

    # Work in units of Gamma_2*(tau_in), since Gamma_2* is largest at the inner edge.
    def gstar_scaled(temp):
        out = np.zeros(temp.shape)
        ok = temp > 0.0
        out[ok] = temp[ok] / t_in * np.exp(-EPS_2_ERG / BOLTZMANN_ERG_K * (1.0 / temp[ok] - 1.0 / t_in))
        return out

    def fun(tau, y):
        _, rho, temp = local_state(tau, luminosity, alpha, rho16, kappa_t)
        k = kappa_ff_over_kappa_t(rho, temp, kappa_t)
        return np.vstack([y[1], 3.0 * k * (y[0] - gstar_scaled(temp))])

    if tau_match is None:
        def bc(ya, yb):
            return np.array([ya[1] - np.sqrt(3.0) * ya[0], yb[0] - 1.0])

        tau = np.concatenate([[0.0], np.geomspace(1.0e-6 * tau_in, tau_in, 400)])
        mesh = np.concatenate([[0.0], np.geomspace(1.0e-4 * tau_in, tau_in, nmesh)])
    else:
        g_match = gamma_match / gstar_in

        def bc(ya, yb):
            return np.array([ya[0] - g_match, yb[0] - 1.0])

        tau = np.geomspace(tau_match, tau_in, 400)
        mesh = np.geomspace(tau_match, tau_in, nmesh)

    guess = np.vstack([np.ones_like(mesh), np.zeros_like(mesh)])
    sol = solve_bvp(fun, bc, mesh, guess, tol=tol, max_nodes=200000)
    if not sol.success:
        raise RuntimeError("solve_bvp failed: " + sol.message)
    g = sol.sol(tau)[0]
    nodes = len(sol.x)

    r, rho, temp = local_state(tau, luminosity, alpha, rho16, kappa_t)
    gs = gstar_scaled(temp)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ratio = np.where(gs > 1.0e-300, g / gs, np.nan)
    return {
        "tau": tau,
        "radius": r,
        "density": rho,
        "temperature": temp,
        "gamma2": g * gstar_in,
        "gamma2_star": gs * gstar_in,
        "ratio": ratio,
        "tau_in": tau_in,
        "t_in": t_in,
        "nodes": nodes,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--luminosity", type=float, required=True, help="L in erg/s")
    p.add_argument("--rho16", type=float, required=True, help="density at 1e16 cm in g/cm^3")
    p.add_argument("--alpha", type=float, required=True, help="density power-law index (> 1)")
    p.add_argument("--rin", type=float, required=True, help="inner radius in cm")
    p.add_argument("--kappa-t", type=float, default=DEFAULT_KAPPA_T, help="Thomson opacity in cm^2/g")
    p.add_argument("--sigma2", type=float, default=DEFAULT_SIGMA_2, help="n=2 threshold photoionization cross section in cm^2")
    p.add_argument("--tau-match", type=float, default=None, help="tau at which Gamma_2 is fixed (replaces the tau = 0 condition)")
    p.add_argument("--gamma-match", type=float, default=None, help="value of Gamma_2 at --tau-match")
    p.add_argument("--out", default="gamma2_ode.txt", help="output table")
    p.add_argument("--plot", default=None, help="optional PNG of Gamma_2 and Gamma_2* versus tau")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    res = solve_gamma2(
        args.luminosity, args.rho16, args.alpha, args.rin, args.kappa_t, args.sigma2,
        tau_match=args.tau_match, gamma_match=args.gamma_match,
    )
    header = "tau radius_cm density_g_cm3 T_e_K Gamma2 Gamma2_star Gamma2/Gamma2_star"
    cols = [res[k] for k in ("tau", "radius", "density", "temperature", "gamma2", "gamma2_star", "ratio")]
    np.savetxt(args.out, np.column_stack(cols), header=header, fmt="%.6e")
    print(f"tau_in = {res['tau_in']:.4g}, T_e(r_in) = {res['t_in']:.4g} K, mesh nodes = {res['nodes']}")
    print(f"Gamma2(tau={res['tau'][0]:.4g}) = {res['gamma2'][0]:.4e}, Gamma2*(tau_in) = {res['gamma2_star'][-1]:.4e}")
    print(f"wrote {args.out}")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pos = res["tau"] > 0.0
        tau = res["tau"][pos]
        fig, ax = plt.subplots()
        ax.loglog(tau, res["gamma2"][pos], label=r"$\Gamma_2$")
        ax.loglog(tau, res["gamma2_star"][pos], "--", label=r"$\Gamma_2^*$")
        ax.set_xlabel(r"$\tau$")
        ax.set_ylabel(r"$\Gamma_2$")
        ax.set_xlim(tau[-1], 1)
        ax.set_ylim(res["gamma2"][0],res["gamma2"][-1])
        ax.set_title(f"L={args.luminosity:.2e} erg/s, rho16={args.rho16:.2e} g/cm^3, alpha={args.alpha:.2f}, r_in={args.rin:.2e} cm")
        ax.legend()
        fig.savefig(args.plot, dpi=150)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
