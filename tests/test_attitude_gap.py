"""Attitude-manifold hyperbolicity (task R1-V2, Paper 1 §4.5).

The load-bearing claim is the *bank invariance* identity: rotating a body
about the oncoming flow direction does not change which of its panels face
the flow, so the aerodynamic moment cannot depend on bank angle. That is
what leaves an exact zero in the attitude spectrum and keeps bank in the
skeleton, so it is asserted to machine precision rather than measured.
"""

from __future__ import annotations

import numpy as np
import pytest

from aether.aerodynamics.panels import PanelModel, curved_lifting_body
from aether.verification.r1v2_attitude_gap import attitude_spectrum, damped_moment


@pytest.fixture(scope="module")
def model() -> PanelModel:
    base = curved_lifting_body()
    return PanelModel(
        centroids=base.centroids,
        normals=base.normals,
        areas=base.areas,
        reference_point=np.array([0.35 * 6.0, 0.0, 0.0]),
    )


def _rodrigues(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate row-vectors ``v`` about ``axis`` by ``angle`` (Rodrigues)."""
    k = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    return v * c + np.cross(k, v) * s + np.outer(v @ k, k) * (1.0 - c)


def _rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _body_from_aero(alpha: float, beta: float) -> np.ndarray:
    """Velocity-frame to body-frame rotation for a given incidence pair."""
    ca, sa, cb, sb = np.cos(alpha), np.sin(alpha), np.cos(beta), np.sin(beta)
    # first column is the freestream direction in body axes, matching
    # PanelModel.velocity_direction
    x = np.array([ca * cb, sb, sa * cb])
    x /= np.linalg.norm(x)
    helper = np.array([0.0, 0.0, 1.0])
    y = np.cross(helper, x)
    y /= np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])


def test_bank_does_not_change_the_body_axis_freestream() -> None:
    """Bank rotates the body *about* the flow, so it cannot move (alpha, beta).

    This is the substantive half of the invariance claim. Bank enters the
    attitude as a rotation about the velocity vector, and that rotation
    fixes the velocity vector by construction -- so two genuinely different
    attitudes sharing ``(alpha, beta)`` present the *same* freestream
    direction in body axes, and hence the same aerodynamic moment.
    """
    alpha, beta = np.radians(25.0), np.radians(4.0)
    r_ba = _body_from_aero(alpha, beta)
    v_inertial = np.array([1.0, 0.0, 0.0])

    reference = None
    attitudes = []
    for bank in np.radians([0.0, 30.0, 60.0, 90.0, 180.0]):
        r_bi = r_ba @ _rot_x(bank)
        attitudes.append(r_bi)
        v_body = r_bi @ v_inertial
        if reference is None:
            reference = v_body
        assert v_body == pytest.approx(reference, abs=1e-14)

    # the attitudes really are distinct -- otherwise the above is vacuous
    for other in attitudes[1:]:
        assert not np.allclose(other, attitudes[0])


def test_moment_is_independent_of_bank(model: PanelModel) -> None:
    """dM/dsigma == 0: bank is not an argument of the aerodynamic moment.

    Given the preceding test, the moment depends on bank only through
    ``(alpha, beta)``, which bank does not move. The identity is then that
    the body-axis moment is a function of ``(alpha, beta, Omega, q)`` alone
    -- there is no bank argument to pass, and that absence *is* the claim.
    Here it is pinned down by checking that a body rotated about the flow
    axis carries a moment of unchanged magnitude and unchanged component
    along the flow, which is the coordinate-free content.
    """
    alpha, q_dyn = np.radians(25.0), 2.0e4
    v_hat = model.velocity_direction(alpha, 0.0)
    reference = damped_moment(model, alpha, 0.0, np.zeros(3), q_dyn)

    for bank in np.radians([0.0, 30.0, 60.0, 90.0, 180.0]):
        rotated = PanelModel(
            centroids=_rodrigues(model.centroids, v_hat, bank),
            normals=_rodrigues(model.normals, v_hat, bank),
            areas=model.areas,
            reference_point=_rodrigues(model.reference_point[None, :], v_hat, bank)[0],
        )
        moment = damped_moment(rotated, alpha, 0.0, np.zeros(3), q_dyn)
        assert np.linalg.norm(moment) == pytest.approx(
            np.linalg.norm(reference), rel=1e-10
        )
        assert moment @ v_hat == pytest.approx(reference @ v_hat, abs=1e-6)


def test_rate_damping_is_dissipative(model: PanelModel) -> None:
    """All three Newtonian rate-damping derivatives are stabilizing.

    This is R1-V2's stated failure criterion; if it fails the attitude rate
    directions do not relax and prop:skeleton_dimension loses its five
    attitude removals at every altitude.
    """
    from aether.verification.r1v2_attitude_gap import _derivatives

    for alpha_deg in (15.0, 25.0, 35.0):
        d = _derivatives(model, np.radians(alpha_deg), 2.0e4)
        for key in ("L_p", "M_q", "N_r"):
            assert d[key] < 0.0, f"{key} = {d[key]:+.3e} at alpha={alpha_deg} deg"


def test_spectrum_is_hyperbolic_low_and_marginal_high(model: PanelModel) -> None:
    """The separation is won by dynamic pressure, and only well down.

    Deep in the corridor the attitude modes decay an order of magnitude
    faster than the trajectory evolves; near the top they do not. The
    reduction is a mode-local statement and this is the evidence for it.
    """
    t_slow, alpha = 806.0, np.radians(25.0)

    deep, _ = attitude_spectrum(model, alpha, 1.0e5)
    assert np.all(deep.real < 0.0)
    assert np.abs(deep.real).min() * t_slow > 10.0

    high, _ = attitude_spectrum(model, alpha, 2.0e2)
    assert np.abs(high.real).min() * t_slow < 1.0


def test_abscissa_is_linear_in_dynamic_pressure(model: PanelModel) -> None:
    """The abscissa scales as q_inf to within one part in 1e4.

    Damping is linear in dynamic pressure and inertia is fixed, so the
    aerodynamic entries of the attitude matrix scale exactly. The scaling
    is not *exact* because the kinematic row of the lateral block --
    ``[0, sin(alpha), -cos(alpha)]`` -- carries no dynamic pressure, and it
    perturbs the spiral root at relative order (kinematic rate)/(aero
    rate). Over three decades of corridor that perturbation stays near
    1e-4, which is what makes the floor ``q_underbar`` a genuine crossing
    rather than an artefact of where the corridor was sampled.
    """
    alpha = np.radians(25.0)
    q_dyn = np.array([5.0e1, 1.0e3, 2.0e4, 1.0e5])
    absc = np.array(
        [np.abs(attitude_spectrum(model, alpha, q)[0].real).min() for q in q_dyn]
    )
    ratio = absc / q_dyn
    assert (ratio.max() - ratio.min()) / ratio.mean() < 1.0e-3

    slope, _ = np.polyfit(np.log10(q_dyn), np.log10(absc), 1)
    assert slope == pytest.approx(1.0, abs=1.0e-4)
