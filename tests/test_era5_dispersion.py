"""ERA5-wind terminal dispersion: measured winds into CEP/R95.

Exercised on synthetic in-memory ensembles so the full-fidelity numerics are
covered without needing the ERA5 archive or a NetCDF backend present. Each
replicate flies through its own measured wind-and-density profile.
"""

from __future__ import annotations

import numpy as np
import pytest

from aether.atmosphere.era5 import ERA5Column, WindEnsemble
from aether.batch.entry_demo import EntryDispersionModel
from aether.batch.era5_dispersion import (
    ERA5EntryDispersion,
    load_site_ensemble,
    representative_wind,
    sample_era5_winds,
    terminal_dispersion,
)

# ERA5-like pressure levels (Pa), surface to ~30 km.
_P = np.array(
    [100000, 92500, 85000, 70000, 50000, 40000, 30000, 20000, 10000, 5000, 1000.0]
)
_T = np.linspace(288.0, 228.0, _P.size)
_Q = np.full(_P.size, 1.0e-3)


def _column(east: float, north: float, shear: float = 1.0) -> ERA5Column:
    """A profile with a chosen low-level wind and a linear shear aloft."""
    ramp = np.linspace(1.0, shear, _P.size)
    return ERA5Column(
        latitude=45.0, longitude=30.0, epoch="t",
        pressure=_P, temperature=_T,
        eastward=east * ramp, northward=north * ramp, specific_humidity=_Q,
    )


def _ensemble_crosswind(n: int = 60, seed: int = 1) -> WindEnsemble:
    """Strong prevailing *northward* (crossrange) wind, weak eastward."""
    rng = np.random.default_rng(seed)
    cols = [
        _column(rng.normal(0.0, 2.0), rng.normal(25.0, 6.0), rng.uniform(1.5, 3.0))
        for _ in range(n)
    ]
    return WindEnsemble(columns=tuple(cols))


def _ensemble_calm(n: int = 60) -> WindEnsemble:
    """Zero wind at every level — same density structure, no wind."""
    return WindEnsemble(columns=tuple(_column(0.0, 0.0) for _ in range(n)))


class TestRepresentativeWind:
    def test_density_weighting_favours_the_dense_low_levels(self):
        col = _column(10.0, 0.0, shear=5.0)
        e, _ = representative_wind(col)
        assert e < float(np.mean(col.eastward))       # below the column mean
        assert e > col.eastward[0] * 0.5              # but not negligible

    def test_band_restriction_changes_the_answer(self):
        col = _column(10.0, 0.0, shear=5.0)
        low = representative_wind(col, band=(0.0, 5.0e3))[0]
        high = representative_wind(col, band=(10.0e3, 25.0e3))[0]
        assert high > low                             # stronger aloft


class TestSampling:
    def test_preserves_ensemble_anisotropy(self):
        winds = sample_era5_winds(_ensemble_crosswind(), 2000, np.random.default_rng(0))
        assert winds[:, 1].mean() > 15.0              # prevailing north
        assert abs(winds[:, 0].mean()) < 3.0          # weak east
        assert winds[:, 1].std() > 2.0 * winds[:, 0].std()

    def test_an_empty_ensemble_cannot_be_constructed(self):
        # The ensemble type refuses to be empty, so downstream code never has
        # to guess an atmosphere from nothing.
        with pytest.raises(ValueError):
            WindEnsemble(columns=())


class TestFullFidelityDescent:
    def _model(self) -> EntryDispersionModel:
        return EntryDispersionModel()

    def test_impacts_are_finite_and_correctly_shaped(self):
        impacts = ERA5EntryDispersion(self._model(), _ensemble_crosswind()).fly(400, 3)
        assert impacts.shape == (400, 2)
        assert np.all(np.isfinite(impacts))

    def test_measured_density_actually_varies_with_altitude(self):
        # A profile field query must return the surface density at z=0 and a
        # much smaller density at 30 km — i.e. we are reading rho(z), not a
        # constant. Guards against a silent reduction to a fixed density.
        model = ERA5EntryDispersion(self._model(), _ensemble_calm())
        grid, _, _, rho = model._profile_fields(4, np.random.default_rng(0))
        i0 = int(np.argmin(np.abs(grid - 0.0)))
        i30 = int(np.argmin(np.abs(grid - 30.0e3)))
        assert rho[0, i0] > 1.0                       # ~1.2 at the surface
        assert rho[0, i30] < 0.1                      # thin at 30 km
        assert rho[0, i0] > 10.0 * rho[0, i30]

    def test_crosswind_biases_the_mean_crossrange_impact(self):
        # A prevailing northward wind must push the mean crossrange (y) impact
        # off the calm nominal. Same density both runs, so only the wind differs.
        model = self._model()
        cross = ERA5EntryDispersion(model, _ensemble_crosswind()).fly(2000, 5)
        calm = ERA5EntryDispersion(model, _ensemble_calm()).fly(2000, 5)
        assert abs(float(np.mean(calm[:, 1]))) < 5.0
        assert float(np.mean(cross[:, 1])) > 50.0
        # And it inflates the crossrange spread relative to calm.
        assert cross[:, 1].std() > calm[:, 1].std()


class TestTerminalDispersion:
    def test_reports_cep_and_r95(self):
        report = terminal_dispersion(
            EntryDispersionModel(), ensemble=_ensemble_crosswind(),
            n_replicates=1500, seed=7,
        )
        assert report.r95 > 0.0
        assert report.cep > 0.0
        assert report.cep < report.r95

    def test_no_approximation_fallback_without_a_real_ensemble(self):
        # The whole point: no assumed-wind fallback. It must raise, not guess.
        with pytest.raises(ValueError, match="real wind ensemble"):
            terminal_dispersion(EntryDispersionModel(), ensemble=None)


class TestArchiveLoader:
    def test_missing_archive_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_site_ensemble(tmp_path / "nope", 45.0, 30.0)

    def test_empty_archive_raises(self, tmp_path):
        (tmp_path / "2024" / "01").mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            load_site_ensemble(tmp_path, 45.0, 30.0)
