r"""Fast volumetric renderer using precomputed emission LUT.

Replaces the per-sample spectral emission computation with a single
(T_tr, T_ve) → RGB lookup per ray sample, reducing render time from
hours to minutes at full resolution.

The density-weighted RGB contribution at each sample is accumulated
along the ray.  The composition dependence is baked into the LUT at
a representative post-shock composition — the dominant temperature
dependence (exponential Boltzmann factors spanning 10+ orders of
magnitude) dwarfs composition variation for rendering purposes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .fast_emission import RGBEmissionLUT, build_rgb_lut, save_lut, load_lut
from .render import Camera, tone_map, composite, save_image, _ray_sphere_intersect
from .spectral_data import srgb_gamma
from .transport import VolumeData

__all__ = ["fast_render"]


def fast_render(
    volume: VolumeData,
    camera: Camera,
    lut: RGBEmissionLUT | None = None,
    *,
    n_steps: int = 400,
    T_threshold: float = 1500.0,
    domain_radius: float | None = None,
    body_centre: NDArray[np.float64] | None = None,
    body_radius: float = 0.0,
    progress_callback: Any = None,
) -> NDArray[np.float64]:
    """Render the radiating flow field using precomputed RGB emission LUT.

    Returns (H, W, 3) linear-RGB image (unnormalised radiance).
    """
    if lut is None:
        print("Building emission LUT (one-time cost)...")
        lut = build_rgb_lut(progress=True)

    H, W = camera.height, camera.width
    dirs = camera.ray_directions()

    bbox_min, bbox_max = volume.bounds()
    domain_centre = (bbox_min + bbox_max) / 2.0
    if domain_radius is None:
        domain_radius = float(np.linalg.norm(bbox_max - bbox_min)) / 2.0 * 1.1

    image = np.zeros((H, W, 3), dtype=np.float64)

    for row in range(H):
        if progress_callback is not None:
            progress_callback(row, H)

        for col in range(W):
            d = dirs[row, col]

            hit = _ray_sphere_intersect(
                camera.position, d, domain_centre, domain_radius,
            )
            if hit is None:
                continue
            t_min, t_max = hit
            t_min = max(t_min, 0.0)
            if t_max <= t_min:
                continue

            is_body = False
            if body_radius > 0 and body_centre is not None:
                body_hit = _ray_sphere_intersect(
                    camera.position, d, body_centre, body_radius,
                )
                if body_hit is not None and body_hit[0] > 0:
                    is_body = True
                    t_max = min(t_max, body_hit[0])

            ds = (t_max - t_min) / n_steps
            t_values = np.linspace(t_min, t_max, n_steps)
            points = (camera.position[np.newaxis, :]
                      + t_values[:, np.newaxis] * d[np.newaxis, :])

            rho_batch, T_tr_batch, T_ve_batch, p_batch = volume.sample_batch(points)

            hot = T_tr_batch >= T_threshold
            if not np.any(hot):
                if is_body:
                    image[row, col] = [0.02, 0.02, 0.02]
                continue

            hot_idx = np.where(hot)[0]
            rgb_emission = lut.lookup_batch(
                T_tr_batch[hot_idx], T_ve_batch[hot_idx],
            )

            n_total = rho_batch[hot_idx].sum(axis=1) / (28.97e-3 / 6.022e23)

            for k, idx in enumerate(hot_idx):
                image[row, col] += rgb_emission[k] * n_total[k] * ds

            if is_body and np.all(image[row, col] == 0):
                image[row, col] = [0.02, 0.02, 0.02]

    return image


def fast_render_vectorised(
    volume: VolumeData,
    camera: Camera,
    lut: RGBEmissionLUT | None = None,
    *,
    n_steps: int = 400,
    T_threshold: float = 1500.0,
    domain_radius: float | None = None,
    body_centre: NDArray[np.float64] | None = None,
    body_radius: float = 0.0,
    body_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    body_profile: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    k_neighbors: int = 1,
    progress_callback: Any = None,
) -> NDArray[np.float64]:
    """Row-vectorised fast renderer — processes each row as a batch.

    For a 1920×1080 image with 400 steps, this processes 768,000 sample
    points per row in one KD-tree query, which is dramatically faster
    than the per-pixel loop.

    body_profile: (x_array, r_array) defining the axisymmetric capsule
    outer-mold-line.  A point at (x, y, z) is inside the body when
    sqrt(y² + z²) < interp(x, x_array, r_array).  Rays that enter the
    body are truncated at the surface (opaque body).
    """
    if lut is None:
        print("Building emission LUT (one-time cost)...")
        lut = build_rgb_lut(progress=True)

    H, W = camera.height, camera.width
    dirs = camera.ray_directions()

    bbox_min, bbox_max = volume.bounds()
    domain_centre = (bbox_min + bbox_max) / 2.0
    if domain_radius is None:
        domain_radius = float(np.linalg.norm(bbox_max - bbox_min)) / 2.0 * 1.1

    use_profile = body_profile is not None
    if use_profile:
        prof_x, prof_r = body_profile

    image = np.zeros((H, W, 3), dtype=np.float64)

    for row in range(H):
        if progress_callback is not None:
            progress_callback(row, H)

        row_points_all = []
        row_meta = []
        total_pts = 0

        for col in range(W):
            d = dirs[row, col]
            hit = _ray_sphere_intersect(
                camera.position, d, domain_centre, domain_radius,
            )
            if hit is None:
                row_meta.append(None)
                continue

            t_min, t_max = hit
            t_min = max(t_min, 0.0)
            if t_max <= t_min:
                row_meta.append(None)
                continue

            if not use_profile and body_radius > 0 and body_centre is not None:
                body_hit = _ray_sphere_intersect(
                    camera.position, d, body_centre, body_radius,
                )
                if body_hit is not None and body_hit[0] > 0:
                    t_max = min(t_max, body_hit[0])

            ds = (t_max - t_min) / n_steps
            t_vals = np.linspace(t_min, t_max, n_steps)
            pts = (camera.position[np.newaxis, :]
                   + t_vals[:, np.newaxis] * d[np.newaxis, :])

            start = total_pts
            row_points_all.append(pts)
            total_pts += n_steps
            row_meta.append((col, start, start + n_steps, ds))

        if not row_points_all:
            continue

        all_pts = np.vstack(row_points_all)
        rho_all, T_tr_all, T_ve_all, p_all = volume.sample_batch(
            all_pts, k=k_neighbors,
        )

        if use_profile:
            pts_r = np.sqrt(all_pts[:, 1] ** 2 + all_pts[:, 2] ** 2)
            body_r = np.interp(all_pts[:, 0], prof_x, prof_r, left=0.0, right=0.0)
            inside_body = pts_r < body_r

        for meta in row_meta:
            if meta is None:
                continue
            col, s, e, ds = meta

            if use_profile:
                seg_inside = inside_body[s:e]
                is_body = bool(np.any(seg_inside))
                if is_body:
                    first_in = int(np.argmax(seg_inside))
                    e_eff = s + first_in
                else:
                    e_eff = e
            else:
                e_eff = e
                is_body = False

            T_tr_seg = T_tr_all[s:e_eff]
            T_ve_seg = T_ve_all[s:e_eff]
            rho_seg = rho_all[s:e_eff]

            hot = T_tr_seg >= T_threshold
            if not np.any(hot):
                if is_body:
                    image[row, col] = list(body_color)
                continue

            hot_idx = np.where(hot)[0]
            rgb_em = lut.lookup_batch(T_tr_seg[hot_idx], T_ve_seg[hot_idx])

            n_total = rho_seg[hot_idx].sum(axis=1) / (28.97e-3 / 6.022e23)

            contribution = (rgb_em * n_total[:, np.newaxis] * ds).sum(axis=0)
            image[row, col] = contribution

            if is_body and np.all(contribution == 0):
                image[row, col] = list(body_color)

    return image
