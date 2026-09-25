r"""Camera, ray casting, and final image composition.

Sets up a virtual camera looking at the vehicle, casts rays through the
radiating flow field, integrates spectral emission along each ray, and
converts the spectral radiance at each pixel to sRGB via CIE 1931 colour
matching.

Tone mapping uses a physically-motivated exposure model: the peak
radiance in the image sets the white point, with a controllable exposure
parameter.  The vehicle silhouette is rendered by checking whether the
ray intersects the body surface (approximated as a convex hull or
explicit mesh intersection).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .spectral_data import cie_xyz, srgb_gamma, xyz_to_srgb
from .transport import VolumeData, integrate_ray

__all__ = ["Camera", "render_image", "composite"]


@dataclass
class Camera:
    """Pinhole camera model."""
    position: NDArray[np.float64]       # camera location (m)
    look_at: NDArray[np.float64]        # point the camera points at
    up: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0])
    )
    fov_deg: float = 30.0               # vertical field of view
    width: int = 1920
    height: int = 1080

    def ray_directions(self) -> NDArray[np.float64]:
        """Generate (H, W, 3) array of normalised ray directions."""
        forward = self.look_at - self.position
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, self.up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        aspect = self.width / self.height
        half_h = np.tan(np.radians(self.fov_deg) / 2.0)
        half_w = half_h * aspect

        v = np.linspace(half_h, -half_h, self.height)
        u = np.linspace(-half_w, half_w, self.width)
        uu, vv = np.meshgrid(u, v)

        dirs = (
            forward[np.newaxis, np.newaxis, :]
            + uu[:, :, np.newaxis] * right[np.newaxis, np.newaxis, :]
            + vv[:, :, np.newaxis] * up[np.newaxis, np.newaxis, :]
        )
        norms = np.linalg.norm(dirs, axis=2, keepdims=True)
        return dirs / norms


def _ray_sphere_intersect(
    origin: NDArray, direction: NDArray, centre: NDArray, radius: float,
) -> tuple[float, float] | None:
    """Ray–sphere intersection; returns (t_near, t_far) or None."""
    oc = origin - centre
    a = np.dot(direction, direction)
    b = 2.0 * np.dot(oc, direction)
    c = np.dot(oc, oc) - radius * radius
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return None
    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)
    return t1, t2


def render_image(
    volume: VolumeData,
    camera: Camera,
    *,
    wavelength_nm: NDArray[np.float64] | None = None,
    n_steps: int = 400,
    T_threshold: float = 1500.0,
    domain_radius: float | None = None,
    body_centre: NDArray[np.float64] | None = None,
    body_radius: float = 0.0,
    progress_callback: Any = None,
) -> NDArray[np.float64]:
    """Render the radiating flow field to an (H, W, 3) linear-RGB image.

    Parameters
    ----------
    volume : loaded SU2 volume data
    camera : camera specification
    wavelength_nm : spectral grid (default 380–900 nm at 2 nm spacing)
    n_steps : samples per ray
    T_threshold : skip cells below this temperature
    domain_radius : bounding sphere radius for ray clipping
    body_centre : centre of the vehicle (for silhouette)
    body_radius : approximate vehicle radius for silhouette occlusion
    progress_callback : called with (row, total_rows) if provided

    Returns
    -------
    image : (H, W, 3) linear RGB, unnormalised radiance
    """
    if wavelength_nm is None:
        wavelength_nm = np.arange(380.0, 902.0, 2.0)

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

            I_lambda = integrate_ray(
                volume, camera.position, d,
                wavelength_nm, t_min, t_max, n_steps, T_threshold,
            )

            X, Y, Z = cie_xyz(wavelength_nm, I_lambda)
            r, g, b = xyz_to_srgb(X, Y, Z)

            if is_body:
                image[row, col] = [0.02, 0.02, 0.02]
            else:
                image[row, col] = [max(r, 0), max(g, 0), max(b, 0)]

    return image


def tone_map(
    image: NDArray[np.float64],
    exposure: float = 1.0,
    white_point: float | None = None,
) -> NDArray[np.float64]:
    """Filmic tone mapping: Reinhard with controllable exposure.

    Returns (H, W, 3) in [0, 1] ready for gamma correction.
    """
    img = image * exposure
    if white_point is None:
        white_point = max(np.percentile(img[img > 0], 99.5), 1e-30)
    mapped = img * (1.0 + img / white_point**2) / (1.0 + img)
    return np.clip(mapped, 0.0, 1.0)


def _bloom(
    image: NDArray[np.float64],
    threshold: float = 0.3,
    radius: float = 20.0,
    strength: float = 0.3,
) -> NDArray[np.float64]:
    """Add bloom (lens scattering) to a tone-mapped image.

    Extracts bright regions above threshold, applies Gaussian blur,
    and adds back as a soft glow.
    """
    from scipy.ndimage import gaussian_filter
    bright = np.clip(image - threshold, 0.0, None)
    glow = np.zeros_like(bright)
    for c in range(3):
        glow[:, :, c] = gaussian_filter(bright[:, :, c], sigma=radius)
    return np.clip(image + glow * strength, 0.0, 1.0)


def composite(
    image: NDArray[np.float64],
    exposure: float = 1.0,
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    bloom_strength: float = 0.0,
    bloom_radius: float = 20.0,
) -> NDArray[np.uint8]:
    """Tone map, gamma correct, and convert to 8-bit sRGB.

    Parameters
    ----------
    image : (H, W, 3) linear RGB from render_image
    exposure : exposure multiplier
    background : RGB background colour (linear, 0–1)
    bloom_strength : bloom intensity (0 = off, 0.3 = subtle, 1.0 = heavy)
    bloom_radius : Gaussian blur radius in pixels

    Returns
    -------
    (H, W, 3) uint8 sRGB image
    """
    mapped = tone_map(image, exposure)

    mask = np.all(image == 0, axis=2)
    for c in range(3):
        mapped[:, :, c] = np.where(mask, background[c], mapped[:, :, c])

    if bloom_strength > 0:
        mapped = _bloom(mapped, strength=bloom_strength, radius=bloom_radius)

    out = srgb_gamma(mapped)
    return (out * 255.0).astype(np.uint8)


def save_image(image_uint8: NDArray[np.uint8], path: str | Path) -> None:
    """Write an (H, W, 3) uint8 array as PNG."""
    from PIL import Image
    img = Image.fromarray(image_uint8, mode="RGB")
    img.save(str(path), optimize=True)
    print(f"  saved {path} ({img.width}×{img.height})")
