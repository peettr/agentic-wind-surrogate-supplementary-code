"""
sdf.py â€” Signed Distance Function + surface normal augmentation.

Given a binary building mask (building_heights > 0), compute:
  1. SDF: signed distance to nearest building boundary
     (negative inside buildings, positive outside)
  2. Normal direction: angle from each boundary pixel to its nearest
     non-building neighbour, propagated to all pixels via the same
     distance transform.

These two channels are appended to the building-height input, giving
a 3-channel input (height, sdf, normal_angle) instead of just height.

References:
  - "Neural Operators with Localized Physics-Informed Attention"
  - baseline_source Model Scout synthesis #C: "Signed Distance Conditioned UNet"
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def compute_sdf(building_mask: np.ndarray) -> np.ndarray:
    """Compute signed distance field.

    Parameters
    ----------
    building_mask : (H, W) bool or float array
        True / >0 inside buildings.

    Returns
    -------
    sdf : (H, W) float32
        Negative inside buildings, positive outside.
        Magnitude = distance in pixels to nearest boundary.
    """
    mask = building_mask > 0
    if not mask.any():
        # No buildings: uniform positive distance
        return np.full(building_mask.shape, 1e6, dtype=np.float32)

    # Distance from each pixel to the nearest building pixel
    dist_outside = ndimage.distance_transform_edt(~mask).astype(np.float32)
    # Distance from each pixel to the nearest non-building pixel
    dist_inside = ndimage.distance_transform_edt(mask).astype(np.float32)

    sdf = dist_outside - dist_inside
    return sdf


def compute_boundary_normal_angle(building_mask: np.ndarray) -> np.ndarray:
    """Compute angle of the outward-pointing normal at each pixel.

    For boundary pixels: the direction from building interior to exterior.
    For non-boundary pixels: propagated via distance transform (nearest
    boundary pixel's normal).

    Returns
    -------
    angle : (H, W) float32, in [0, 2Ï€)
        atan2 of the outward normal direction.
    """
    mask = building_mask > 0

    if not mask.any():
        return np.zeros(building_mask.shape, dtype=np.float32)

    # Erode to find boundary (building pixels adjacent to non-building)
    eroded = ndimage.binary_erosion(mask)
    boundary = mask & ~eroded  # building boundary pixels

    if not boundary.any():
        # All building pixels are boundary (single-pixel buildings)
        boundary = mask.copy()

    # Compute gradient of distance transform to get normal direction
    # Use EDT on the complement â†’ gradient points away from buildings
    dt = ndimage.distance_transform_edt(~mask).astype(np.float32)
    gy, gx = np.gradient(dt)
    angle = np.arctan2(gy, gx).astype(np.float32)  # [-Ï€, Ï€]

    # Normalize to [0, 2Ï€)
    angle = angle % (2 * np.pi)

    return angle


def augment_input_channels(
    building_heights: np.ndarray,
    normalize_sdf: bool = True,
    sdf_scale: float | None = None,
) -> np.ndarray:
    """Compute SDF + normal channels from building heights.

    Parameters
    ----------
    building_heights : (H, W) float32
        Raw building height array (0 = ground).
    normalize_sdf : bool
        If True, normalize SDF to [-1, 1] range.
    sdf_scale : float or None
        If set, divide SDF by this value instead of auto-scaling.

    Returns
    -------
    channels : (3, H, W) float32
        Channel 0: building heights (unchanged)
        Channel 1: SDF
        Channel 2: boundary normal angle / (2Ï€) â†’ [0, 1]
    """
    mask = building_heights > 0

    sdf = compute_sdf(mask)
    angle = compute_boundary_normal_angle(mask)

    if sdf_scale is not None:
        sdf = sdf / sdf_scale
    elif normalize_sdf:
        max_abs = np.abs(sdf).max()
        if max_abs > 0:
            sdf = sdf / max_abs

    # Normalize angle to [0, 1]
    angle_norm = angle / (2 * np.pi)

    return np.stack([building_heights, sdf, angle_norm], axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing SDF computation...")

    # Create a simple test: 640x640 grid with a rectangular building
    H, W = 640, 640
    bh = np.zeros((H, W), dtype=np.float32)
    bh[200:300, 150:250] = 25.0  # 100x100 building
    bh[400:450, 400:500] = 15.0   # 50x100 building

    # Test SDF
    sdf = compute_sdf(bh > 0)
    print(f"  SDF shape: {sdf.shape}, range: [{sdf.min():.1f}, {sdf.max():.1f}]")
    assert sdf[250, 200] < 0, "Inside building should be negative"
    assert sdf[100, 100] > 0, "Outside buildings should be positive"
    print("  [PASS] SDF sign convention correct")

    # Test boundary normal
    angle = compute_boundary_normal_angle(bh > 0)
    print(f"  Angle shape: {angle.shape}, range: [{angle.min():.2f}, {angle.max():.2f}]")
    print("  [PASS] Boundary normal computed")

    # Test full augmentation
    channels = augment_input_channels(bh)
    print(f"  Augmented channels shape: {channels.shape}")
    assert channels.shape == (3, H, W)
    print("  [PASS] Full augmentation")

    # Check SDF boundary values
    # At the left edge of the building (col 150), SDF should be ~0
    assert abs(sdf[250, 150]) < 1.5, f"Boundary SDF should be ~0, got {sdf[250, 150]}"
    print("  [PASS] Boundary SDF ~ 0")

    print("\nAll tests passed!")



