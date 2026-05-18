from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_common import experiment_id


def test_experiment_id_distinguishes_depth_features_and_augmentation():
    base = {
        "arch_name": "boundary_gated_multiscale_unet",
        "n_c": 24,
        "depth": 6,
        "lr": 0.0005,
        "loss_name": "masked_l1_gradient",
        "input_features": "height_sdf_normal",
        "augmentation": "flip_rot",
    }
    depth7 = dict(base, depth=7)
    no_aug = dict(base, augmentation="none")
    height_only = dict(base, input_features="height")

    ids = {experiment_id(x) for x in [base, depth7, no_aug, height_only]}
    assert len(ids) == 4, ids
    assert "_d6_" in experiment_id(base)
    assert "_d7_" in experiment_id(depth7)
    assert "height_sdf_normal" in experiment_id(base)
    assert "flip_rot" in experiment_id(base)


if __name__ == "__main__":
    test_experiment_id_distinguishes_depth_features_and_augmentation()
    print("PASS test_experiment_id_distinguishes_depth_features_and_augmentation")



