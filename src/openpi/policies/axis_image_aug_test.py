import numpy as np

from openpi.policies import axis_image_aug

KEY = "observation.images.third_person"


def test_a_square_frame_is_never_touched():
    img = np.random.default_rng(0).integers(0, 255, (224, 224, 3), np.uint8)
    out = axis_image_aug.SimImageAug(p_grayscale=1.0)({KEY: img})
    np.testing.assert_array_equal(out[KEY], img)


def test_a_wide_frame_keeps_its_shape_and_dtype_and_both_cameras_change_together():
    rng = np.random.default_rng(0)
    data = {
        KEY: rng.integers(0, 255, (360, 640, 3), np.uint8),
        "observation.images.wrist": rng.integers(0, 255, (360, 640, 3), np.uint8),
    }
    out = axis_image_aug.SimImageAug(p_jitter=0, p_blur=0, p_posterize=0, p_sharpness=0, p_grayscale=1.0)(data)
    for k in data:
        assert out[k].shape == (360, 640, 3)
        assert out[k].dtype == np.uint8
        assert (out[k][..., 0] == out[k][..., 1]).all()
        assert (out[k][..., 1] == out[k][..., 2]).all()
