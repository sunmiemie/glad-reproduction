import numpy as np
import pytest
import torch

from glad.appearance import CandidateNet, letterbox, restore_boxes


@pytest.mark.parametrize("height,width", [(720, 1280), (1080, 1920), (319, 301), (301, 319)])
def test_letterbox_coordinates_roundtrip(height, width):
    image = np.zeros((height, width, 3), np.uint8)
    tensor, scale, pad = letterbox(image)
    original = torch.tensor([[0., 0., width, height], [12., 25., 71., 99.]])
    transformed = original.clone()
    transformed[:, (0, 2)] = transformed[:, (0, 2)] * scale[0] + pad[0]
    transformed[:, (1, 3)] = transformed[:, (1, 3)] * scale[1] + pad[1]
    result = restore_boxes(transformed, scale, pad, image.shape)
    assert tensor.shape == (1, 3, 640, 640)
    assert torch.allclose(result, original, atol=0.0002)


def test_letterbox_rgb_and_padding():
    image = np.zeros((100, 200, 3), np.uint8)
    image[:] = (20, 80, 240)
    tensor, _, pad = letterbox(image)
    assert pad == (0, 160)
    assert torch.allclose(tensor[0, :, 200, 100], torch.tensor([240, 80, 20]) / 255)
    assert torch.allclose(tensor[0, :, 0, 0], torch.tensor([128, 128, 128]) / 255)


def test_boxes_in_padding_are_clipped_to_crop():
    image = np.zeros((100, 200, 3), np.uint8)
    _, scale, pad = letterbox(image)
    result = restore_boxes(torch.tensor([[-20., -50., 700., 680.]]), scale, pad, image.shape)
    assert result.tolist() == [[0., 0., 200., 100.]]


def test_classifier_checkpoint_shape():
    net = CandidateNet().eval()
    assert net(torch.zeros(2, 3, 32, 32)).shape == (2, 2)
    assert net.fc1.in_features == 400


def test_empty_image_is_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        letterbox(np.empty((0, 0, 3), np.uint8))
