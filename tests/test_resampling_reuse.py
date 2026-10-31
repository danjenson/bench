import pytest
import torch

from bench.geometry import logits_on_canvas
from bench.probes import FitResampling


@pytest.mark.parametrize('source', [(3, 4), (33, 34)])
@pytest.mark.parametrize('capacity', [0, 1024])
def test_sampling_reuse_matches_values_gradients_and_clears_after_error(source, capacity):
    cache = FitResampling({'kind': 'segmentation'}, 'cpu')
    cache.limit = capacity
    generator = torch.Generator().manual_seed(81)
    initial = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError, match='intentional interruption'):
        with cache:
            # Formerly collision-prone strings exercise independent entry namespaces.
            for sample, aligned in [('aligned-shape', False), ('other', True), ('aligned-shape', False), ('other', True)]:
                affine = torch.tensor([[.9, .1, .03], [0., 1.1, -.02], [0., 0., 1.]])
                canvas = affine if aligned else torch.eye(3)
                values = torch.randn(3, *source, generator=generator)
                expected_values = values.clone().requires_grad_()
                actual_values = values.clone().requires_grad_()
                expected = logits_on_canvas(expected_values, affine, canvas, (8, 10))
                actual = cache(sample, actual_values, affine, canvas, (8, 10))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                expected.square().sum().backward()
                actual.square().sum().backward()
                torch.testing.assert_close(actual_values.grad, expected_values.grad, rtol=0, atol=0)
                assert 0 <= cache.current_bytes <= cache.limit
            raise RuntimeError('intentional interruption')
    assert not cache.entries and cache.current_bytes == 0
    assert torch.equal(initial, torch.random.get_rng_state())
