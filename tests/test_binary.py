import numpy as np
from ln_mmsbm import simulate_bernoulli,fit_bernoulli


def test_full_binary_pipeline_and_block_symmetry():
    data=simulate_bernoulli(42,12,seed=33)
    result=fit_bernoulli(data.weights,2,2,seed=33,max_iter=15)
    assert result.layer_types.shape==(12,)
    assert result.global_nodes.shape==(42,)
    assert result.local_nodes.shape==(2,42)
    assert result.block_probabilities.shape==(2,2,2)
    np.testing.assert_allclose(result.block_probabilities,result.block_probabilities.transpose(0,2,1),equal_nan=True)
    assert np.nanmin(result.block_probabilities)>=0
    assert np.nanmax(result.block_probabilities)<=1
