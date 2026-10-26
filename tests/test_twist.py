import numpy as np
import pytest
from ln_mmsbm import twist, reconstruct, simulate_lognormal, simulate_bernoulli


def test_rectangular_decomposition_dimensions_and_projection():
    x=simulate_lognormal(24,19,9,seed=2)
    r=twist(np.log(x.weights),(3,3,2),max_iter=8)
    assert r.source_factors.shape==(24,3)
    assert r.target_factors.shape==(19,3)
    assert r.layer_factors.shape==(9,2)
    assert r.core.shape==(3,3,2)
    assert np.isfinite(reconstruct(r)).all()
    for mat in (r.source_factors,r.target_factors,r.layer_factors):
        np.testing.assert_allclose(mat.T @ mat,np.eye(mat.shape[1]),atol=1e-9)
    assert 0 <= r.relative_residuals[-1] <= 1.0000001


def test_symmetric_binary_retains_common_factor():
    sample=simulate_bernoulli(35,10,seed=7)
    r=twist(sample.weights,(3,3,2),symmetric=True,max_iter=7)
    np.testing.assert_allclose(r.source_factors,r.target_factors)
    np.testing.assert_allclose(reconstruct(r),reconstruct(r).transpose(1,0,2))


def test_rejects_bad_tensor_and_ranks():
    with pytest.raises(ValueError): twist(np.zeros((3,4,2)),(2,2,2))
    with pytest.raises(ValueError): twist(np.ones((3,4,2)),(4,2,2))
    with pytest.raises(ValueError): twist(np.ones((3,4,2)),(2,2,2),symmetric=True)
    with pytest.raises(ValueError): twist(np.full((3,4,2),np.nan),(2,2,2))


def test_simulated_data_is_reproducible_and_undirected_binary():
    a=simulate_lognormal(25,19,9,seed=77)
    b=simulate_lognormal(25,19,9,seed=77)
    np.testing.assert_array_equal(a.weights,b.weights)
    assert np.all(a.weights>0)
    d=simulate_bernoulli(25,9,seed=77).weights
    np.testing.assert_array_equal(d,d.transpose(1,0,2))
    assert np.all(np.diagonal(d,axis1=0,axis2=1)==0)
