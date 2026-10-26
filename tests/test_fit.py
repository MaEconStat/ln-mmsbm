import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score
from ln_mmsbm import estimate_blocks,fit_lognormal,fit_binary_baseline,simulate_lognormal,simulate_bernoulli


def test_block_conditional_mle():
    logx=np.array([[[1.0,3.0],[2.0,4.0]],[[5.0,7.0],[6.0,8.0]]])
    x=np.exp(logx)
    types=np.array([0,0]); source=np.array([[0,0]]);target=np.array([[0,0]])
    mu,var=estimate_blocks(x,types,source,target)
    assert mu.shape==(1,1,1)
    np.testing.assert_allclose(mu[0,0,0],logx.mean())
    np.testing.assert_allclose(var,np.var(logx))


def test_positive_validation():
    a=np.ones((8,7,6)); a[0,0,0]=0
    with pytest.raises(ValueError):fit_lognormal(a,2,2,2)
    with pytest.raises(ValueError):estimate_blocks(a,np.array([0,0,0,1,1,1]),np.zeros((2,8),dtype=int),np.zeros((2,7),dtype=int))


def test_fitted_shapes_and_finite_sigma():
    sample=simulate_lognormal(36,32,10,seed=42)
    fit=fit_lognormal(sample.weights,2,2,2,seed=42,max_iter=8)
    assert fit.layer_types.shape==(10,)
    assert fit.local_source.shape==(2,36)
    assert fit.local_target.shape==(2,32)
    assert fit.block_log_means.shape==(2,2,2)
    assert np.isfinite(fit.log_variance) and fit.log_variance>=0


def test_layer_center_invariant_to_layerwise_scaling():
    base=simulate_lognormal(25,22,8,seed=6)
    scales=np.exp(np.linspace(-3,4,8))
    a=fit_lognormal(base.weights,2,2,2,center_layers=True,seed=6,max_iter=6)
    b=fit_lognormal(base.weights*scales[None,None,:],2,2,2,center_layers=True,seed=6,max_iter=6)
    np.testing.assert_allclose(a.decomposition.relative_residuals,b.decomposition.relative_residuals,atol=1e-10)
    assert adjusted_rand_score(a.layer_types,b.layer_types)==1.0


def test_binary_baseline_results():
    sample=simulate_bernoulli(30,8,seed=15)
    r,l,g=fit_binary_baseline(sample.weights,2,2,seed=15,max_iter=8)
    assert l.shape==(8,) and g.shape==(30,)
    assert r.core.shape==(3,3,2)
