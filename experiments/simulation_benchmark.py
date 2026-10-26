"""Exploratory multi-seed comparison: log-TWIST vs raw-weight TWIST.

Usage: python experiments/simulation_benchmark.py
Generates a CSV (to stdout) over varying block separation and Gaussian noise.
This intentionally includes regimes where reliable recovery may NOT be possible.
A public performance claim requires many more replications and statistical tests.
"""
from __future__ import annotations
import csv
import sys
from time import perf_counter
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from ln_mmsbm import simulate_lognormal, fit_lognormal, twist


def matched_types(actual: np.ndarray, inferred: np.ndarray, n_types: int) -> dict[int,int]:
    """Permute inferred labels to maximize overlap with true type labels."""
    mat=np.zeros((n_types,n_types),dtype=int)
    for true_label,estimated_label in zip(actual,inferred):
        mat[int(true_label),int(estimated_label)]+=1
    true_idx,pred_idx=linear_sum_assignment(-mat)
    return {int(p):int(t) for t,p in zip(true_idx,pred_idx)}


def joint_labels(memberships: np.ndarray) -> np.ndarray:
    """Encode type-specific node group tuples into a global class ID."""
    _,ids=np.unique(memberships.T,axis=0,return_inverse=True)
    return ids


def main() -> None:
    writer=csv.writer(sys.stdout)
    writer.writerow(["seed","contrast","sigma","layer_ari_log","layer_ari_raw",
        "global_source_ari_log","global_target_ari_log","local_source_ari_log_mean",
        "local_target_ari_log_mean","seconds_log","log_scale_relative_residual"])
    for contrast in (0.2,0.6,1.6):
        for sigma in (0.8,1.5):
            for seed in range(3):
                block_means=np.random.default_rng(seed+200).normal(0,1,(2,2,2))*contrast
                data=simulate_lognormal(42,39,10,n_types=2,k_source=2,k_target=2,
                                        block_means=block_means,sigma=sigma,seed=seed)
                start=perf_counter()
                fit=fit_lognormal(data.weights,2,2,2,seed=seed,max_iter=25)
                elapsed=perf_counter()-start
                raw=twist(data.weights,(3,3,2),max_iter=25)
                raw_layer=KMeans(n_clusters=2,random_state=seed,n_init=20).fit_predict(raw.layer_factors)
                align=matched_types(data.layer_types,fit.layer_types,2)
                inv_map={t:p for p,t in align.items()}
                local_s=[adjusted_rand_score(data.source_communities[t], fit.local_source[inv_map[t]]) for t in range(2)]
                local_t=[adjusted_rand_score(data.target_communities[t], fit.local_target[inv_map[t]]) for t in range(2)]
                writer.writerow([seed,contrast,sigma,
                    round(adjusted_rand_score(data.layer_types,fit.layer_types),4),
                    round(adjusted_rand_score(data.layer_types,raw_layer),4),
                    round(adjusted_rand_score(joint_labels(data.source_communities),fit.global_source),4),
                    round(adjusted_rand_score(joint_labels(data.target_communities),fit.global_target),4),
                    round(float(np.mean(local_s)),4),round(float(np.mean(local_t)),4),
                    round(elapsed,3),round(fit.decomposition.relative_residuals[-1],4)])


if __name__=="__main__": main()
