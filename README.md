# First-Order Manifold Data Augmentation for Regression Learning

Official implementation of "First-Order Manifold Data Augmentation for Regression Learning" (ICML 2024)

<div align=center><img src="figures/foma.png" width="50%"></div>

## Overview

FOMA is a data-driven approach for data augmentation in regression tasks. Key features:
- Samples new examples from tangent planes of the training distribution
- Aligns with neural networks' tendency to capture dominant features
- Improves generalization across multiple architectures and datasets
- Fully differentiable implementation



<div align=center><img src="figures/NO2_stability.png" width="90%"></div>
## Example Usage

```python
def scale(A, k, lam):
    U, s, Vt = torch.linalg.svd(A, full_matrices=False)
    lam_repeat = lam.repeat(s.shape[-1] - k)
    lam_ones = torch.ones(k)
    s = s * torch.cat((lam_ones, lam_repeat))
    A = U @ torch.diag(s) @ Vt
    return A

# X, Y are in batch x feats
for (X, Y) in loader:
    lam = torch.distributions.beta.Beta(alpha, alpha).sample()
    A = torch.concatenate((X, Y), axis=1)
    A = scale(A, k, lam)
    X, Y = A[:, :n], A[:, n:]
    optimizer.zero_grad()
    loss(net(X), Y).backward()
    optimizer.step()
```

## Datasets

We evaluate on the following datasets:
- Airfoil Self-Noise
- NO2
- Exchange-Rate
- Electricity
- Echo
- RCF
- Crimes
- PovertyMap
- DTI

## Citation

```bibtex
@inproceedings{kaufman2024firstorder,
  title={First-Order Manifold Data Augmentation for Regression Learning},
  author={Kaufman, Ilya and Azencot, Omri},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```
