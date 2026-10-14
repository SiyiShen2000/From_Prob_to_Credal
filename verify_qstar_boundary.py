"""Verify q-star trajectory counters without retaining per-event targets."""
import numpy as np
import torch

import credal_gaussian_experiment as exp


def bank(vertices, constraints=False):
    vertices = torch.tensor(vertices, dtype=torch.float32)
    n, vmax, classes = vertices.shape
    if constraints:
        # Full-simplex nonnegativity constraints; softmax predictions are feasible.
        A = -torch.eye(classes).repeat(n, 1, 1)
        b = torch.zeros((n, classes))
        cmask = torch.ones((n, classes), dtype=torch.bool)
    else:
        A = torch.empty((n, 0, classes))
        b = torch.empty((n, 0))
        cmask = torch.empty((n, 0), dtype=torch.bool)
    return exp.VertexBank(vertices, torch.ones((n, vmax), dtype=torch.bool),
                          A, b, cmask, float(vmax), 0., 0.)


def main():
    device = torch.device("cpu")
    x = np.array([[0., 0.], [1., -1.]], dtype=np.float64)
    y = np.array([0, 1])

    ce_stats = {}
    exp.train_mlp(
        x, y, "credal", exp.TrainConfig(epochs=3, batch_size=1,
                                         hidden_dims=(), loss_name="cross_entropy"),
        device, seed=7,
        vertex_bank=bank([[[1., 0., 0.]], [[.2, .3, .5]]]),
        qstar_statistics=ce_stats,
    )
    assert ce_stats["qstar_trajectory_boundary_count"] == 3
    assert ce_stats["qstar_trajectory_evaluated_count"] == 6
    assert ce_stats["qstar_trajectory_boundary_rate"] == .5
    assert ce_stats["qstar_boundary_tolerance"] == 1e-7

    kl_stats = {}
    simplex = [[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]] * 2
    exp.train_mlp(
        x, y, "credal", exp.TrainConfig(epochs=2, batch_size=2,
                                         hidden_dims=(), loss_name="kl"),
        device, seed=9, vertex_bank=bank(simplex, constraints=True),
        qstar_statistics=kl_stats,
    )
    assert kl_stats["qstar_trajectory_boundary_count"] == 0
    assert kl_stats["qstar_trajectory_evaluated_count"] == 4
    assert kl_stats["qstar_trajectory_boundary_rate"] == 0.
    assert all(np.isnan(value) for value in exp.unavailable_qstar_statistics().values())
    print("PASS: CE/KL trajectory counters, N*epochs events, no stored q-star history")


if __name__ == "__main__":
    main()
