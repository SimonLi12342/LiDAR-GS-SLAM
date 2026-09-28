"""Run with rebuilt pygicp on PYTHONPATH, the project runtime dependencies, and EVO."""
from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import pygicp
from scipy.spatial.transform import Rotation
from evo.core import metrics, trajectory
from mp_Tracker import Tracker
from mp_Mapper import Mapper
from scene.gaussian_model import GaussianModel
from utils.loss_utils import mean_or_zero


def check_quaternions():
    points = np.random.default_rng(0).uniform(-3, 3, size=(64, 3))
    scales = np.array([3, 2, 1], dtype=np.float32)
    q_source = Rotation.from_euler('xyz', [20, 35, -40], degrees=True).as_quat().astype(np.float32)
    q_target = Rotation.from_euler('xyz', [-15, 50, 70], degrees=True).as_quat().astype(np.float32)
    reg = pygicp.FastGICP()
    reg.set_num_threads(1)
    reg.set_input_source(points)
    reg.set_input_target(points.copy())
    reg.set_source_covariances_fromqs(np.tile(q_source, len(points)), np.tile(scales, len(points)))
    reg.set_target_covariances_fromqs(np.tile(q_target, len(points)), np.tile(scales, len(points)))
    reg.set_target_control_score(np.ones(len(points), dtype=np.float32))
    np.testing.assert_allclose(reg.align(np.eye(4)), np.eye(4), atol=1e-6)
    # NORMALIZED_ELLIPSE uses the second scale squared as its normalizer.
    diagonal = np.diag(scales.astype(float) ** 2 / float(scales[1]) ** 2)
    rs, rt = Rotation.from_quat(q_source).as_matrix(), Rotation.from_quat(q_target).as_matrix()
    covariance = rs @ diagonal @ rs.T + rt @ diagonal @ rt.T
    expected = len(points) / 1.001 * np.linalg.inv(covariance)
    np.testing.assert_allclose(reg.get_final_hessian()[3:, 3:], expected, rtol=2e-5, atol=1e-5)
    print('PASS: source and target xyzw quaternions produce the expected GICP covariance/Hessian.')


def check_ate():
    gt = np.repeat(np.eye(4)[None], 8, axis=0)
    gt[:, :3, 3] = [[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3],
                    [1, 2, 0], [1, 0, 3], [0, 2, 3], [1, 2, 3]]
    est = gt.copy()
    est[:, :3, 3] += [3, -2, 5]
    est[-1, :3, 3] += [0.2, 0.4, 1]
    ref = trajectory.PosePath3D(poses_se3=gt)
    pred = trajectory.PosePath3D(poses_se3=est)
    pred.align(ref, correct_scale=False)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((ref, pred))
    expected = ape.get_all_statistics()
    assert not np.isclose(expected['rmse'], expected['mean'])
    for cls in (Tracker, Mapper):
        obj = cls.__new__(cls)
        np.testing.assert_allclose(obj.evaluate_ate_rmse(gt, est), expected['rmse'], rtol=1e-6)
    print('PASS: tracker and mapper ATE helpers agree with EVO RMSE, not mean error.')


def check_masked_losses():
    pred = torch.tensor([3., float('nan'), 9., float('nan')], requires_grad=True)
    gt = torch.tensor([1., 0., 3., 0.])
    loss = mean_or_zero((pred - gt)[gt > 0].abs())
    torch.testing.assert_close(loss, torch.tensor(4.))
    loss.backward()
    torch.testing.assert_close(pred.grad, torch.tensor([0.5, 0., 0.5, 0.]))
    pred = torch.ones(4, requires_grad=True)
    empty = pred[torch.zeros(4, dtype=torch.bool)]
    loss = mean_or_zero(empty.abs())
    loss += mean_or_zero(torch.nn.functional.binary_cross_entropy(empty, empty, reduction='none'))
    assert loss.item() == 0
    loss.backward()
    torch.testing.assert_close(pred.grad, torch.zeros(4))
    print('PASS: valid-pixel loss ignores invalid padding and empty depth/alpha selections have zero gradients.')


def check_map_management():
    for enabled in (False, True):
        calls = []
        split_mask = torch.tensor([False, True])
        model = SimpleNamespace(
            get_opacity=torch.tensor([[0.01], [0.8]]),
            _select_control_masks=lambda extent: (split_mask, None, None),
            densify_and_split_control=lambda mask, N: calls.append(('split', mask.clone())),
            prune_points=lambda mask: calls.append(('prune', mask.clone())),
        )
        GaussianModel.densify_and_prune(model, 0, 0.05, None, None, enable_split=enabled)
        assert [kind for kind, _ in calls] == (['split', 'prune'] if enabled else ['prune'])
        torch.testing.assert_close(calls[-1][1], torch.tensor([True, False]))
    if torch.cuda.is_available():
        model = GaussianModel(0)
        model._xyz = torch.zeros((128, 3), device='cuda')
        model.control_score = torch.zeros(128, device='cuda')
        split, clone, _ = model._select_control_masks(None)
        assert not split.any() and not clone.any()
        model.control_score[:64] = torch.linspace(0.1, 1, 64, device='cuda')
        split, clone, info = model._select_control_masks(None)
        assert not split[64:].any() and not clone[64:].any()
        np.testing.assert_allclose(info['t_split'], 0.955, atol=1e-6)
        print('PASS: zero map-control sentinels do not affect split quantiles or get selected.')
    else:
        print('SKIP: CUDA-only control-score selection check (no GPU).')
    print('PASS: opacity pruning runs both with and without optional splitting.')


if __name__ == '__main__':
    check_quaternions()
    check_ate()
    check_masked_losses()
    check_map_management()
