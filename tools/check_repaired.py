"""Check actual compiled GICP geometry, degenerate CUDA footprints, and upstream numerics."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from repaired_profile import ROOT, PACKAGE, environment, verify_sources, verify_native


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output already exists.')
    verify_sources()
    env = environment(args.gpu)
    native = verify_native(env)
    # Import all native modules only after selecting their absolute private paths.
    os.environ.update(env)
    sys.path[:0] = [str(ROOT)] + env['PYTHONPATH'].split(os.pathsep)
    import importlib
    import numpy as np
    import torch
    import pygicp
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation
    from scene.shared_objs import SharedCam
    import pyprojections as pyp
    torch.set_num_threads(4)
    rng = np.random.default_rng(26)

    def features(points):
        reg = pygicp.FastGICP()
        reg.set_num_threads(4)
        reg.set_input_target(points)
        reg.set_target_filter(len(points), np.arange(1, len(points)+1, dtype=np.int32))
        reg.calculate_target_covariance_with_filter()
        q = np.array(reg.get_target_rotationsq()).reshape(-1, 4)
        scales = np.array(reg.get_target_scales()).reshape(-1, 3)
        return Rotation.from_quat(q).as_matrix(), scales

    plane_errors = []
    for _ in range(10):
        basis = Rotation.random(random_state=rng).as_matrix()
        x, y = np.meshgrid(np.linspace(-4, 4, 31), np.linspace(-3, 3, 29))
        points = (np.column_stack([x.ravel(), y.ravel(), rng.normal(0, .0001, x.size)]) @ basis.T).astype(np.float32)
        rotations, _ = features(points)
        angle = np.rad2deg(np.arccos(np.clip(np.abs(rotations[:, :, 2] @ basis[:, 2]), 0, 1)))
        plane_errors.append(float(np.quantile(angle, .95)))
    assert max(plane_errors) < .05, plane_errors
    points = rng.normal(size=(2000, 3)).astype(np.float32)
    rotations, scales = features(points)
    actual = np.einsum('nij,nj,nkj->nik', rotations, scales**2, rotations)
    neighbours = points[cKDTree(points).query(points, k=25)[1]].astype(np.float64)
    centered = neighbours - neighbours.mean(1, keepdims=True)
    expected = np.einsum('nki,nkj->nij', centered, centered) / 25
    cov_error = float(np.linalg.norm(actual-expected, axis=(1, 2)).max())
    assert cov_error < 1e-5, cov_error

    renderer = importlib.import_module(PACKAGE)
    h, w = 64, 256
    yaw, pitch = np.meshgrid(np.linspace(-3.13, 3.13, 128), np.linspace(-.65, .65, 32))
    shell = np.stack([np.cos(pitch)*np.cos(yaw), np.cos(pitch)*np.sin(yaw), np.sin(pitch)], -1).reshape(-1, 3)*10
    k = pyp.calculate_spherical_intrinsics(shell.T, h, w)[0]
    cam = SharedCam(1., 1., np.zeros((h, w), np.float32), w/2, h/2, w/6.28, h/1.3)
    cam.setup_cam(np.eye(3), np.zeros(3), np.zeros((h, w), np.float32), k)
    cam.on_cuda()
    center = np.array([5., .3, -.8])
    normal = -center / np.linalg.norm(center)
    axis = np.cross([0, 0, 1.], normal)
    q = Rotation.from_rotvec(axis/np.linalg.norm(axis)*np.arccos(normal[2])).as_quat()[[3, 0, 1, 2]]
    def array(value):
        return torch.tensor(np.asarray(value), dtype=torch.float32, device='cuda').contiguous()
    zero_cases = []
    for scale in [0., 1e-20, 1e-12]:
        output = renderer._C.rasterize_gaussians(array([center]), array([[.9]]), array([[scale, scale]]),
                    array([q]), 1., torch.empty(0, device='cuda'), cam.world_view_transform,
                    cam.projection_matrix, h, w, False, True)
        torch.cuda.synchronize()
        assert int(output[0]) == 0 and float(output[1].abs().max()) == 0
        zero_cases.append(scale)
    upstream = subprocess.run([sys.executable, str(ROOT / 'tests/check_slam_numerics.py')],
                              cwd=ROOT, env=env, capture_output=True, text=True)
    if upstream.returncode:
        raise RuntimeError(upstream.stdout + upstream.stderr)
    report = dict(native=native, plane_normal_p95_degrees=plane_errors,
                  covariance_max_frobenius_error=cov_error, zero_footprint_scales=zero_cases,
                  upstream_numerics_stdout=upstream.stdout, passed=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
