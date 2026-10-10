"""Explicit opt-in repairs for c51abf3; never used by S2b."""
import json
import os
from pathlib import Path
import numpy as np
import torch

FIXES = frozenset(filter(None, os.environ.get('LIDARGS_FIXES', '').split(',')))


def enabled(name):
    return name in FIXES


def xyzw_to_wxyz(q):
    return q[..., [3, 0, 1, 2]]


def record(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


@torch.no_grad()
def transform_map(model, frame_ids, deltas):
    """Same source-keyframe rigid correction in O(N+K), not O(N*K)."""
    from scipy.spatial.transform import Rotation
    ids = np.asarray(frame_ids, dtype=np.int64)
    assert len(np.unique(ids)) == len(ids), 'each source keyframe must occur once'
    ids_t = model.keyframe_idx.squeeze(-1).long()
    size = max(int(ids.max()) + 1, int(ids_t.max()) + 1)
    lookup = torch.full((size,), -1, dtype=torch.long, device=ids_t.device)
    lookup[torch.as_tensor(ids, device=ids_t.device)] = torch.arange(len(ids), device=ids_t.device)
    selected = lookup[ids_t]
    assert bool((selected >= 0).all()), 'missing source-keyframe transform'
    deltas = np.asarray(deltas)
    transforms = torch.as_tensor(deltas, dtype=model._xyz.dtype, device=model._xyz.device)
    rotations = torch.as_tensor(Rotation.from_matrix(deltas[:, :3, :3]).as_quat(),
                               dtype=model._rotation.dtype, device=model._rotation.device)
    xyz, quat = model._xyz.detach().clone(), model.get_rotation.detach().clone()
    for start in range(0, len(xyz), 250000):
        stop = min(len(xyz), start + 250000)
        idx = selected[start:stop]
        tr = transforms[idx]
        xyz[start:stop] = torch.bmm(tr[:, :3, :3], xyz[start:stop, :, None]).squeeze(-1) + tr[:, :3, 3]
        left = rotations[idx]
        right = quat[start:stop]
        av, aw = left[:, :3], left[:, 3:]
        bv, bw = right[:, :3], right[:, 3:]
        product = torch.cat((aw*bv + bw*av + torch.linalg.cross(av, bv),
                             aw*bw - (av*bv).sum(-1, keepdim=True)), -1)
        quat[start:stop] = torch.nn.functional.normalize(product, dim=-1)
    # Upstream resets full-map moments on every KF transform; one reset is equivalent.
    for name, value in [('xyz', xyz), ('rotation', quat)]:
        group = next(g for g in model.optimizer.param_groups if g['name'] == name)
        old = group['params'][0]
        state = model.optimizer.state.pop(old, None)
        new = torch.nn.Parameter(value)
        group['params'][0] = new
        if state is not None:
            state['exp_avg'] = torch.zeros_like(value)
            state['exp_avg_sq'] = torch.zeros_like(value)
            model.optimizer.state[new] = state
        setattr(model, '_' + name, new)

def orient_towards_sensor(q, points, sensor_position):
    """Resolve eigenvector sign only; preserve covariance and the tangent plane."""
    from scipy.spatial.transform import Rotation
    q = np.asarray(q).copy()
    normals = Rotation.from_quat(q).as_matrix()[:, :, 2]
    flip = np.einsum('ij,ij->i', normals, np.asarray(points) - sensor_position) > 0
    old = q[flip].copy()
    # Right multiply by an X-axis half turn: signs of tangent-Y and normal-Z flip.
    q[flip] = old[:, [3, 2, 1, 0]] * np.array([1, 1, -1, -1])
    return q


