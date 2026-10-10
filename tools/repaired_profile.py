"""Portable paths and provenance for the frozen, explicitly repaired profile."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROFILE = json.loads((ROOT / 'reproduction/profile.json').read_text())
BUILD = ROOT / 'build/repaired_native'
PACKAGE = PROFILE['native_package']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources():
    for name, digest in PROFILE['source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f'Frozen source differs: {name}. Create a new profile for algorithm changes.')
    return PROFILE['source_sha256']


def native_sources():
    return {name: digest for name, digest in verify_sources().items() if name.startswith('submodules/')}


def environment(gpu, threads=16, mesh=False):
    env = os.environ.copy()
    if env.get('CUDA_LAUNCH_BLOCKING') not in (None, '', '0'):
        raise RuntimeError('CUDA_LAUNCH_BLOCKING changes timing; unset it for this profile.')
    paths = [BUILD / 'fast_gicp', BUILD / 'renderer']
    if mesh:
        paths.insert(0, BUILD / 'mesh_alias')
    env.update(CUDA_VISIBLE_DEVICES=gpu, LIDARGS_FIXES=','.join(PROFILE['fixes']),
               LIDARGS_MAX_FRAMES='0', LIDARGS_SAVE_PRE='1', LIDARGS_MESH_INTEGER_RAYS='1',
               OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
               MKL_NUM_THREADS=str(threads), PYTHONUNBUFFERED='1', MPLBACKEND='Agg')
    # An inherited hard thread limit can silently invalidate the selected budget.
    env.pop('OMP_THREAD_LIMIT', None)
    if mesh:
        env['OMP_THREAD_LIMIT'] = str(threads)
    env['PYTHONPATH'] = os.pathsep.join(map(str, paths)) + os.pathsep + env.get('PYTHONPATH', '')
    return env


def verify_native(env):
    receipt = json.loads((BUILD / 'build_receipt.json').read_text())
    if receipt['source_sha256'] != native_sources():
        raise RuntimeError('Native source changed; rebuild in a fresh build/repaired_native directory.')
    if receipt['python_abi'] != sys.implementation.cache_tag:
        raise RuntimeError('Native build Python ABI mismatch.')
    for name, digest in receipt['binary_sha256'].items():
        if sha(BUILD / name) != digest:
            raise RuntimeError(f'Native binary changed: {name}')
    probe = ('import torch, pygicp, importlib, json; '
             f'm=importlib.import_module({PACKAGE!r}); '
             'print(json.dumps({"pygicp":pygicp.__file__,"renderer":m._C.__file__, '
             '"cuda_available":torch.cuda.is_available(), '
             '"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))')
    result = json.loads(subprocess.check_output([sys.executable, '-c', probe],
                                               cwd=ROOT, env=env, text=True))
    for key, folder in [('pygicp', BUILD / 'fast_gicp'), ('renderer', BUILD / 'renderer' / PACKAGE)]:
        if Path(result[key]).resolve().parent != folder.resolve():
            raise RuntimeError(f'Wrong {key} loaded: {result[key]}')
    if not result['cuda_available']:
        raise RuntimeError('No CUDA device available in the selected execution environment.')
    return result


def git_receipt():
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    return {'commit': git('rev-parse', 'HEAD'), 'status': git('status', '--porcelain'),
            'diff_sha256': hashlib.sha256(git('diff', 'HEAD').encode()).hexdigest()}
