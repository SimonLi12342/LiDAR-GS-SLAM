"""Build private copies of the measured FastGICP and safe renderer; never pip-install them."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from repaired_profile import ROOT, BUILD, PACKAGE, native_sources, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cuda-home', type=Path, required=True)
    p.add_argument('--arch', required=True, help='Measured L40: 8.9; select the actual target architecture.')
    p.add_argument('--jobs', type=int, default=2)
    args = p.parse_args()
    if args.jobs < 1:
        p.error('--jobs must be positive')
    toolkit = args.cuda_home.resolve()
    if not (toolkit / 'bin/nvcc').is_file():
        p.error('--cuda-home must contain bin/nvcc')
    sources = native_sources()
    BUILD.mkdir(parents=True, exist_ok=False)
    ignore = shutil.ignore_patterns('.git', 'build', '*.so', '*.o', '*.egg-info', '__pycache__')
    shutil.copytree(ROOT / 'submodules/fast_gicp', BUILD / 'fast_gicp', ignore=ignore)
    shutil.copytree(ROOT / 'submodules/diff-surfel-spherical-rasterization', BUILD / 'renderer', ignore=ignore)
    renderer = BUILD / 'renderer'
    original = 'diff_surfel_spherical_rasterization'
    (renderer / original).rename(renderer / PACKAGE)
    setup = renderer / 'setup.py'
    setup.write_text(setup.read_text().replace(original, PACKAGE))
    alias = BUILD / 'mesh_alias' / original
    alias.mkdir(parents=True)
    (alias / '__init__.py').write_text(f'from {PACKAGE} import *\n')
    env = os.environ.copy()
    env.update(CUDA_HOME=str(toolkit), TORCH_CUDA_ARCH_LIST=args.arch,
               MAX_JOBS=str(args.jobs), CMAKE_BUILD_PARALLEL_LEVEL=str(args.jobs))
    for name, paths in [('CPATH', [toolkit / 'targets/x86_64-linux/include', toolkit / 'include']),
                        ('LIBRARY_PATH', [toolkit / 'targets/x86_64-linux/lib', toolkit / 'lib']),
                        ('PATH', [toolkit / 'bin', Path(sys.executable).parent])]:
        env[name] = os.pathsep.join(map(str, paths)) + os.pathsep + env.get(name, '')
    for folder in ['fast_gicp', 'renderer']:
        print(f'Building {folder}; log: {BUILD / (folder + ".log")}', flush=True)
        with (BUILD / (folder + '.log')).open('x') as log:
            subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'],
                           cwd=BUILD / folder, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    binaries = list((BUILD / 'fast_gicp').glob('*.so'))
    binaries += list((BUILD / 'fast_gicp/build').glob('lib*/libfast_gicp.so'))
    binaries += list((renderer / PACKAGE).glob('*.so'))
    if len(list((BUILD / 'fast_gicp').glob('pygicp*.so'))) != 1 or len(list((renderer / PACKAGE).glob('*.so'))) != 1:
        raise RuntimeError('Missing or ambiguous native extension.')
    import torch
    receipt = dict(source_sha256=sources, python=sys.executable, python_version=sys.version,
                   python_abi=sys.implementation.cache_tag, torch=torch.__version__,
                   torch_cuda=torch.version.cuda, toolkit=str(toolkit), arch=args.arch,
                   nvcc=subprocess.check_output([str(toolkit / 'bin/nvcc'), '--version'], text=True),
                   binary_sha256={str(path.relative_to(BUILD)): sha(path) for path in binaries})
    (BUILD / 'build_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('Private native build completed. Existing installed extensions were not replaced.')


if __name__ == '__main__':
    main()
