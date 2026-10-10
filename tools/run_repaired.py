"""Run the accepted Quad Easy Full/KF30 profile with explicit device and provenance."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import yaml
from repaired_profile import ROOT, PROFILE, environment, verify_sources, verify_native, git_receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True, help='Directory containing the NCD ROS bag.')
    p.add_argument('--trajectory', type=Path, required=True, help='TUM trajectory, e.g. gt-nc-quad-easy.csv.')
    p.add_argument('--output', type=Path, required=True, help='New output directory; existing paths are refused.')
    p.add_argument('--gpu', required=True, help='Physical GPU UUID recommended; logical index also accepted.')
    p.add_argument('--frames', type=int, default=0, help='0: full sequence. Positive values: smoke checks only.')
    p.add_argument('--dry-run', action='store_true', help='Validate source/config and show command without loading CUDA.')
    args = p.parse_args()
    if args.frames < 0:
        p.error('--frames must be nonnegative')
    if not args.dataset.is_dir() or not args.trajectory.is_file():
        p.error('Dataset directory and trajectory file must exist.')
    out = args.output.resolve()
    if out.exists():
        p.error('Output already exists; use a new name to preserve earlier runs.')
    hashes = verify_sources()
    cfg = yaml.safe_load((ROOT / 'configs/ncd_repaired_full.yaml').read_text())
    cfg['data']['dataset_path'] = str(args.dataset.resolve())
    cfg['data']['trajectory_reader']['filename'] = str(args.trajectory.resolve())
    cfg['output_path'] = str(out)
    # Validate the frozen scientific configuration independently of local I/O paths.
    reference = yaml.safe_load((ROOT / 'reproduction/evidence/full_final_quality_repeat/input_config.yaml').read_text())
    for c in (cfg, reference):
        c['data']['dataset_path'] = str(args.dataset.resolve())
        c['data']['trajectory_reader']['filename'] = str(args.trajectory.resolve())
        c['output_path'] = str(out)
    if cfg != reference:
        raise RuntimeError('Preset differs from the measured configuration; define a separate profile.')
    env = environment(args.gpu, PROFILE['threads'])
    env['LIDARGS_MAX_FRAMES'] = str(args.frames)
    cmd = [sys.executable, str(ROOT / 'gs_icp_slam.py'), '--config_file', str(out / 'input_config.yaml')]
    receipt = dict(profile=PROFILE['profile'], scope='full_sequence' if not args.frames else 'smoke_only',
                   max_frames=args.frames, command=cmd, source_sha256=hashes, git=git_receipt(),
                   settings=dict(fixes=PROFILE['fixes'], threads=PROFILE['threads'], seed=PROFILE['seed']),
                   environment={key: env.get(key) for key in ['CUDA_VISIBLE_DEVICES', 'CUDA_LAUNCH_BLOCKING',
                                'PYTORCH_CUDA_ALLOC_CONF', 'PYTHONPATH', 'OMP_NUM_THREADS', 'OMP_THREAD_LIMIT',
                                'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']})
    if args.dry_run:
        print(json.dumps(dict(receipt=receipt, config=cfg), indent=2))
        return
    receipt['native'] = verify_native(env)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'input_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    record = out / 'repaired_run.json'
    receipt.update(status='running', started_epoch=time.time())
    record.write_text(json.dumps(receipt, indent=2) + '\n')
    try:
        with (out / 'console.log').open('x') as log:
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        receipt['returncode'] = result.returncode
        receipt['source_unchanged'] = verify_sources() == hashes
        complete = (out / 'results/post_pgo/models/0000.ply').exists() and (out / 'poses_post_pgo.npy').exists()
        log_text = (out / 'console.log').read_text(errors='replace')
        if result.returncode or not complete or 'Traceback (most recent call last)' in log_text:
            raise RuntimeError(f'Run failed or incomplete; inspect {out / "console.log"}')
        receipt['status'] = 'completed'
    except BaseException:
        receipt['status'] = 'failed_or_interrupted'
        raise
    finally:
        receipt['command_wall_s'] = time.time() - receipt['started_epoch']
        record.write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'Results: {out}; scan FPS is in tracking_diagnostics.json (not command throughput).')


if __name__ == '__main__':
    main()
