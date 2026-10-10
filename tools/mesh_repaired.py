"""Export a frozen repaired map using the matching renderer and integer rays."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from repaired_profile import ROOT, PROFILE, environment, verify_sources, verify_native, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', required=True)
    p.add_argument('--samples', type=int, choices=[5000, 20000], default=5000)
    p.add_argument('--stage', choices=['pre_pgo', 'post_pgo'], default='post_pgo')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    verify_sources()
    run = args.run.resolve()
    receipt = json.loads((run / 'repaired_run.json').read_text())
    if receipt['status'] != 'completed' or receipt['profile'] != PROFILE['profile']:
        raise RuntimeError('Input must be a completed run of the matching repaired profile.')
    source = run / 'results' / args.stage
    model = source / 'models/0000.ply'
    output = args.output.resolve()
    status = output.with_suffix('.receipt.json')
    log = output.with_suffix('.log')
    if any(p.exists() for p in [output, status, log]):
        p.error('Output or its log/receipt already exists; choose a new filename.')
    env = environment(args.gpu, 8, mesh=True)
    cmd = [sys.executable, str(ROOT / 'Splat-LOAM/run.py'), 'mesh', str(source),
           '--output-filename', str(output), '--kf-samples', str(args.samples)]
    before = sha(model)
    record = dict(command=cmd, samples_per_keyframe=args.samples, integer_rays=True,
                  stage=args.stage, model_sha256=before, profile=PROFILE['profile'])
    if args.dry_run:
        print(json.dumps(record, indent=2))
        return
    record['native'] = verify_native(env)
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log.open('x') as stream:
        result = subprocess.run(cmd, cwd=ROOT / 'Splat-LOAM', env=env, stdout=stream, stderr=subprocess.STDOUT)
    record.update(returncode=result.returncode, export_wall_s=time.time()-start,
                  model_unchanged=sha(model) == before)
    record['success'] = result.returncode == 0 and output.is_file() and record['model_unchanged']
    status.write_text(json.dumps(record, indent=2) + '\n')
    if not record['success']:
        raise RuntimeError(f'Export failed; inspect {log}')
    print(output)


if __name__ == '__main__':
    main()
