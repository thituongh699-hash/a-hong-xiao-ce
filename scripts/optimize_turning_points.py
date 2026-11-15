"""Run the project parameter optimizer with a causal training/validation split."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.optimization.job_state import brief_error, file_lock, read_json, write_json


class OptimizationCancelled(Exception):
    pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--out',type=Path,default=ROOT/'reports/turning-point-optimization-2026-10-08')
    parser.add_argument('--max-trials',type=int)
    parser.add_argument('--job-state',type=Path,help='Durable job.json updated by this worker')
    parser.add_argument('--cancel-file',type=Path,help='Cooperative cancellation marker')
    parser.add_argument('--lease-file',type=Path,help='Exclusive worker liveness lock')
    parser.add_argument('--resume',action='store_true',help='Reuse this output directory and its checkpoint if present')
    args=parser.parse_args()
    if any((args.job_state,args.cancel_file,args.lease_file)) and not all((args.job_state,args.cancel_file,args.lease_file)):
        parser.error('--job-state, --cancel-file and --lease-file must be supplied together')

    def run():
        from src.optimization.turning_point import default_config, run_optimization
        config=json.loads(args.config.read_text(encoding='utf-8')) if args.config else default_config()
        if args.max_trials is not None: config['max_trials']=args.max_trials
        last=[time.monotonic()]
        def cancelled():
            if args.cancel_file and args.cancel_file.exists():
                raise OptimizationCancelled('用户已取消优化任务')
        def progress(record,done,total):
            if args.job_state:
                write_json(args.job_state.parent/'progress.json',dict(completed=done,total=total))
            cancelled()  # Every callback, even when progress printing is throttled.
            if done==total or time.monotonic()-last[0]>10:
                print(json.dumps(dict(completed=done,total=total,status=record['status'],score=record['score']),ensure_ascii=False),flush=True)
                last[0]=time.monotonic()
        cancelled()
        summary=run_optimization(config,args.out,progress)
        cancelled()
        print(json.dumps(dict(best_training=summary['best_training'],validation=summary['validation'],metadata=summary['metadata'],elapsed_seconds=summary['elapsed_seconds']),ensure_ascii=False,indent=2),flush=True)

    def finish(status,error=None):
        if args.job_state:
            job=read_json(args.job_state)
            job.update(status=status,finished_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                       return_code=0 if status=='completed' else (2 if status=='cancelled' else 1),error=error)
            write_json(args.job_state,job)

    from contextlib import nullcontext
    lock=file_lock(args.lease_file,timeout=2.0) if args.lease_file else nullcontext()
    with lock:
        if args.job_state:
            job=read_json(args.job_state)
            job['lease_acquired_at']=datetime.now(timezone.utc).isoformat(timespec='seconds')
            write_json(args.job_state,job)
        previous_parameters=None
        try:
            previous_parameters=(args.out/'parameters.json').read_bytes() if args.resume and (args.out/'parameters.json').is_file() else None
            run()
        except OptimizationCancelled as exc:
            finish('cancelled',str(exc))
            print(json.dumps(dict(status='cancelled',error=str(exc)),ensure_ascii=False),flush=True)
            return 2
        except Exception as exc:
            error=brief_error(exc)
            if 'checkpoint fingerprint mismatch' in str(exc) and args.resume:
                parameters=args.out/'parameters.json'
                if previous_parameters is None:
                    parameters.unlink(missing_ok=True)
                else:
                    temp=parameters.with_name('.parameters.restore.tmp')
                    with temp.open('wb') as stream:
                        stream.write(previous_parameters)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temp,parameters)
            finish('failed',error)
            print(json.dumps(dict(status='failed',error=error),ensure_ascii=False),flush=True)
            return 1
        finish('completed')
        return 0


if __name__=='__main__': raise SystemExit(main())
