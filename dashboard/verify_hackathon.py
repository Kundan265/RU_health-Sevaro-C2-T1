"""Run existing checks and focused readiness regressions without changing demo audit history."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Evidence output directory; defaults to a new temporary directory')
    args = parser.parse_args()
    output = args.output or Path(tempfile.mkdtemp(prefix='sevaro-verification-'))
    output.mkdir(parents=True, exist_ok=True)
    dashboard = Path(__file__).resolve().parent
    scripts = ['check_snapshot.py', 'test_e2e.py', 'tests/test_predictor.py',
               'tests/test_pathway.py', 'tests/test_rank.py', 'tests/test_offer_log.py',
               'tests/test_simulator.py', 'tests/test_pool_sizing.py', 'tests/test_rebase.py',
               'tests/test_compare_provenance.py', 'tests/test_rank_direction.py',
               'tests/test_hackathon_readiness.py', 'tests/test_dashboard_workflow.py']
    results = []
    for script in scripts:
        # Existing tests instantiate ActionStore with its default persistent audit path.
        # Disable that default only in these subprocesses; explicit temporary-path audit
        # tests still run. No predictor/model import path is altered.
        code = ("import runpy,sys;from engine.act import ActionStore;"
                "original=ActionStore.__init__;"
                "ActionStore.__init__=lambda self,*a,**k: original(self,*a,**({'audit_path':None,**k} if not a else k));")
        if script.endswith('test_dashboard_workflow.py'):
            code += f"sys.argv=[{script!r},{str(output/'ui_checks.json')!r}];"
        else:
            code += f"sys.argv=[{script!r}];"
        code += f"runpy.run_path({script!r},run_name='__main__')"
        start = time.perf_counter()
        run = subprocess.run([sys.executable, '-B', '-c', code], cwd=dashboard,
                             text=True, capture_output=True)
        results.append({'script': script, 'exit_code': run.returncode,
                        'automated_seconds': round(time.perf_counter()-start, 3),
                        'output': run.stdout, 'stderr': run.stderr})
        print(f"{'PASS' if run.returncode == 0 else 'FAIL'} {script}", flush=True)
        if run.returncode:
            print(run.stdout[-1500:] + run.stderr[-2500:], flush=True)
    (output/'test_results.json').write_text(json.dumps({
        'evidence_type': 'automated verification; execution times are not human task timings',
        'python': sys.version, 'audit_isolation': 'Default ActionStore audit disabled in test subprocesses only',
        'results': results, 'passed': sum(r['exit_code']==0 for r in results), 'total': len(results),
    }, indent=2))
    print(f'Evidence: {output}')
    return int(any(r['exit_code'] for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
