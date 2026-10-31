"""Run every CPU test in isolated pytest processes and retain combined evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=30.)
    args = parser.parse_args()
    if args.jobs < 1 or args.max_seconds < 0:
        parser.error('jobs must be positive and max-seconds nonnegative')
    repo = Path(__file__).resolve().parents[1]
    files = sorted(p for p in (repo / 'tests').rglob('*.py')
        if any(fnmatch.fnmatch(p.name, pattern) for pattern in ['test_*.py', '*_test.py']))
    if not files:
        parser.error('No test files found')
    output = args.output.resolve()
    if output.exists():
        parser.error('Choose a fresh output directory to preserve prior evidence')
    output.mkdir(parents=True)
    buckets = [[] for _ in range(min(args.jobs, len(files)))]
    weights = [0] * len(buckets)
    for path in sorted(files, key=lambda p: (-p.stat().st_size, str(p))):
        index = min(range(len(buckets)), key=lambda i: (weights[i], i))
        buckets[index].append(path)
        weights[index] += path.stat().st_size
    assert sorted(p for bucket in buckets for p in bucket) == files
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1', 'PYTHONDONTWRITEBYTECODE': '1',
        'PYTHONPATH': str(repo / 'src') + (os.pathsep + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else '')}
    processes, reports = [], []
    started = time.monotonic()
    for index, bucket in enumerate(buckets):
        xml = output / f'shard-{index}.xml'
        command = [sys.executable, '-m', 'pytest', '-q', '-m', 'not gpu',
            '-o', 'cache_dir=' + str(output / f'cache-{index}'), '--junitxml=' + str(xml),
            *[str(p.relative_to(repo)) for p in sorted(bucket)]]
        processes.append(subprocess.Popen(command, cwd=repo, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
        reports.append({'index': index, 'files': [str(p.relative_to(repo)) for p in sorted(bucket)],
            'junitxml': str(xml), 'command': command})
    try:
        with ThreadPoolExecutor(max_workers=len(processes)) as pool:
            pending = {pool.submit(process.communicate): index for index, process in enumerate(processes)}
            for future in as_completed(pending):
                index = pending[future]
                stdout, _ = future.result()
                (output / f'shard-{index}.log').write_text(stdout)
                reports[index]['returncode'] = processes[index].returncode
                print(f'CPU shard {index + 1}/{len(processes)}: exit {processes[index].returncode}', flush=True)
    except KeyboardInterrupt:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        raise
    seconds = time.monotonic() - started
    cases = []
    for report in reports:
        xml = Path(report['junitxml'])
        if xml.exists():
            cases.extend(ET.parse(xml).getroot().iter('testcase'))
            report['junitxml_sha256'] = hashlib.sha256(xml.read_bytes()).hexdigest()
    identities = [(case.attrib['classname'], case.attrib['name']) for case in cases]
    assert len(identities) == len(set(identities)), 'A test ran in more than one shard'
    failures = sum(any(child.tag in ['failure', 'error'] for child in case) for case in cases)
    skipped = sum(any(child.tag == 'skipped' for child in case) for case in cases)
    tests_passed = all(report['returncode'] == 0 for report in reports) and not failures and bool(cases)
    within_limit = not args.max_seconds or seconds < args.max_seconds
    receipt = {'status': 'passed' if tests_passed and within_limit else 'failed',
        'tests_passed': tests_passed, 'within_time_limit': within_limit, 'wall_seconds': seconds,
        'limit_seconds': args.max_seconds, 'worker_processes': len(processes), 'torch_threads_per_process': 1,
        'tests': len(cases), 'failures': failures, 'skipped': skipped, 'shards': reports,
        'test_source_sha256': {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'scope': 'All repository CPU test files exactly once with the unchanged not-gpu marker. Separate pytest caches, hidden CUDA and one CPU thread per worker; no mocked fitting, skipped durability or reduced test coverage. GPU and real-dataset acceptance remain separate.'}
    (output / 'summary.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(f'{len(cases)} CPU tests, {failures} failures, {skipped} skipped; {seconds:.2f}s with {len(processes)} workers.', flush=True)
    return 0 if tests_passed and within_limit else 1


if __name__ == '__main__':
    sys.exit(main())
