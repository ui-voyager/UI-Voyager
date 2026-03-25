#!/usr/bin/env python3
"""Android World Parallel Evaluation Script

Supports multi-emulator parallel evaluation.

Usage:
    python run_eval_parallel.py --config eval/configs/Qwen3-VL-4B-Instruct.yaml --num_workers 4
    python run_eval_parallel.py --config eval/configs/Qwen3-VL-4B-Instruct.yaml --num_workers 4 --start_port 5556
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = 'none'

# Get project root and add android_env path to sys.path
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent  # androidworld -> project root
ANDROID_ENV_PATH = PROJECT_ROOT / "android_env"

# Add android_env to sys.path for imports
sys.path.insert(0, str(ANDROID_ENV_PATH))


def setup_logging(log_file: str):
    """Setup logging"""
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    log_fp = open(log_file, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.stdout, log_fp)
    sys.stderr = Tee(sys.stderr, log_fp)
    
    return log_file


def split_tasks(tasks: List[str], num_workers: int) -> List[List[str]]:
    """Split task list into chunks for workers"""
    chunks = [[] for _ in range(num_workers)]
    for i, task in enumerate(tasks):
        chunks[i % num_workers].append(task)
    
    return [c for c in chunks if c]


def get_all_tasks() -> List[str]:
    """Get all available task names"""
    from android_world import registry
    
    task_registry = registry.TaskRegistry()
    reg = task_registry.get_registry(family='android_world')
    return list(reg.keys())


def worker_process(
    worker_id: int,
    tasks: List[str],
    config: Dict[str, Any],
    console_port: int,
    grpc_port: int,
    result_queue: mp.Queue,
    log_dir: str,
    repeat_id: int = 0,
):
    """Worker process - runs a group of tasks"""
    
    print(f'[Worker {worker_id}] Started, handling {len(tasks)} tasks (repeat={repeat_id})')
    print(f'[Worker {worker_id}] Ports: console={console_port}, grpc={grpc_port}')
    print(f'[Worker {worker_id}] Tasks: {tasks}')
    
    worker_config = config.copy()
    worker_config['env'] = config.get('env', {}).copy()
    worker_config['env']['console_port'] = console_port
    worker_config['env']['grpc_port'] = grpc_port
    worker_config['eval'] = config.get('eval', {}).copy()
    worker_config['eval']['tasks'] = tasks
    
    base_output = os.path.expanduser(worker_config['eval'].get('output_path', '~/android_world/runs'))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    worker_config['eval']['checkpoint_dir'] = ''
    worker_config['eval']['output_path'] = os.path.join(base_output, f'repeat_{repeat_id:02d}', f'{timestamp}_worker_{worker_id}')
    
    worker_config['agent'] = config.get('agent', {}).copy()
    
    results = {
        'worker_id': worker_id,
        'repeat_id': repeat_id,
        'tasks': tasks,
        'success': 0,
        'total': 0,
        'failed_tasks': [],
        'success_tasks': [],
        'error': None,
    }
    
    try:
        from eval.clients import get_llm_client
        from eval.runner import EvalRunner
        
        llm_client = get_llm_client(worker_config['llm'])
        print(f'[Worker {worker_id}] LLM client initialized')
        
        runner = EvalRunner(worker_config)
        
        print(f'[Worker {worker_id}] Initializing Android environment...')
        runner.setup_env()
        
        print(f'[Worker {worker_id}] Initializing Agent...')
        runner.setup_agent(llm_client, repeat_id=repeat_id)
        
        print(f'[Worker {worker_id}] Creating task suite...')
        suite = runner.create_suite()
        
        print(f'[Worker {worker_id}] Starting evaluation...')
        episodes = runner.run(suite)
        
        for ep in episodes:
            results['total'] += 1
            task_name = ep.get('task_template', 'unknown')
            is_success = ep.get('is_successful', 0)
            
            if ep.get('exception_info') is None:
                if is_success > 0.5:
                    results['success'] += 1
                    results['success_tasks'].append(task_name)
                else:
                    results['failed_tasks'].append(task_name)
        
        results['episodes'] = episodes
        
        print(f'[Worker {worker_id}] Evaluation completed: {results["success"]}/{results["total"]}')
        
        if episodes:
            print(f'\n[Worker {worker_id}] Detailed statistics:')
            runner.get_results_summary()
        
    except Exception as e:
        import traceback
        results['error'] = traceback.format_exc()
        print(f'[Worker {worker_id}] Error: {e}')
        traceback.print_exc()
    
    finally:
        try:
            runner.close()
        except:
            pass
    
    result_queue.put(results)


def run_parallel_eval(
    config: Dict[str, Any],
    num_workers: int,
    start_port: int,
    tasks: Optional[List[str]] = None,
    log_dir: str = 'logs',
    repeat_id: int = 0,
):
    """Run parallel evaluation (single round)"""
    if tasks is None:
        eval_tasks = config.get('eval', {}).get('tasks')
        if eval_tasks:
            tasks = eval_tasks
        else:
            print('Getting all tasks...')
            tasks = get_all_tasks()
    
    print(f'Total tasks: {len(tasks)}')
    print(f'Num workers: {num_workers}')
    print(f'Current round: {repeat_id}')
    
    task_chunks = split_tasks(tasks, num_workers)
    actual_workers = len(task_chunks)
    
    print(f'Actual workers: {actual_workers}')
    for i, chunk in enumerate(task_chunks):
        print(f'  Worker {i}: {len(chunk)} tasks')
    
    result_queue = mp.Queue()
    
    start_time = time.time()
    
    processes = []
    for i, chunk in enumerate(task_chunks):
        console_port = start_port + i * 2  # 5556, 5558, 5560, ...
        grpc_port = 8554 + i               # 8554, 8555, 8556, ...
        
        p = mp.Process(
            target=worker_process,
            args=(i, chunk, config, console_port, grpc_port, result_queue, log_dir, repeat_id),
        )
        processes.append(p)
    
    print(f'\nStarting {actual_workers} worker processes (repeat={repeat_id})...')
    for p in processes:
        p.start()
        time.sleep(5)
    
    print('Waiting for all processes to complete...\n')
    for p in processes:
        p.join()
    
    all_results = []
    while not result_queue.empty():
        all_results.append(result_queue.get())
    
    total_time = time.time() - start_time
    total_success = sum(r['success'] for r in all_results)
    total_tasks = sum(r['total'] for r in all_results)
    
    print('\n' + '=' * 60)
    print(f'Round {repeat_id} Summary')
    print('=' * 60)
    print(f'Total time: {total_time / 60:.1f} min')
    print(f'Total tasks: {total_tasks}')
    print(f'Successful tasks: {total_success}')
    print(f'Success rate: {total_success / total_tasks * 100:.1f}%' if total_tasks > 0 else 'No valid results')
    print('=' * 60)
    
    all_episodes = []
    for r in all_results:
        if 'episodes' in r and r['episodes']:
            all_episodes.extend(r['episodes'])
    
    if all_episodes:
        print('\n' + '=' * 60)
        print('Detailed Statistics')
        print('=' * 60)
        _print_detailed_statistics(all_episodes, log_dir)
    
    print('\nWorker Results:')
    for r in sorted(all_results, key=lambda x: x['worker_id']):
        status = '✅' if r['error'] is None else '❌'
        print(f"  Worker {r['worker_id']}: {r['success']}/{r['total']} {status}")
        if r['error']:
            print(f"    Error: {r['error'][:200]}...")
    
    all_failed = []
    for r in all_results:
        all_failed.extend(r['failed_tasks'])
    
    if all_failed:
        print(f'\nFailed tasks ({len(all_failed)}):')
        for task in all_failed:
            print(f'  - {task}')
    
    serializable_results = []
    for r in all_results:
        r_copy = r.copy()
        r_copy.pop('episodes', None)
        serializable_results.append(r_copy)
    
    summary_file = os.path.join(log_dir, f'parallel_summary_repeat_{repeat_id}.json')
    with open(summary_file, 'w') as f:
        json.dump({
            'total_time_seconds': total_time,
            'total_tasks': total_tasks,
            'total_success': total_success,
            'success_rate': total_success / total_tasks if total_tasks > 0 else 0,
            'num_workers': actual_workers,
            'repeat_id': repeat_id,
            'worker_results': serializable_results,
        }, f, indent=2)
    print(f'\nSummary saved to: {summary_file}')


def _print_detailed_statistics(episodes: List[Dict[str, Any]], log_dir: str) -> None:
    """Print detailed evaluation statistics"""
    from android_world.suite_utils import process_episodes
    
    result_df = process_episodes(episodes, print_summary=True)
    
    csv_file = os.path.join(log_dir, 'detailed_results.csv')
    result_df.to_csv(csv_file)
    print(f'\nDetailed results saved to: {csv_file}')


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Android World Parallel Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--config', type=str, default='eval/configs/Qwen3-VL-4B-Instruct.yaml', help='Config file path')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--start_port', type=int, default=5556, help='Starting port number')
    parser.add_argument('--tasks', type=str, default=None, help='Task list (comma-separated)')
    parser.add_argument('--tasks_file', type=str, default=None, help='Task list file')
    parser.add_argument('--log_dir', type=str, default='logs', help='Log directory')
    parser.add_argument('--repeat_id', type=int, default=0, help='Current evaluation round ID')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f'parallel_eval_{timestamp}.log')
    setup_logging(log_file)
    
    print('\n' + '=' * 60)
    print('Android World Parallel Evaluation')
    print('=' * 60)
    print(f'Config file: {args.config}')
    print(f'Num workers: {args.num_workers}')
    print(f'Start port: {args.start_port}')
    print(f'Log dir: {args.log_dir}')
    print(f'Current round: {args.repeat_id}')
    print('=' * 60 + '\n')
    
    from eval.configs import load_config, get_default_config
    
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        config = get_default_config()
    
    tasks = None
    if args.tasks_file:
        tasks = []
        with open(args.tasks_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    tasks.extend([t.strip() for t in line.split(',') if t.strip()])
    elif args.tasks:
        tasks = [t.strip() for t in args.tasks.split(',')]
    
    try:
        run_parallel_eval(
            config=config,
            num_workers=args.num_workers,
            start_port=args.start_port,
            tasks=tasks,
            log_dir=args.log_dir,
            repeat_id=args.repeat_id,
        )
    except KeyboardInterrupt:
        print('\nEvaluation interrupted by user')
    except Exception as e:
        print(f'\nEvaluation error: {e}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    main()
