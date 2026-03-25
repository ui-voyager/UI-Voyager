# coding=utf-8
# Copyright 2025 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A class to manage and control an external ADB process."""

import os
import subprocess
import time

from absl import logging
from android_env.components import config_classes
from android_env.components import errors


class AdbController:
  """Manages communication with adb."""

  def __init__(self, config: config_classes.AdbControllerConfig):
    """Instantiates an AdbController object."""

    self._config = config
    logging.info('config: %r', self._config)

    if not self._config.use_adb_server_port_from_os_env:
      # Unset problematic environment variables. ADB commands will fail if these
      # are set. They are normally exported by AndroidStudio.
      if 'ANDROID_HOME' in os.environ:
        logging.info('Removing ANDROID_HOME from os.environ')
        del os.environ['ANDROID_HOME']
      if 'ANDROID_ADB_SERVER_PORT' in os.environ:
        logging.info('Removing ANDROID_ADB_SERVER_PORT from os.environ')
        del os.environ['ANDROID_ADB_SERVER_PORT']

    # Explicitly expand the $HOME environment variable.
    self._os_env_vars = dict(os.environ).copy()
    self._os_env_vars.update(
        {'HOME': os.path.expandvars(self._os_env_vars.get('HOME', ''))}
    )
    logging.info('self._os_env_vars: %r', self._os_env_vars)

  def command_prefix(self, include_device_name: bool = True) -> list[str]:
    """The command for instantiating an adb client to this server."""
    if self._config.use_adb_server_port_from_os_env:
      # When using the adb server port set from the OS environment, we don't
      # need to pass the port explicitly.
      adb_port_args = []
    else:
      # When using the adb server port set from the config, we need to pass the
      # port explicitly.
      adb_port_args = ['-P', str(self._config.adb_server_port)]
    command_prefix = [
        self._config.adb_path,
        *adb_port_args,
    ]
    if include_device_name:
      command_prefix.extend(['-s', self._config.device_name])
    return command_prefix

  def init_server(self, timeout: float | None = None):
    """Initialize the ADB server deamon on the given port.

    This function should be called immediately after initializing the first
    adb_controller, and before launching the simulator.

    Args:
      timeout: A timeout to use for this operation. If not set the default
        timeout set on the constructor will be used.
    """
    # Make an initial device-independent call to ADB to start the deamon.
    self.execute_command(['devices'], timeout, device_specific=False)
    time.sleep(0.2)

  def _execute_server_command(
      self,
      args: list[str],
      timeout: float | None = None,
      max_retries: int = 3,
  ) -> bool:
    """Execute a simple ADB server command with retries.

    Args:
      args: ADB command arguments, e.g. ['devices'], ['kill-server'], ['start-server'].
      timeout: Operation timeout, default 5 seconds (these commands are typically fast).
      max_retries: Maximum number of retries.

    Returns:
      Whether the execution was successful.
    """
    command = self.command_prefix(include_device_name=False) + args

    for attempt in range(max_retries):
      try:
        subprocess.check_output(
            command,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            env=self._os_env_vars,
        )
        return True
      except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logging.warning('Command %s attempt %d/%d failed: %s', args, attempt + 1, max_retries, e)
        time.sleep(0.5)

    return False

  def _debug_adb_status(self):
    """Diagnose ADB issues from the system shell level when the devices command fails."""
    import os
    import threading
    
    adb_path = self._config.adb_path
    adb_port = self._config.adb_server_port
    device_name = self._config.device_name
    pid = os.getpid()
    
    logging.error('=== ADB Debug Info Start (port=%s, device=%s, pid=%s) ===', adb_port, device_name, pid)
    
    # ============================================================
    # Phase 1: Pure Python diagnostics (no fork() dependency, works under high load)
    # ============================================================
    
    # First print the Python process's own status (no subprocess needed, won't hang)
    try:
      thread_count = threading.active_count()
      logging.error('[Python Process]: pid=%s, threads=%s', pid, thread_count)
    except Exception as e:
      logging.error('[Python Process]: Error getting info - %s', e)
    
    # Try to read this process's fd count (pure Python, no subprocess calls)
    try:
      fd_path = f'/proc/{pid}/fd'
      if os.path.exists(fd_path):
        fd_count = len(os.listdir(fd_path))
        logging.error('[Python FD count]: %s', fd_count)
    except Exception as e:
      logging.error('[Python FD count]: Error - %s', e)
    
    # === Key: Check system process count (pure Python), this is the main cause of slow fork() ===
    system_overloaded = False
    total_process_count = 0
    try:
      with open('/proc/loadavg', 'r') as f:
        load_info = f.read().strip()
        logging.error('[Load Average]: %s', load_info)
        # 解析进程数：格式为 "load1 load5 load15 running/total last_pid"
        parts = load_info.split()
        if len(parts) >= 4:
          running_total = parts[3].split('/')
          if len(running_total) == 2:
            running = int(running_total[0])
            total_process_count = int(running_total[1])
            logging.error('[Process Count]: running=%d, total=%d', running, total_process_count)
            if total_process_count > 10000:
              logging.error('[CRITICAL WARNING]: Very high process count (%d)! This causes fork() slowdown and subprocess timeouts.', total_process_count)
              system_overloaded = True
            if total_process_count > 30000:
              logging.error('[CRITICAL WARNING]: Process count extremely high (%d)! System likely has zombie process leak. fork() will be extremely slow.', total_process_count)
    except Exception as e:
      logging.error('[Load Average]: Failed - %s', e)
    
    # 检查 PID 限制
    try:
      with open('/proc/sys/kernel/pid_max', 'r') as f:
        pid_max = int(f.read().strip())
        logging.error('[PID Max]: %d (current usage: %.1f%%)', pid_max, 100.0 * total_process_count / pid_max if pid_max > 0 else 0)
        if total_process_count > pid_max * 0.8:
          logging.error('[CRITICAL WARNING]: Approaching PID limit! (%d/%d = %.1f%%)', total_process_count, pid_max, 100.0 * total_process_count / pid_max)
    except Exception as e:
      logging.error('[PID Max]: Failed - %s', e)
    
    # 检查线程限制
    try:
      with open('/proc/sys/kernel/threads-max', 'r') as f:
        threads_max = int(f.read().strip())
        logging.error('[Threads Max]: %d', threads_max)
    except Exception as e:
      logging.error('[Threads Max]: Failed - %s', e)
    
    # 检查当前用户的进程数限制
    try:
      import resource
      soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
      logging.error('[RLIMIT_NPROC]: soft=%s, hard=%s', soft, hard)
    except Exception as e:
      logging.error('[RLIMIT_NPROC]: Failed - %s', e)
    
    # 统计当前用户的进程数和僵尸进程数（纯 Python）
    try:
      uid = os.getuid()
      proc_count = 0
      zombie_count = 0
      zombie_pids = []
      for p in os.listdir('/proc'):
        if p.isdigit():
          try:
            with open(f'/proc/{p}/status', 'r') as f:
              proc_uid = None
              is_zombie = False
              for line in f:
                if line.startswith('Uid:'):
                  proc_uid = int(line.split()[1])
                elif line.startswith('State:') and 'Z' in line:
                  is_zombie = True
                if proc_uid is not None and (is_zombie or 'State:' in line):
                  break
              if proc_uid == uid:
                proc_count += 1
                if is_zombie:
                  zombie_count += 1
                  if len(zombie_pids) < 20:
                    zombie_pids.append(p)
          except:
            pass
      logging.error('[User %d Processes]: total=%d, zombies=%d', uid, proc_count, zombie_count)
      if zombie_count > 100:
        logging.error('[CRITICAL WARNING]: High zombie count (%d)! This indicates child processes are not being reaped properly.', zombie_count)
      if zombie_pids:
        logging.error('[Zombie PIDs (first 20)]: %s', ', '.join(zombie_pids))
    except Exception as e:
      logging.error('[User Processes]: Failed - %s', e)
    
    # 检查当前 Python 进程的所有线程信息
    try:
      threads = threading.enumerate()
      logging.error('[Python Threads]: count=%d', len(threads))
      for t in threads[:10]:  # 只打印前 10 个
        logging.error('  - Thread: name=%s, daemon=%s, alive=%s', t.name, t.daemon, t.is_alive())
      if len(threads) > 10:
        logging.error('  ... and %d more threads', len(threads) - 10)
    except Exception as e:
      logging.error('[Python Threads]: Failed - %s', e)
    
    # 检查内存信息（纯 Python）
    try:
      with open('/proc/meminfo', 'r') as f:
        lines = f.readlines()[:5]
        logging.error('[Memory Info]: %s', ' | '.join(l.strip() for l in lines))
    except Exception as e:
      logging.error('[Memory Info]: Failed - %s', e)
    
    # 检查 CPU 利用率（通过两次采样计算，纯 Python）
    try:
      def read_cpu_stats():
        """读取 /proc/stat 获取 CPU 统计信息"""
        cpu_stats = {}
        with open('/proc/stat', 'r') as f:
          for line in f:
            if line.startswith('cpu'):
              parts = line.split()
              cpu_name = parts[0]
              # user, nice, system, idle, iowait, irq, softirq, steal
              values = [int(x) for x in parts[1:9]] if len(parts) >= 9 else [int(x) for x in parts[1:]]
              cpu_stats[cpu_name] = values
        return cpu_stats
      
      def sort_cpu_key(name):
        """按数字顺序排序 CPU 核心"""
        if name == 'cpu':
          return (-1, 0)  # 总 CPU 排最前面
        try:
          return (0, int(name[3:]))  # cpu0, cpu1, cpu2...
        except:
          return (1, 0)
      
      # 第一次采样
      stats1 = read_cpu_stats()
      time.sleep(0.5)  # 等待 0.5 秒
      # 第二次采样
      stats2 = read_cpu_stats()
      
      cpu_usage = {}  # cpu_name -> usage%
      cpu_iowait = {}  # cpu_name -> iowait%
      cpu_sys = {}  # cpu_name -> system%
      cpu_user = {}  # cpu_name -> user%
      for cpu_name in stats1.keys():
        if cpu_name in stats2:
          v1, v2 = stats1[cpu_name], stats2[cpu_name]
          idle1 = v1[3]
          idle2 = v2[3]
          iowait1 = v1[4] if len(v1) > 4 else 0
          iowait2 = v2[4] if len(v2) > 4 else 0
          sys1 = v1[2]
          sys2 = v2[2]
          user1 = v1[0] + v1[1]  # user + nice
          user2 = v2[0] + v2[1]
          total1 = sum(v1)
          total2 = sum(v2)
          
          total_delta = total2 - total1
          
          if total_delta > 0:
            idle_delta = idle2 - idle1
            iowait_delta = iowait2 - iowait1
            sys_delta = sys2 - sys1
            user_delta = user2 - user1
            
            # 实际 CPU 使用率（排除 idle 和 iowait）
            usage = 100.0 * (1.0 - (idle_delta + iowait_delta) / total_delta)
            cpu_usage[cpu_name] = usage
            # I/O 等待占比
            cpu_iowait[cpu_name] = 100.0 * iowait_delta / total_delta
            # 系统态占比
            cpu_sys[cpu_name] = 100.0 * sys_delta / total_delta
            # 用户态占比
            cpu_user[cpu_name] = 100.0 * user_delta / total_delta
      
      # 按数字顺序排序
      sorted_cpus = sorted(cpu_usage.keys(), key=sort_cpu_key)
      
      # 打印总 CPU 概览（包括 iowait）
      if 'cpu' in cpu_usage:
        logging.error('[CPU Total]: usage=%.1f%%, iowait=%.1f%%, sys=%.1f%%, user=%.1f%%', 
                     cpu_usage['cpu'], cpu_iowait['cpu'], cpu_sys['cpu'], cpu_user['cpu'])
      
      # 打印所有核心（每行 8 个）
      cores = [c for c in sorted_cpus if c != 'cpu']
      logging.error('[CPU Cores]: %d cores total', len(cores))
      
      for i in range(0, len(cores), 8):
        batch = cores[i:i+8]
        line = ' | '.join(f'{c[3:]}:{cpu_usage[c]:.0f}%/io:{cpu_iowait[c]:.0f}%' for c in batch)
        logging.error('[CPU %03d-%03d]: %s', i, min(i+7, len(cores)-1), line)
      
      # 统计高负载核心（>80%）
      high_load_cores = [c for c in cores if cpu_usage[c] > 80]
      if high_load_cores:
        logging.error('[High Load Cores (>80%%)]: %d cores - %s', 
                     len(high_load_cores), 
                     ', '.join(f'{c}:{cpu_usage[c]:.0f}%' for c in high_load_cores[:20]))
        if len(high_load_cores) > 20:
          logging.error('[High Load Cores]: ... and %d more', len(high_load_cores) - 20)
      
      # 统计高 I/O 等待核心（>20%）
      high_iowait_cores = [c for c in cores if cpu_iowait[c] > 20]
      if high_iowait_cores:
        logging.error('[High IOWait Cores (>20%%)]: %d cores - %s', 
                     len(high_iowait_cores), 
                     ', '.join(f'{c}:io{cpu_iowait[c]:.0f}%' for c in high_iowait_cores[:20]))
        if len(high_iowait_cores) > 20:
          logging.error('[High IOWait Cores]: ... and %d more', len(high_iowait_cores) - 20)
    except Exception as e:
      logging.error('[CPU Usage]: Failed - %s', e)
    
    # 检查是否有僵尸子进程（纯 Python）
    try:
      child_pids = f'/proc/{pid}/task/{pid}/children'
      if os.path.exists(child_pids):
        with open(child_pids, 'r') as f:
          children = f.read().strip()
          logging.error('[Child processes]: %s', children or '(none)')
    except Exception as e:
      logging.error('[Child processes]: Error - %s', e)
    
    # ============================================================
    # Phase 2: If the system is not overloaded, try using subprocess for more info
    # ============================================================
    
    if system_overloaded:
      logging.error('[System Status]: System overloaded (process count: %d), skipping subprocess-based diagnostics to avoid further slowdown.', total_process_count)
      logging.error('[Recommendation]: Check for zombie process leaks. Run: ps aux | awk \'{print $8}\' | grep -c Z')
      logging.error('[Recommendation]: Check which process is creating too many children.')
      logging.error('=== ADB Debug Info End (pure Python diagnostics only) ===')
      return
    
    # 检测 subprocess 是否能正常工作
    system_responsive = False
    try:
      result = subprocess.run(['echo', 'test'], capture_output=True, timeout=2.0)
      system_responsive = True
      logging.error('[Subprocess Check]: OK - subprocess is working')
    except subprocess.TimeoutExpired:
      logging.error('[Subprocess Check]: CRITICAL - even "echo test" timed out! Subprocess may be broken.')
    except Exception as e:
      logging.error('[Subprocess Check]: Error - %s', e)
    
    if not system_responsive:
      logging.error('=== ADB Debug Info End (subprocess broken) ===')
      return
    
    # 按优先级排序：先执行轻量命令，后执行复杂命令
    debug_commands = [
        # === ADB 进程检查 ===
        ('ADB processes', 'ps aux | grep adb | grep -v grep', 10.0),
        ('ADB server on port %s' % adb_port, f'ps aux | grep "adb" | grep "\\-P {adb_port}" | grep -v grep', 10.0),
        
        # === 端口检查 ===
        ('Port %s status' % adb_port, f'ss -tlnp 2>/dev/null | grep ":{adb_port}" || netstat -tlnp 2>/dev/null | grep ":{adb_port}"', 10.0),
        ('All ADB ports', 'ss -tlnp 2>/dev/null | grep adb || netstat -tlnp 2>/dev/null | grep adb', 10.0),
        
        # === ADB 命令检查（可能较慢）===
        ('ADB -P %s devices' % adb_port, f'{adb_path} -P {adb_port} devices 2>&1', 15.0),
        ('Device %s status' % device_name, f'{adb_path} -P {adb_port} -s {device_name} get-state 2>&1', 15.0),
        
        # === 模拟器检查 ===
        ('Emulator processes', 'ps aux | grep -E "emulator|qemu" | grep -v grep', 10.0),
    ]
    
    timeout_count = 0
    for desc, cmd, cmd_timeout in debug_commands:
      try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=cmd_timeout,
            env=self._os_env_vars,
        )
        output = result.stdout.strip() or result.stderr.strip() or '(empty)'
        logging.error('[%s]: %s', desc, output)
      except subprocess.TimeoutExpired:
        timeout_count += 1
        logging.error('[%s]: Command timed out (%.1fs)', desc, cmd_timeout)
        # 如果连续多个命令超时，可能系统已经过载，提前退出
        if timeout_count >= 3:
          logging.error('[System Check]: Multiple commands timed out, system may be overloaded. Skipping remaining checks.')
          break
      except Exception as e:
        logging.error('[%s]: Error - %s', desc, e)
    
    logging.error('=== ADB Debug Info End ===')

  def _restart_server(self, timeout: float | None = None):
    """Kills and restarts the adb server."""
    logging.info('Restarting adb server.')

    # If the server is running, kill it first then start
    if self._execute_server_command(['devices'], timeout=10.0):
      if self._execute_server_command(['kill-server'], timeout):
        time.sleep(0.2)
        self._execute_server_command(['start-server'], timeout)
        time.sleep(2.0)
    else:
      # Server not running or unresponsive, debug and diagnose first
      self._debug_adb_status()
      # Start directly
      self._execute_server_command(['start-server'], timeout)
      time.sleep(2.0)

    # 最终验证
    if self._execute_server_command(['devices'], timeout=10.0):
      logging.info('ADB server restart completed successfully.')
      time.sleep(0.2)
    else:
      logging.warning('ADB server restart may have failed.')

  def _log_command_error(
      self,
      error_type: str,
      n_tries: int,
      max_retries: int,
      command_str: str,
      exception: subprocess.CalledProcessError | subprocess.TimeoutExpired,
  ) -> None:
    """Log command execution errors."""
    logging.exception(
        'ADB command %s (try %r of %r): [%s]',
        error_type,
        n_tries,
        max_retries,
        command_str,
    )
    if exception.stdout is not None:
      logging.error('**stdout**:')
      for line in exception.stdout.splitlines():
        logging.error('    %s', line)
    if exception.stderr is not None:
      logging.error('**stderr**:')
      for line in exception.stderr.splitlines():
        logging.error('    %s', line)

  def _reap_zombies(self):
    """Proactively reap zombie child processes to prevent process accumulation"""
    import os
    try:
      reaped = 0
      while True:
        pid, status = os.waitpid(-1, os.WNOHANG)
        if pid == 0:
          break
        reaped += 1
      if reaped > 0:
        logging.debug('Reaped %d zombie processes', reaped)
    except (ChildProcessError, OSError):
      pass  # 没有子进程了

  def execute_command(
      self,
      args: list[str],
      timeout: float | None = None,
      device_specific: bool = True,
  ) -> bytes:
    """Executes an adb command.

    Args:
      args: A list of strings representing each adb argument. For example:
        ['install', '/my/app.apk']
      timeout: A timeout to use for this operation. If not set the default
        timeout set on the constructor will be used.
      device_specific: Whether the call is device-specific or independent.

    Returns:
      The output of running such command as a binary string.
    """
    timeout = self._config.default_timeout if timeout is None else timeout
    command = self.command_prefix(include_device_name=device_specific) + args
    command_str = 'adb ' + ' '.join(command[1:])

    max_retries = self._config.max_retries
    n_tries = 1
    latest_error = None
    last_error_type = None  # Record the error type from the last attempt
    while n_tries <= max_retries:
      try:
        logging.info('Executing ADB command: [%s]', command_str)
        cmd_output = subprocess.check_output(
            command,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            env=self._os_env_vars,
        )
        logging.debug('ADB command output: %s', cmd_output)
        if n_tries > 1:
          logging.warning('[%s] ADB command succeeded after %d retries: [%s]', last_error_type, n_tries - 1, command_str)
        # 每次成功执行后尝试回收僵尸进程
        self._reap_zombies()
        return cmd_output
      except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        is_timeout = isinstance(e, subprocess.TimeoutExpired)
        error_type = 'timed out' if is_timeout else 'execution failed'
        last_error_type = 'timeout' if is_timeout else 'CalledProcessError'
        
        self._log_command_error(error_type, n_tries, max_retries, command_str, e)
        n_tries += 1
        latest_error = e
        
        # Only restart ADB server on timeout
        if is_timeout and device_specific and n_tries <= max_retries:
          self._restart_server(timeout=timeout)

    raise errors.AdbControllerError(
        f'Error executing adb command: [{command_str}]\n'
        f'Caused by: {latest_error}\n'
        f'adb stdout: [{latest_error.stdout}]\n'
        f'adb stderr: [{latest_error.stderr}]'
    ) from latest_error
