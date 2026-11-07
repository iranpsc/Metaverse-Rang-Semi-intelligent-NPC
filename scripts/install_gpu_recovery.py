#!/usr/bin/env python3
"""Install the host watchdog in the current Docker operator's crontab."""

import argparse
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


def cron_block(project, args):
    job = shlex.join(args).replace("%", "\\%") + " >/dev/null 2>&1"
    return (f"# BEGIN metarang-gpu-recovery {project}\n"
            f"@reboot {job}\n* * * * * {job}\n"
            f"# END metarang-gpu-recovery {project}\n")


def update_crontab(existing, project, block):
    pattern = rf"^# BEGIN metarang-gpu-recovery {re.escape(project)}\n.*?^# END metarang-gpu-recovery {re.escape(project)}\n?"
    cleaned = re.sub(pattern, "", existing, flags=re.MULTILINE | re.DOTALL).rstrip()
    return (cleaned + "\n\n" if cleaned else "") + block


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.project):
        parser.error("invalid Compose project name")
    root = Path(__file__).resolve().parents[1]
    directory = root / ".gpu-recovery"
    directory.mkdir(parents=True, exist_ok=True)
    docker = shutil.which("docker")
    if not docker:
        parser.error("docker is required")
    job = [sys.executable, str(root / "scripts/gpu_recovery.py"), "--project", args.project,
           "--docker", docker, "--state-dir", str(directory), "--watch"]
    # Cron has a minimal PATH; the watchdog also needs the host NVIDIA utilities.
    job = ["/usr/bin/env", "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin", *job]
    block = cron_block(args.project, job)
    candidate = directory / "crontab.candidate"
    if args.prepare_only:
        candidate.write_text(block)
        print(block)
        return
    result = subprocess.run(["crontab", "-l"], text=True, capture_output=True)
    if result.returncode and "no crontab for" not in result.stderr.lower():
        raise RuntimeError(result.stderr.strip())
    existing = result.stdout if result.returncode == 0 else ""
    backup = directory / "crontab.before-install"
    if not backup.exists():
        backup.write_text(existing)
    candidate.write_text(update_crontab(existing, args.project, block))
    subprocess.run(["crontab", str(candidate)], check=True)
    # The file lock makes launching again safe if cron already started a watcher.
    subprocess.Popen(job, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)
    print(f"Installed GPU recovery for {args.project}; checks every 15 seconds.")
    print(f"Pause for maintenance: touch {directory / 'paused'}")
    print(f"Status: {directory / 'status.json'}")


if __name__ == "__main__":
    main()
