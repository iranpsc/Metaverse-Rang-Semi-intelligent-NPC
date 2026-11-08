#!/usr/bin/env python3
"""Host watchdog for existing Compose GPU services; uses only the standard library."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import shutil
import subprocess
import time


SERVICES = ("ollama", "llm", "tts", "stt", "vc")
LOG = logging.getLogger("gpu-recovery")


def command(args, timeout=30):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip()[:1000])
    return result.stdout


def uptime(container, now):
    started = container["State"].get("StartedAt", "")
    try:
        return now - datetime.fromisoformat(started.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0


def recovery_reason(container, now):
    state = container["State"]
    status = state["Status"]
    if status == "exited":
        # Respect normal/manual shutdowns. Use the pause marker before maintenance
        # involving SIGKILL, since Docker does not expose its manual-stop flag.
        if state.get("Error") or state.get("OOMKilled") or state.get("ExitCode", 0) not in (0, 143):
            return "abnormal exit: " + (state.get("Error") or str(state.get("ExitCode")))
    if status == "running":
        health = state.get("Health", {})
        start_period = container.get("Config", {}).get("Healthcheck", {}).get("StartPeriod", 0) / 1e9
        age = uptime(container, now)
        if age >= max(120, start_period) and (
            health.get("Status") == "unhealthy" or health.get("FailingStreak", 0) >= 3
        ):
            return "readiness checks failing"
        if health.get("Status") == "starting" and age >= max(600, start_period + 120):
            return "model startup timed out"
    return None


def prepare_gpu():
    nodes = (Path("/dev/nvidiactl"), Path("/dev/nvidia-uvm"), Path("/dev/nvidia-uvm-tools"))
    if not all(path.is_char_device() for path in nodes):
        helper = shutil.which("nvidia-modprobe")
        if not helper:
            raise RuntimeError("nvidia-modprobe is required to restore missing GPU devices")
        command([helper, "-u", "-c=0"], timeout=15)
    if not all(path.is_char_device() for path in nodes):
        raise RuntimeError("NVIDIA device nodes are still missing")
    command(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], timeout=10)


def containers(docker, project, services):
    ids = command([docker, "ps", "-aq", "--filter", f"label=com.docker.compose.project={project}"]).split()
    if not ids:
        return []
    result = json.loads(command([docker, "inspect", *ids]))
    return [c for c in result if c.get("Config", {}).get("Labels", {}).get("com.docker.compose.service") in services
            and c["Config"]["Labels"].get("com.docker.compose.oneoff", "False").lower() != "true"]


def recover(items, state, docker, *, now, dry_run=False):
    """Return observations and update retry state; never restart healthy services."""
    observations = []
    available = {c["Config"]["Labels"]["com.docker.compose.service"]: c for c in items}
    gpu_ready = None
    # Docker can also remain in its own restart loop. Repair host devices while
    # leaving that loop in charge of the actual restart.
    if not dry_run and any(c["State"]["Status"] == "restarting" for c in items):
        try:
            prepare_gpu()
            gpu_ready = True
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            gpu_ready = False
            LOG.error("GPU initialization during Docker restart failed: %s", error)
    for service in SERVICES:
        container = available.get(service)
        if not container:
            continue
        key = container["Id"]
        record = state.setdefault(key, {})
        health = container["State"].get("Health", {}).get("Status")
        reason = recovery_reason(container, now)
        observation = {"service": service, "status": container["State"]["Status"], "health": health, "reason": reason}
        observations.append(observation)
        if health == "healthy" and not reason:
            record.setdefault("healthy_since", now)
            if now - record["healthy_since"] >= 300:
                record["attempts"] = 0
                record["retry_after"] = 0
        else:
            record.pop("healthy_since", None)
        if not reason or now < record.get("retry_after", 0):
            continue
        if service == "llm":
            dependency = available.get("ollama")
            if dependency and (dependency["State"]["Status"] != "running" or dependency["State"].get("Health", {}).get("Status") != "healthy"):
                observation["waiting_for"] = "ollama"
                continue
        action = "restart" if container["State"]["Status"] == "running" else "start"
        observation["action"] = action
        if dry_run:
            continue
        record["attempts"] = min(record.get("attempts", 0) + 1, 10)
        record["retry_after"] = now + min(30 * 2 ** (record["attempts"] - 1), 300)
        record["last_reason"] = reason
        try:
            if gpu_ready is None:
                try:
                    prepare_gpu()
                    gpu_ready = True
                except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                    gpu_ready = False
                    LOG.error("GPU initialization failed: %s", error)
            if not gpu_ready:
                observation["error"] = "GPU unavailable; retry scheduled"
                continue
            LOG.warning("%s %s: %s", action, service, reason)
            args = [docker, action]
            if action == "restart":
                args += ["--time", "10"]
            command([*args, key], timeout=30)
            record["last_action"] = action
            record["last_attempt"] = now
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            observation["error"] = str(error)
            LOG.error("Recovery of %s failed: %s", service, error)
    active = {c["Id"] for c in items}
    for key in set(state) - active:
        del state[key]
    return observations


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--docker", default=shutil.which("docker") or "/snap/bin/docker")
    parser.add_argument("--state-dir", type=Path, default=Path(__file__).resolve().parents[1] / ".gpu-recovery")
    parser.add_argument("--services", nargs="+", choices=SERVICES, default=list(SERVICES))
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--interval", type=float, default=15)
    args = parser.parse_args()
    if args.interval < 5:
        parser.error("interval must be at least 5 seconds")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.state_dir / "watchdog.lock").open("a")
    if not args.dry_run:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
    LOG.setLevel(logging.INFO)
    handler = RotatingFileHandler(args.state_dir / "watchdog.log", maxBytes=1024 * 1024, backupCount=3)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(handler)
    state_path = args.state_dir / "retries.json"
    try:
        state = json.loads(state_path.read_text())
    except (OSError, ValueError):
        state = {}
    if not args.dry_run:
        LOG.info("Watchdog started for Compose project %s", args.project)
    while True:
        started = time.monotonic()
        try:
            paused = (args.state_dir / "paused").exists()
            observations = [] if paused else recover(
                containers(args.docker, args.project, args.services), state,
                args.docker, now=time.time(), dry_run=args.dry_run,
            )
            snapshot = {"checked_at": datetime.now(timezone.utc).isoformat(), "paused": paused, "services": observations}
            if not args.dry_run:
                write_json(state_path, state)
                write_json(args.state_dir / "status.json", snapshot)
            if not args.watch:
                print(json.dumps(snapshot, indent=2))
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired):
            LOG.exception("Recovery check failed; will retry")
            if not args.watch:
                raise
        if not args.watch:
            break
        time.sleep(max(0, args.interval - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
