# Automatic GPU service recovery

The host watchdog monitors existing `ollama`, `llm`, `tts`, `stt`, and `vc`
containers in one explicitly selected Compose project. It does not create
containers, rebuild images, or load another copy of a model.

Docker's restart policies remain enabled. A separate host check is needed for
failures before a container can start, such as a missing `/dev/nvidia-uvm`.
Docker documents the startup limitation of restart policies in its
[restart policy details](https://docs.docker.com/engine/containers/start-containers-automatically/#restart-policy-details).

## Installation

Run as the host user who already operates Docker, from the repository root:

```sh
python scripts/gpu_recovery.py --project metarang-npc --dry-run
python scripts/install_gpu_recovery.py --project metarang-npc
```

Use the actual Compose project name if different. The installer preserves other
cron jobs and is safe to run again. It records absolute Python, Docker, script,
and state paths so cron does not depend on an interactive shell environment.
Keep the repository and Python installation at these paths, or reinstall the
cron entries after moving them.

The cron daemon must be active. On this host `nvidia-modprobe` is installed
setuid by the NVIDIA package, so the Docker operator can restore device nodes
without an interactive sudo prompt. On other hosts configure the NVIDIA
device initialization mechanism before enabling recovery. Do not make this
Python script setuid or grant it passwordless root shell access.

The watchdog starts immediately and checks every 15 seconds. An `@reboot` job
starts it after a reboot, and a once-per-minute cron job brings the watchdog
itself back if it exits. An exclusive file lock prevents duplicate supervisors.
Docker being temporarily unavailable at boot is retried on the next check.

## Recovery behavior

- Before starting failed GPU containers, restore missing NVIDIA device nodes
  with `nvidia-modprobe -u -c=0`, then check `nvidia-smi`.
- Start abnormal exits, including the original missing-device error. Leave
  normal exits (0 or 143), paused containers, and unstarted containers alone.
- When Docker is already restarting a container, repair GPU initialization
  and let Docker complete its own restart.
- Restart a running service after three consecutive failing Docker readiness
  checks, or when Docker marks it unhealthy. Respect each model's startup
  period, with a minimum 120-second grace. A service stuck in `starting` is
  retried after at least ten minutes.
- Wait for Ollama readiness before recovering RAG. Other speech services can
  recover independently. Healthy containers keep running.
- Space repeated attempts by 30, 60, 120, 240, and then 300 seconds. Persist
  this state across supervisor restarts and reset it after five healthy minutes.

If the GPU driver itself is broken or the hardware is unavailable, the
watchdog records the error and retries. It does not reset a GPU shared by
healthy services or reboot the host. Service health probes must detect their
own application failures; a probe that always reports success cannot detect
an inference failure.

## Status and maintenance

```sh
cat .gpu-recovery/status.json
tail -n 30 .gpu-recovery/watchdog.log
crontab -l
```

Logs rotate at 1 MiB with three backups. Retry state and logs stay in the
ignored `.gpu-recovery/` directory. `checked_at` is UTC; a stale value indicates
that the supervisor is no longer checking Docker.

Pause before deliberate failures, GPU driver work, or forced container stops:

```sh
touch .gpu-recovery/paused
```

Wait until `status.json` shows `"paused": true` before doing maintenance. Resume:

```sh
rm .gpu-recovery/paused
```

To disable automatic recovery, pause it and remove only the marked
`metarang-gpu-recovery` block using `crontab -e`.

## Validation and availability limits

```sh
python -m unittest tests.test_gpu_recovery -v
```

The initial live deployment recovered the existing STT container that had
failed with a missing UVM device, while healthy chat and speech containers
continued running. Stopping only the watchdog process confirmed that cron
restarted it automatically without changing any service's start time.
Test cases cover unavailable GPUs, restart backoff,
dependency order, readiness failures, startup grace, clean stops, project
selection, and repeat installation.

This provides automatic recovery, not zero downtime: detection, restart,
and model loading take time. A single host with a single GPU cannot continue
GPU inference while that GPU is unavailable. Continuous availability requires
another ready inference host/GPU and request routing that fails over to it.
