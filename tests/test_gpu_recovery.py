import copy
from datetime import datetime, timezone
import unittest
from unittest.mock import patch

from scripts import gpu_recovery as recovery
from scripts.install_gpu_recovery import cron_block, update_crontab


NOW = 1800000000


def container(service="tts", *, status="running", health="healthy", age=1000,
              code=0, error="", failing=0, start_period=180):
    return {
        "Id": service + "-container",
        "Config": {"Labels": {"com.docker.compose.service": service},
                   "Healthcheck": {"StartPeriod": start_period * 10**9}},
        "State": {"Status": status, "ExitCode": code, "Error": error,
                  "StartedAt": datetime.fromtimestamp(NOW - age, timezone.utc).isoformat(),
                  "Health": {"Status": health, "FailingStreak": failing}},
    }


class RecoveryTests(unittest.TestCase):
    @patch.object(recovery, "command")
    @patch.object(recovery.shutil, "which", return_value="/usr/bin/nvidia-modprobe")
    @patch.object(recovery.Path, "is_char_device", side_effect=[False, True, True, True])
    def test_gpu_helper_restores_nodes_before_the_driver_probe(self, nodes, helper, command):
        recovery.prepare_gpu()
        self.assertEqual([call.args[0] for call in command.call_args_list], [
            ["/usr/bin/nvidia-modprobe", "-u", "-c=0"],
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        ])

    @patch.object(recovery, "command")
    @patch.object(recovery.shutil, "which", return_value="/usr/bin/nvidia-modprobe")
    @patch.object(recovery.Path, "is_char_device", return_value=False)
    def test_persistently_missing_nodes_stop_recovery(self, nodes, helper, command):
        with self.assertRaisesRegex(RuntimeError, "still missing"):
            recovery.prepare_gpu()
        command.assert_called_once_with(["/usr/bin/nvidia-modprobe", "-u", "-c=0"], timeout=15)

    @patch.object(recovery, "command")
    @patch.object(recovery, "prepare_gpu")
    def test_missing_device_failure_is_repaired_before_start(self, gpu, command):
        events = []
        gpu.side_effect = lambda: events.append("gpu")
        command.side_effect = lambda args, **kw: events.append(args)
        failed = container(status="exited", code=255, error="/dev/nvidia-uvm: no such file")
        state = {}
        recovery.recover([failed], state, "docker", now=NOW)
        self.assertEqual(events, ["gpu", ["docker", "start", failed["Id"]]])
        self.assertEqual(state[failed["Id"]]["retry_after"], NOW + 30)

    @patch.object(recovery, "command")
    @patch.object(recovery, "prepare_gpu")
    def test_healthy_and_cleanly_stopped_services_are_untouched(self, gpu, command):
        for item in (container(), container(status="exited", code=0),
                     container(status="exited", code=143), container(status="created"),
                     container(status="paused")):
            recovery.recover([item], {}, "docker", now=NOW)
        gpu.assert_not_called()
        command.assert_not_called()

    @patch.object(recovery, "command")
    @patch.object(recovery, "prepare_gpu", side_effect=RuntimeError("GPU broken"))
    def test_unavailable_gpu_never_restarts_services_and_backs_off(self, gpu, command):
        items = [container(service=s, status="exited", code=255) for s in ("tts", "stt")]
        state = {}
        recovery.recover(items, state, "docker", now=NOW)
        recovery.recover(items, state, "docker", now=NOW + 15)
        gpu.assert_called_once()
        command.assert_not_called()
        recovery.recover(items, state, "docker", now=NOW + 31)
        self.assertEqual(gpu.call_count, 2)
        self.assertEqual(state[items[0]["Id"]]["retry_after"], NOW + 91)

    @patch.object(recovery, "command")
    @patch.object(recovery, "prepare_gpu")
    def test_sustained_health_failures_restart_only_the_failed_service(self, gpu, command):
        items = [container(service="ollama"), container(service="llm", failing=3)]
        recovery.recover(items, {}, "docker", now=NOW)
        command.assert_called_once_with(["docker", "restart", "--time", "10", "llm-container"], timeout=30)

    def test_startup_grace_and_stuck_startup_are_distinct(self):
        self.assertIsNone(recovery.recovery_reason(container(health="unhealthy", age=60), NOW))
        self.assertIsNone(recovery.recovery_reason(container(health="starting", age=400), NOW))
        self.assertEqual(recovery.recovery_reason(container(health="starting", age=601), NOW), "model startup timed out")

    @patch.object(recovery, "command")
    @patch.object(recovery, "prepare_gpu")
    def test_rag_waits_for_ollama_while_tts_recovers_independently(self, gpu, command):
        items = [container(service=s, status="exited", code=255) for s in ("llm", "tts", "ollama")]
        observations = recovery.recover(items, {}, "docker", now=NOW)
        self.assertEqual([c.args[0] for c in command.call_args_list],
                         [["docker", "start", "ollama-container"], ["docker", "start", "tts-container"]])
        self.assertEqual(next(o for o in observations if o["service"] == "llm")["waiting_for"], "ollama")

    @patch.object(recovery, "command")
    @patch.object(recovery, "prepare_gpu")
    def test_docker_restart_loop_repairs_gpu_without_competing_restarts(self, gpu, command):
        recovery.recover([container(status="restarting")], {}, "docker", now=NOW)
        gpu.assert_called_once()
        command.assert_not_called()

    @patch.object(recovery, "command")
    @patch.object(recovery, "prepare_gpu")
    def test_dry_run_does_not_touch_gpu_or_containers(self, gpu, command):
        observations = recovery.recover([container(status="exited", code=255)], {}, "docker", now=NOW, dry_run=True)
        self.assertEqual(observations[0]["action"], "start")
        command.assert_not_called()
        gpu.assert_not_called()

    @patch.object(recovery, "command", side_effect=RuntimeError("start failed"))
    @patch.object(recovery, "prepare_gpu")
    def test_failed_start_cooldown_survives_a_process_restart(self, gpu, command):
        state = {}
        items = [container(status="exited", code=255)]
        recovery.recover(items, state, "docker", now=NOW)
        restored = copy.deepcopy(state)
        recovery.recover(items, restored, "docker", now=NOW + 10)
        self.assertEqual(command.call_count, 1)

    @patch.object(recovery, "command")
    def test_discovery_filters_project_and_skips_oneoff_containers(self, command):
        ordinary = container()
        oneoff = container(service="llm")
        oneoff["Config"]["Labels"]["com.docker.compose.oneoff"] = "True"
        unrelated = container(service="web")
        import json
        command.side_effect = ["a b c", json.dumps([ordinary, oneoff, unrelated])]
        self.assertEqual(recovery.containers("docker", "metarang-npc", recovery.SERVICES), [ordinary])
        self.assertIn("label=com.docker.compose.project=metarang-npc", command.call_args_list[0].args[0])


class InstallationTests(unittest.TestCase):
    def test_installation_is_idempotent_and_preserves_other_jobs(self):
        existing = "# existing backup\n0 2 * * * /opt/backup\n"
        block = cron_block("metarang-npc", ["/usr/bin/python3", "/project path/watchdog.py", "--watch"])
        installed = update_crontab(existing, "metarang-npc", block)
        self.assertIn(existing, installed)
        self.assertEqual(update_crontab(installed, "metarang-npc", block), installed)
        self.assertEqual(installed.count("@reboot"), 1)
        self.assertIn("'/project path/watchdog.py'", installed)


if __name__ == "__main__":
    unittest.main()
