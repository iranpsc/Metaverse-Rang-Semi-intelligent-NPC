#!/usr/bin/env python3
"""Check the public/local endpoints an operator needs first."""

from __future__ import annotations

import json
import socket
import urllib.request


def http_check(name: str, url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            body = response.read(300).decode("utf-8", "replace")
        print(f"OK   {name:12} {url} {body}")
        return True
    except Exception as exc:
        print(f"FAIL {name:12} {url} {type(exc).__name__}: {exc}")
        return False


def tcp_check(name: str, host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=3):
            pass
        print(f"OK   {name:12} tcp://{host}:{port}")
        return True
    except OSError as exc:
        print(f"FAIL {name:12} tcp://{host}:{port} {exc}")
        return False


def main() -> None:
    checks = [
        http_check("Django live", "http://localhost:8000/api/v1/health/live/"),
        http_check("Django ready", "http://localhost:8000/api/v1/health/ready/"),
        tcp_check("LiveKit", "localhost", 7880),
        http_check("Prometheus", "http://localhost:9091/-/ready"),
        http_check("Grafana", "http://localhost:3000/api/health"),
    ]
    print(json.dumps({"healthy": all(checks), "passed": sum(checks), "total": len(checks)}))
    raise SystemExit(0 if all(checks) else 1)


if __name__ == "__main__":
    main()
