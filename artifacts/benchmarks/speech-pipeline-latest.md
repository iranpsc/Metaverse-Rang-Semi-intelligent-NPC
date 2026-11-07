# Speech pipeline benchmark

Generated: `2026-11-08T01:46:08.612601+00:00`

This is an automated functional/performance benchmark. Voice similarity and
Persian naturalness still require human listening approval with authorized samples.

| Measurement | Result |
|---|---:|
| Mana model load | 1.5415650559589267 s |
| Mana cold synthesis | 0.8596896769013256 s |
| OpenVoice model load | 0.7347180260112509 s |
| OpenVoice recommended free VRAM | 765 MiB |
| Peak GPU utilization | 63.0% |
| Peak total GPU memory used | 12101.0 MiB |
| Successful requests | 3 |
| Failed requests | 0 |
| Failure rate | 0.000 |
| Aggregate end-to-end RTF | 0.11454592753091623 |
| First-request latency | 0.6442502608988434 s |
| Warm average latency | 0.6006501035299152 s |
| Warm end-to-end RTF | 0.10828939803074601 |

## Requests

| # | Mode | End-to-end | TTS RTF | VC RTF | VC cache hit |
|---:|---|---:|---:|---:|---:|
| 1 | first_request | 0.644 s | 0.059869 | 0.025604 | False |
| 2 | warm | 0.624 s | 0.053022 | 0.022902 | True |
| 3 | warm | 0.578 s | 0.056229 | 0.023289 | True |

Reference: `auto-generated Mana reference (functional test only)`

Human quality sign-off: **not performed by this automated benchmark**.
