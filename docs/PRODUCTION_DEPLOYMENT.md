# Production deployment and WebRTC networking

This topology targets one Linux Docker Compose host. LiveKit uses host networking; application and observability services remain on the Compose bridge network.

## DNS and certificates

Create these records without hardcoding them in the repository:

| Variable | Purpose |
|---|---|
| `API_DOMAIN` | Django HTTPS and Swagger, for example `api.example.com` |
| `LIVEKIT_DOMAIN` | LiveKit secure signaling, for example `livekit.example.com` |
| `LIVEKIT_TURN_DOMAIN` | Embedded TURN certificate name, for example `turn.example.com` |

`API_DOMAIN` and `LIVEKIT_DOMAIN` resolve to the Nginx public address. `LIVEKIT_TURN_DOMAIN` resolves to the public address receiving TURN traffic. Certificates must have a complete chain and match their respective names.

Expected certificate layout below `TLS_CERTS_DIR`:

```text
api/fullchain.pem
api/privkey.pem
livekit/fullchain.pem
livekit/privkey.pem
turn/fullchain.pem
turn/privkey.pem
```

Set `LIVEKIT_PUBLIC_URL=wss://<LIVEKIT_DOMAIN>` and `LIVEKIT_NODE_IP` to the public media IP. Do not use a private/NAT address for the advertised node IP.

## Required inbound firewall/NAT rules

| Protocol/port | Destination | Purpose |
|---|---|---|
| TCP 80 | Nginx | ACME/HTTP redirect if used |
| TCP 443 | Nginx | Django HTTPS and LiveKit WSS signaling |
| TCP 7881 | LiveKit host | WebRTC ICE/TCP fallback |
| UDP 50000–60000 | LiveKit host | WebRTC media range |
| UDP 3478 | LiveKit host | TURN/UDP |
| TCP 5349 | LiveKit host | TURN/TLS in the included single-IP configuration |

Forward the same ports through any upstream NAT. Restrict Grafana and Prometheus to operators; Compose binds their local ports to `127.0.0.1`.

Many corporate networks allow TLS only on TCP 443. For that strongest fallback, place TURN/TLS on external TCP 443 using either a second public IP/host for `LIVEKIT_TURN_DOMAIN`, or an L4/SNI-aware load balancer that forwards the TURN name to LiveKit’s internal TLS port. Nginx already owns TCP 443 for HTTP signaling on the default single-IP host, so simply changing `tls_port` to 443 on that host will cause a bind conflict. Validate UDP-blocked and TCP-443-only clients before launch.

## Configuration checklist

1. Copy `.env.example` to `.env` and replace every example domain, IP, password, and secret.
2. Set `DJANGO_DEBUG=false`, a random `DJANGO_SECRET_KEY`, exact `DJANGO_ALLOWED_HOSTS`, `DJANGO_CSRF_TRUSTED_ORIGINS`, and `CORS_ALLOWED_ORIGINS`.
3. Set a distinct LiveKit API key and a secret of at least 32 random characters.
4. Provision model files and the Ollama model before marking the deployment ready.
5. Install NVIDIA Container Toolkit and verify `docker run --gpus all ... nvidia-smi` before enabling FreeVC.
6. Put certificates at `TLS_CERTS_DIR` with read-only permissions accessible to the containers.
7. Open/forward the documented ports and confirm the host has no conflicting Redis, RTC, or TURN listeners.
8. Run `make config`, then `make production-up`.
9. Verify `https://<API_DOMAIN>/api/v1/health/ready/`, Swagger, Grafana, LiveKit metrics, and the Python smoke test from outside the server network.

The production LiveKit configuration reads secrets and TURN/node settings from environment variables. Redis database 1 is dedicated to LiveKit; Django/Celery uses database 0 on the same instance. Redis is bound to host loopback because host-networked LiveKit cannot resolve Compose service DNS.

## TLS termination and internal URLs

Nginx terminates HTTPS/WSS for Django and LiveKit signaling. Media does not pass through Nginx. The production override changes Django and the agent worker to `host.docker.internal:7880` for private signaling/API calls because a host-networked LiveKit container is not attached to Compose DNS.

Do not expose STT, LLM/RAG, TTS, VC, Redis, Ollama, Tempo, or Loki directly to the internet.

## Shutdown and upgrades

Use `make down` for normal termination. The worker CLI drains/cancels jobs on SIGTERM, and each job cancels its audio/turn tasks and closes HTTP/LiveKit resources. Rooms have empty/departure timeouts as a second cleanup layer.

Back up the Django runtime volume, pipeline data/vector stores, Ollama data, Grafana data, and any voice samples before upgrades. Image tags are pinned; test version updates with `make config`, unit tests, and the smoke client before deployment.
