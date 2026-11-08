# Saved voice sample playback

The live sample list returns `/media/voice_samples/<filename>.wav`, but the
recording URL returns HTTP 404. Django saves recordings in the Compose
`media_data` volume (`/app/media`), while the existing host Nginx `/media/`
alias points to the repository's older `media/` directory. The saved recording
exists in the volume. Changing the play button cannot fix this missing route.

## Existing host Nginx deployment

Start the media service from the same Compose project as the running web
service, so it uses the existing volume:

```sh
docker compose --profile host-nginx up -d --no-deps media
```

It serves only voice recordings, mounts the volume read-only, and listens on
host loopback port 8082. Set `MEDIA_HTTP_PORT` if that port is already in use.
Do not point host Nginx directly into Docker's private storage directory or
change the permissions of Docker's storage directories.

Back up `/etc/nginx/sites-available/MetaRangNPC`, then replace its existing
`location /media/` block with:

```nginx
location /media/ {
    proxy_pass http://127.0.0.1:8082;
    proxy_set_header Host $host;
}
```

Keep `proxy_pass` without a trailing slash so the full `/media/voice_samples/`
path reaches the media service. Use the chosen `MEDIA_HTTP_PORT` here if changed.
Validate and reload host Nginx:

```sh
sudo nginx -t && sudo systemctl reload nginx
```

No recording migration or application restart is needed. New recordings use
the same mounted volume automatically.

## Compose Nginx deployment

The production Nginx service now mounts `media_data` read-only and serves
`/media/voice_samples/` directly. Recreate that service to apply both changes:

```sh
docker compose -f docker-compose.yml -f docker-compose.prod.yml --profile production up -d --no-deps nginx
```

## Verification

Obtain a recording URL from `/api/voice-samples/`, then request its first bytes:

```sh
curl -i -H 'Range: bytes=0-63' https://npc.metarang.com/media/voice_samples/RECORDING.wav
```

Expect HTTP 206, `Content-Type: audio/wav`, `Content-Length: 64`, and a
`Content-Range` header. A request without `Range` must return HTTP 200 and the
full recording. Missing recordings must still return HTTP 404. Refresh the
voice samples page and play the saved recording.
