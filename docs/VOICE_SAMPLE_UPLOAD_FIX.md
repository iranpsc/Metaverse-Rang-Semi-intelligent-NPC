# Voice sample upload failures

The public `/api/voice-samples/` endpoint rejected a 1,100,000-byte multipart
upload with an HTML `413 Request Entity Too Large` response from Nginx. A POST
without a file reached Django and returned JSON validation errors. The upload
probe omitted the required name so it could not create a sample.

Django accepts recordings up to 25 MiB, but Nginx defaults to a 1 MiB request
body limit. The proxy must allow the recording plus multipart form overhead.

## Existing host Nginx deployment

Back up and edit the active configuration:

```sh
sudo cp -a /etc/nginx/sites-available/MetaRangNPC /etc/nginx/sites-available/MetaRangNPC.bak
sudoedit /etc/nginx/sites-available/MetaRangNPC
```

Add this directive inside the HTTPS `server` block for `npc.metarang.com`:

```nginx
client_max_body_size 26m;
```

Validate before reloading:

```sh
sudo nginx -t && sudo systemctl reload nginx
```

The repository's `deploy/nginx/default.conf.template` includes the same limit
for Docker deployments. An outer host Nginx proxy needs its own limit too.
If the application's upload limit is increased, adjust the proxy limit as well.

Deploy the updated Django view and template through the normal web deployment
process to show understandable upload errors and preserve recordings on failure.

## Verification

Run the UI regression checks from the repository root:

```sh
node tests/test_voice_samples_ui.cjs
```

To verify the proxy without creating a sample, upload a file without a name:

```sh
truncate -s 1100000 /tmp/voice-upload-check.webm
curl -i -F audio=@/tmp/voice-upload-check.webm https://npc.metarang.com/api/voice-samples/
```

Expect HTTP 400 and `{"error": "name is required"}`, rather than HTTP 413.
Then record and save a real sample in the browser and verify its preview.
