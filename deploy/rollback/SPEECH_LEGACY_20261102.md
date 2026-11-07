# Speech rollback snapshot — 2026-11-02

This manifest records the production speech images and model mounts before the
Mana Piper/OpenVoice V2 migration. The immutable rollback tags were created
before any speech container was rebuilt.

## Images

- Coqui TTS: `metarang-npc-tts:rollback-coqui-20261102`
  - image ID: `sha256:0523a3dc68016cd952764abbdcdc928ee03b449d55540047ccff834682f704fe`
- FreeVC: `metarang-npc-vc:rollback-freevc-20261102`
  - image ID: `sha256:55497bb0a3349bb3fdf6678906fbb002e0640fd87012d8d531d6cef3f02ddce2`

## Legacy model files (preserved in place)

- `TTS/best_model_91323.pth` (997,693,178 bytes)
- `TTS/config.json` (10,128 bytes)
- `VC/freevc.pth` (472,644,351 bytes)
- `VC/wavlm/WavLM-Large.pt` (1,261,965,425 bytes)
- `VC/speaker_encoder/ckpt/pretrained_bak_5805000.pt` (17,090,379 bytes)

## Rollback commands

Use the checked-in override, without `--build`, to recreate only the two speech
containers from their immutable local rollback tags:

```bash
docker compose \
  -f docker-compose.yml \
  -f deploy/rollback/docker-compose.speech-legacy-20261102.yml \
  --profile voice-conversion \
  up -d --no-deps --force-recreate tts vc
```

The legacy images contain their original application code and checkpoints. The
current Mana/OpenVoice model mounts are harmless when attached to those images.
Do not run both legacy and migrated engines at once. Return to the migrated
images with the normal Compose file and the same speech-only force-recreate.

The working tree already contained uncommitted production changes when this
snapshot was made. Do not use `git reset --hard` as a rollback mechanism.
