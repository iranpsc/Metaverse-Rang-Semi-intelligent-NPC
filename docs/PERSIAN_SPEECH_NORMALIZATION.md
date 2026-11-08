# Persian speech preparation

The TTS service prepares LLM sentences immediately before Mana/Piper synthesis.
It runs offline on CPU, with no new model or Python package. Chat transcripts
continue to contain the original LLM output. Both the realtime worker and the
HTTP speech clients use this same TTS service.

## Behavior

- Persian, Arabic and ASCII digits; decimal separators `.` and `٫`; thousands
  separators `,` and `٬`; positive and negative signs.
- Percentages (`%`, `٪`), currencies, common measurement/storage units,
  squared/cubed meters and fractions.
- Year-first `YYYY/MM/DD` and `YYYY-MM-DD` dates, hours/minutes/seconds.
  Dates retain an explicit year/month/day reading without calendar conversion.
  Month/day bounds for years 1200–1600 follow the Persian calendar; other years
  use Gregorian bounds. Obvious invalid dates/times remain unchanged. Persian
  Esfand leap-day validation is intentionally not inferred.
- Mobile numbers, explicitly labeled telephone/card/account/order numbers,
  identifiers, verification codes and numbers with leading zeroes are spoken
  digit by digit. Unknown identifiers should be prefixed with `کد` or `شناسه`.
- Versions and IPv4-style dotted strings are spoken with `نقطه` separators;
  Latin technical tokens such as `GPT-4` and `v2.5` remain intact unless a
  pronunciation override exists.
- Markdown and emoji cleanup, Arabic/Persian letter unification, preservation
  of existing vowel marks and half-spaces, and removal of bidi controls.
- Selective diacritics and ezafe from a phrase lexicon. Longest phrases win;
  explicit vowel marks are preserved. Isolated ambiguous words such as `مرد`
  or `کرم` are not guessed. This is not full contextual diacritization.

`SentenceBuffer` waits for lookahead after a period, so a streamed `3.` followed
by `14` stays one decimal. Chunk length is a soft limit: words, URLs and grouped
phone/card digits are never sliced merely to fit 280 characters.

Input longer than `TTS_MAX_TEXT_LENGTH` (default 2,000 characters), or prepared
text longer than `TTS_MAX_NORMALIZED_TEXT_LENGTH` (default 12,000), receives
HTTP 400. Expanding a number can increase length considerably; speech text is
never silently truncated.

## Preview without loading a model

```bash
python3 -m pipeline_core.persian_text 'قیمت ۱۲۵٬۰۰۰ تومان با ۲۰٪ تخفیف؛ ساعت ۱۴:۳۰' --json
python3 -m pipeline_core.persian_text 'کرم ضد آفتاب' --diacritization off
python3 -m unittest tests.test_persian_text tests.test_sentence_buffer -v
```

The JSON preview prints both original and prepared text only when explicitly
requested. The TTS service does not add transcript logging.

## Pronunciation overrides

Built-in entries live in `pipeline_core/persian_pronunciations.json`.
Add project-specific terms to `TTS/pronunciation/lexicon.json`, for example:

```json
{
  "GPT-4": "جی پی تی چهار",
  "API": "اِی پی آی",
  "کتاب من": "کِتابِ مَن",
  "CPU": ""
}
```

An empty value disables a bundled entry. Overrides are applied once, by whole
word/phrase, before numeric processing. They never rewrite URL/email contents.
Use the normalized Persian spelling for keys (`ی`/`ک`, Persian digits already
converted to ASCII). Explicitly vocalized input takes precedence over a key
with no vowel marks.

Compose mounts this directory read-only. After editing, restart `tts`; no image
rebuild is needed for lexicon changes. `TTS_PRONUNCIATION_DIR_HOST` selects a
different host directory. For running Python directly,
`TTS_PRONUNCIATION_LEXICON` selects a custom JSON file (or use CLI `--lexicon`).
`TTS_DIACRITIZATION=off` disables lexicon substitutions, keeping explicit input
diacritics and number normalization. The default is `lexicon`.

## Generate before/after audio with the existing model

```bash
mkdir -p artifacts/persian-normalization
docker compose run --rm --no-deps -T \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace:ro" \
  -v "$PWD/artifacts/persian-normalization:/output" \
  -w /workspace -e PYTHONPATH=/workspace --entrypoint python \
  tts scripts/preview_persian_speech.py \
  --model /models/mana/fa_IR-mana-medium.onnx --output-dir /output
```

This uses the installed Piper runtime in a disposable container. It exports raw
and prepared WAV files plus a JSON report with prepared text, phonemes, audio
duration and timing. `--text '...'` tests a custom sentence. Omitting `--model`
exports text previews only and needs no Piper installation.

The WAV checks verify valid, non-silent audio. Listen to the before/after files
to assess pronunciation and prosody; automated checks do not establish human
intelligibility or perfect diacritization.

To build and activate source changes:

```bash
docker compose build tts agent-worker
docker compose up -d --no-deps tts agent-worker
```

On this host the initial full build could not reach PyPI. The activated images
were therefore built offline from the installed runtime, copying only the
updated Python source and bundled lexicon. The exact overlay Dockerfiles are
in `artifacts/persian-normalization/runtime-{tts,agent}.Dockerfile`. Previous
images are retained as `metarang-npc-tts:before-normalization` and
`metarang-npc-agent-worker:before-normalization`. Standard source builds above
remain available when the package index is reachable; no dependency changes
are required for this feature.
