# OpenVoice source provenance

This directory contains the minimal unmodified OpenVoice modules required by
MetaRang's `ToneColorConverter` integration.

- Upstream: https://github.com/myshell-ai/OpenVoice
- Commit: `74a1d147b17a8c3092dd5430504bd83ef6c7eb23`
- License: MIT (see `LICENSE`)
- Vendored modules: model, attention, transform, mel-processing, common, and
  configuration utilities only

The upstream demo, MeloTTS integration, language frontends, Whisper-based
speaker extractor, Gradio application, and watermark package are deliberately
excluded. MetaRang uses the converter checkpoint's built-in `ref_enc` through
its own small runtime adapter.
