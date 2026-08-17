import os
import uuid

from django.db import models


def voice_sample_path(instance, filename):
    """Store recordings under media/voice_samples/ with a unique, safe name.

    The FreeVC microservice reads these files back by basename, so the name
    must be collision-free and free of path separators.
    """
    ext = os.path.splitext(filename)[1].lower() or ".webm"
    return f"voice_samples/{uuid.uuid4().hex}{ext}"


class VoiceSample(models.Model):
    """A short recording of a voice to imitate during voice conversion.

    Samples are a shared, global library: any chat session can select any
    sample as the target voice.
    """

    name = models.CharField(max_length=64, help_text="Label / initials for the voice")
    audio_file = models.FileField(upload_to=voice_sample_path)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.name} ({self.audio_file.name})"

    @property
    def filename(self):
        """Basename the VC microservice uses to locate the file on disk."""
        return os.path.basename(self.audio_file.name) if self.audio_file else ""
