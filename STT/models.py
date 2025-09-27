from django.db import models
import random

def random_name():
    return f"Recording_{random.randint(1, 1000000)}"

class AudioRecording(models.Model):
    audio_file = models.FileField(upload_to=random_name, blank=True, null=True)
    transcript = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.audio_file} - {self.created_at}"


class FineTuningJob(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    # user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    language = models.CharField(max_length=10)
    dataset_path = models.CharField(max_length=255)
    base_model = models.CharField(max_length=50, default='base')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    model_path = models.CharField(max_length=255, blank=True, null=True)
    
    def __str__(self):
        return f"{self.language} ({self.status})"