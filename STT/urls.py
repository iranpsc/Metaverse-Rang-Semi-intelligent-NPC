from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('record/', views.record_audio, name='record'),
    path('delete-recording/<int:recording_id>/', views.delete_recording, name='delete_recording'),
    path('train-whisper/', views.WhisperTrainingView.as_view(), name='fine-tune-whisper'),
    path('training-status/', views.training_status, name='training_status'),
    path('create-sample-dataset/', views.create_sample_dataset, name='create-sample-dataset'),
]