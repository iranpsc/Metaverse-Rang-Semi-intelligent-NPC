# app/views.py
from celery.result import AsyncResult
from django.urls import reverse
from django.contrib import messages
from .tasks import run_training_sequence
from .forms import WhisperTrainingForm
import os
import traceback
from pathlib import Path
from django.conf import settings
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_http_methods
import whisper
from .models import AudioRecording, FineTuningJob
from pydub import AudioSegment
from django.views import View
import pandas as pd
import zipfile
import io
from datetime import datetime

# Import LLM service
try:
    from LLM.rag_service import RAGService
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


def home(request):
    recordings = AudioRecording.objects.all().order_by('-created_at')

    # Get completed fine-tuning jobs (trained models)
    trained_models = FineTuningJob.objects.filter(
        status='completed').order_by('-created_at')

    # Get available model files from the model directory
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    available_models = []

    if os.path.exists(model_dir):
        for file in os.listdir(model_dir):
            if file.endswith('.pt'):
                model_path = os.path.join(model_dir, file)
                model_size = os.path.getsize(model_path)
                model_size_mb = round(model_size / (1024 * 1024), 2)
                available_models.append({
                    'name': file,
                    'path': model_path,
                    'size_mb': model_size_mb
                })

    # Standard Whisper models that are available
    standard_models = [
        {'name': 'tiny', 'parameters': '39 M', 'description': 'Fastest, least accurate'},
        {'name': 'base', 'parameters': '74 M',
            'description': 'Good balance of speed and accuracy'},
        {'name': 'small', 'parameters': '244 M',
            'description': 'Better accuracy, slower'},
        {'name': 'medium', 'parameters': '769 M', 'description': 'High accuracy, slower'},
        {'name': 'large', 'parameters': '1550 M',
            'description': 'Best accuracy, slowest'},
        {'name': 'turbo', 'parameters': '809 M', 'description': 'Hybrid'}
    ]

    context = {
        'recordings': recordings,
        'trained_models': trained_models,
        'available_models': available_models,
        'standard_models': standard_models,
    }

    return render(request, 'static/home.html', context)


# Simple in-memory cache of loaded Whisper models by key
WHISPER_MODELS = {}


def _resolve_model_key(model_choice):
    """Normalize model choice to a cache key and loading arguments.

    Accepts either standard Whisper sizes (e.g., "tiny", "base", ...),
    a HuggingFace repo id (e.g., "openai/whisper-small"), or a local .pt path.
    Returns a tuple (cache_key, load_arg, device).
    """
    # Auto-detect device: use CUDA if available, otherwise CPU
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"INFO: Using GPU (CUDA) for Whisper model")
    else:
        print(f"INFO: Using CPU for Whisper model (CUDA not available)")
    if not model_choice:
        # Default to bundled Tiny model if present; otherwise fallback to standard tiny
        bundled_path = os.path.join(
            os.path.dirname(__file__), "model", "Turbo.pt")
        if os.path.exists(bundled_path):
            return (bundled_path, bundled_path, device)
        return ("turbo", "turbo", device)

    # If it's an existing local file, load from path
    if os.path.exists(model_choice):
        return (model_choice, model_choice, device)

    # Otherwise treat it as a model name/repo id
    return (model_choice, model_choice, device)


def get_whisper_model(model_choice):
    """Load or retrieve from cache the requested Whisper model."""
    cache_key, load_arg, device = _resolve_model_key(model_choice)
    if cache_key in WHISPER_MODELS:
        return WHISPER_MODELS[cache_key]
    model = whisper.load_model(load_arg, device=device)
    WHISPER_MODELS[cache_key] = model
    return model


def get_user_id_for_llm(request):
    """
    Get user_id for LLM service from authenticated user, session, or request data.
    """
    # Try to get from authenticated user first
    if request.user.is_authenticated:
        return str(request.user.id)
    
    # Try to get from session
    if 'user_id' in request.session:
        return str(request.session['user_id'])
    
    # Generate a session-based user_id if not exists
    if 'user_id' not in request.session:
        import uuid
        request.session['user_id'] = str(uuid.uuid4())
    
    return str(request.session['user_id'])


def get_user_memory_path_for_llm(user_id):
    """
    Get user-specific memory path for LLM.
    """
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    MEMORY_BASE_PATH = os.path.join(BASE_DIR, "data/memories")
    user_memory_dir = os.path.join(MEMORY_BASE_PATH, f"user_{user_id}")
    os.makedirs(user_memory_dir, exist_ok=True)
    return user_memory_dir


def record_audio(request):
    if request.method == 'POST':
        # Handle POST request without audio file
        if not request.FILES.get('audio_file'):
            return JsonResponse({
                'status': 'error',
                'message': 'No audio file provided'
            }, status=400)
        
        fs = FileSystemStorage()
        filename = None
        temp_path = None

        try:
            # 1. Save the original file first
            audio_file = request.FILES['audio_file']
            filename = fs.save(audio_file.name, audio_file)
            audio_path = os.path.join(settings.MEDIA_ROOT, filename)

            # 2. Process the audio file
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_frame_rate(16000).set_channels(1)

            # Create optimized temporary file
            temp_path = os.path.join(settings.MEDIA_ROOT, 'temp_optimized.wav')
            audio.export(temp_path, format="wav", bitrate="16k")

            # 3. Transcribe with selected model (fallback handled inside helper)
            selected_model = request.POST.get('model')
            model = get_whisper_model(selected_model)
            result = model.transcribe(
                temp_path,  # Use file path, not AudioSegment object
                fp16=False,
                temperature=0.0,
                best_of=1,
                beam_size=3,
                patience=1.0,
                no_speech_threshold=0.6
            )
            text = result["text"]

            # 4. Save to database
            recording = AudioRecording(
                audio_file=filename,  # Save original filename, not AudioSegment
                transcript=text
            )
            recording.save()

            # 5. Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

            # 6. Process with LLM if available and text is not empty
            llm_response = None
            if LLM_AVAILABLE and text.strip():
                try:
                    user_id = get_user_id_for_llm(request)
                    BASE_DIR = Path(__file__).resolve().parent.parent.parent
                    VECTOR_STORE_PATH = os.path.join(BASE_DIR, "data/vectorstores/main_store")
                    memory_path = get_user_memory_path_for_llm(user_id)
                    
                    rag_service = RAGService()
                    # Consume the generator and join all chunks into a single string
                    llm_response_generator = rag_service.get_answer_for_user(
                        user_id=user_id,
                        question=text.strip(),
                        vector_store_path=VECTOR_STORE_PATH,
                        memory_path=memory_path
                    )
                    # Collect all chunks from the generator into a single string
                    llm_response = ''.join(llm_response_generator)
                except Exception as e:
                    print(f"Error processing with LLM: {e}")
                    traceback.print_exc()
                    llm_response = None

            response_data = {
                'status': 'success',
                'transcription': text,
                'audio_url': fs.url(filename)
            }
            
            if llm_response:
                response_data['llm_response'] = llm_response

            return JsonResponse(response_data)

        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"Error in record_audio: {e}")
            print(error_trace)
            
            # Clean up files
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            if filename:
                try:
                    fs.delete(filename)
                except:
                    pass

            return JsonResponse({
                'status': 'error',
                'message': 'An internal error occurred. Please try again later.'
            }, status=500)

    # GET: show recorder; carry through selected model from query param if present
    selected_model = request.GET.get('model')
    # Get user_id for the template
    user_id = get_user_id_for_llm(request)
    
    # Get completed fine-tuning jobs (trained models) - same as home view
    trained_models = FineTuningJob.objects.filter(
        status='completed').order_by('-created_at')
    
    # Get available model files from the model directory (same as home view)
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    available_models = []

    if os.path.exists(model_dir):
        for file in os.listdir(model_dir):
            if file.endswith('.pt'):
                model_path = os.path.join(model_dir, file)
                model_size = os.path.getsize(model_path)
                model_size_mb = round(model_size / (1024 * 1024), 2)
                available_models.append({
                    'name': file,
                    'path': model_path,
                    'size_mb': model_size_mb
                })

    # Standard Whisper models that are available
    standard_models = [
        {'name': 'tiny', 'parameters': '39 M', 'description': 'Fastest, least accurate'},
        {'name': 'base', 'parameters': '74 M',
            'description': 'Good balance of speed and accuracy'},
        {'name': 'small', 'parameters': '244 M',
            'description': 'Better accuracy, slower'},
        {'name': 'medium', 'parameters': '769 M', 'description': 'High accuracy, slower'},
        {'name': 'large', 'parameters': '1550 M',
            'description': 'Best accuracy, slowest'},
        {'name': 'turbo', 'parameters': '809 M', 'description': 'Hybrid'}
    ]
    
    context = {
        'selected_model': selected_model,
        'user_id': user_id,
        'llm_available': LLM_AVAILABLE,
        'trained_models': trained_models,
        'available_models': available_models,
        'standard_models': standard_models,
    }
    return render(request, 'static/record.html', context)


@require_http_methods(["DELETE"])
def delete_recording(request, recording_id):
    try:
        recording = AudioRecording.objects.get(id=recording_id)
        recording.audio_file.delete()
        recording.delete()
        return JsonResponse({'status': 'success'})
    except AudioRecording.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Recording not found'}, status=404)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "nl": "dutch",
    "ar": "arabic",
    "it": "italian",
    "hi": "hindi",
    "ur": "urdu",
    "la": "latin",
    "fa": "persian",
    "az": "azerbaijani",
    "hy": "armenian",
    "tg": "tajik",
    "tk": "turkmen",
}


class WhisperTrainingView(View):
    template_name = 'static/whisper_training.html'

    def get(self, request):
        form = WhisperTrainingForm()
        return render(request, self.template_name, {'form': form})

    def post(self, request):
        form = WhisperTrainingForm(request.POST)
        if form.is_valid():
            try:
                data = form.cleaned_data

                # Validate dataset directory exists
                if not os.path.exists(data['dataset_dir']):
                    messages.error(
                        request, "Dataset directory does not exist!")
                    return render(request, self.template_name, {'form': form})

                # Determine model path
                model_path = (data['custom_model_path'] if data['model_selection'] == 'custom'
                              else f"openai/whisper-{data['model_selection']}")

                # Prepare training arguments
                training_args = {
                    'dataset_dir': data['dataset_dir'],
                    'output_dir': data['output_dir'],
                    'language': data['language'],
                    'WHISPER_MODEL_CHOICE': model_path,
                    'fp16': data['fp16'],
                    'save_steps': data['save_steps'],
                    'eval_steps': data['eval_steps'],
                    'load_best_model_at_end': data['load_best_model'],
                }

                # Start the Celery task
                task = run_training_sequence.delay(**training_args)

                # Store task ID in session
                request.session['training_task_id'] = task.id

                messages.success(request,
                                 "Model training started in background! "
                                 f"<a href='{reverse('training_status')}?task_id={task.id}'>View progress</a>")
                return redirect('training_started')

            except Exception as e:
                messages.error(request, f"Failed to start training: {str(e)}")

        return render(request, self.template_name, {'form': form})


# app/views.py


def training_status(request):
    task_id = request.GET.get(
        'task_id') or request.session.get('training_task_id')
    if not task_id:
        return JsonResponse({'error': 'No task ID provided'}, status=400)

    task_result = AsyncResult(task_id)

    response_data = {
        'task_id': task_id,
        'status': task_result.status,
        'result': task_result.result if task_result.ready() else None
    }

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse(response_data)

    return render(request, 'training_status.html', response_data)


def create_sample_dataset(request):
    """Generate and download a sample dataset structure as a zip file"""
    if request.method == 'POST':
        dataset_name = request.POST.get('dataset_name', 'sample_dataset')
        num_samples = int(request.POST.get('num_samples', 5))

        # Create sample CSV data
        train_data = []
        test_data = []

        sample_sentences = [
            "Hello, this is a sample audio recording",
            "The weather is beautiful today",
            "How are you doing today?",
            "This is a test of the fine-tuning system",
            "Machine learning is fascinating",
            "Audio transcription with Whisper",
            "Fine-tuning custom models",
            "Speech recognition technology",
            "Natural language processing",
            "Artificial intelligence applications"
        ]

        for i in range(num_samples):
            audio_file = f"sample_{i+1}.wav"
            sentence = sample_sentences[i % len(sample_sentences)]

            if i < num_samples * 0.8:  # 80% for training
                train_data.append({"audio": audio_file, "sentence": sentence})
            else:  # 20% for testing
                test_data.append({"audio": audio_file, "sentence": sentence})

        # Create CSV files in memory
        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)

        # Create README content
        readme_content = f"""# Sample Dataset: {dataset_name}

This is a sample dataset structure for Whisper fine-tuning.

## Files:
- train.csv: Training data ({len(train_data)} samples)
- test.csv: Test data ({len(test_data)} samples)
- README.md: This file

## Note:
This dataset contains placeholder audio file references. 
You need to replace the audio file names with actual audio files.

## Audio File Requirements:
- Format: WAV, MP3, or other common audio formats
- Sample Rate: 16kHz (will be converted automatically)
- Duration: Keep under 30 seconds for best results
- Quality: Clear speech, minimal background noise

## Next Steps:
1. Extract this zip file to your desired location
2. Replace the audio file references with your actual audio files
3. Ensure all audio files are in the same directory as the CSV files
4. Use this dataset directory path in the fine-tuning form

## Dataset Structure:
```
{dataset_name}/
├── train.csv
├── test.csv
├── README.md
├── sample_1.wav (replace with your audio)
├── sample_2.wav (replace with your audio)
└── ... (more audio files)
```

## CSV Format:
Both train.csv and test.csv contain exactly these columns:
- audio: Path to audio file (relative to dataset directory)
- sentence: Corresponding transcription text

Example:
```
audio,sentence
sample_1.wav,Hello, this is a sample audio recording
sample_2.wav,The weather is beautiful today
```
"""

        # Create zip file in memory
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add CSV files
            train_csv = train_df.to_csv(index=False)
            test_csv = test_df.to_csv(index=False)

            zip_file.writestr(f"{dataset_name}/train.csv", train_csv)
            zip_file.writestr(f"{dataset_name}/test.csv", test_csv)
            zip_file.writestr(f"{dataset_name}/README.md", readme_content)

            # Add placeholder audio files (empty files with .wav extension)
            for i in range(num_samples):
                zip_file.writestr(f"{dataset_name}/sample_{i+1}.wav", b"")

        # Prepare response
        zip_buffer.seek(0)
        response = HttpResponse(zip_buffer.getvalue(),
                                content_type='application/zip')
        response['Content-Disposition'] = f'attachment; filename="{dataset_name}_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip"'

        return response

    return render(request, 'create_sample_dataset.html')


def websocket_test(request):
    """Serve the WebSocket test page"""
    return render(request, 'websocket_test.html')
