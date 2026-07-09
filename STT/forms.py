# app/forms.py
from django import forms


class WhisperTrainingForm(forms.Form):
    WHISPER_MODEL_CHOICES = [
        ('tiny', 'Tiny (openai/whisper-tiny)'),
        ('base', 'Base (openai/whisper-base)'),
        ('small', 'Small (openai/whisper-small)'),
        ('medium', 'Medium (openai/whisper-medium)'),
        ('large', 'Large (openai/whisper-large)'),
        ('turbo', 'Turbo (openai/whisper-turbo)'),
        ('custom', 'Custom Path'),
    ]

    model_selection = forms.ChoiceField(
        label='Model Selection',
        choices=WHISPER_MODEL_CHOICES,
        widget=forms.Select(attrs={
            'class': 'form-select',
            'id': 'model-selection-select',
            'onchange': "toggleCustomModelField()"  # JavaScript to handle custom path
        })
    )

    custom_model_path = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter custom model path',
            'style': 'display: none;',
            'id': 'custom-model-path'
        })
    )

    LANGUAGE_CHOICES = [
        ('english', 'English'),
        ('french', 'French'),
        ('spanish', 'Spanish'),
        # Add more languages as needed
    ]

    dataset_dir = forms.CharField(
        label='Dataset Directory',
        max_length=255,
        widget=forms.TextInput(attrs={'placeholder': '/path/to/dataset'})
    )

    output_dir = forms.CharField(
        label='Output Directory',
        max_length=255,
        widget=forms.TextInput(attrs={'placeholder': '/path/to/output'})
    )

    language = forms.ChoiceField(choices=LANGUAGE_CHOICES)

    fp16 = forms.BooleanField(
        label='Use FP16 Precision',
        required=False,
        initial=True
    )

    save_steps = forms.IntegerField(
        label='Save Steps',
        initial=500,
        min_value=100,
        max_value=10000
    )

    eval_steps = forms.IntegerField(
        label='Evaluation Steps',
        initial=500,
        min_value=100,
        max_value=10000
    )

    load_best_model = forms.BooleanField(
        label='Load Best Model at End',
        required=False,
        initial=True
    )

    def clean(self):
        cleaned_data = super().clean()
        model_selection = cleaned_data.get('model_selection')
        custom_model_path = cleaned_data.get('custom_model_path')

        if model_selection == 'custom' and not custom_model_path:
            self.add_error('custom_model_path',
                           'Please provide a custom model path')

        return cleaned_data
