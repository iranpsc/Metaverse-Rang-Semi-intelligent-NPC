from datasets import Dataset
import pandas as pd
from datasets import Audio
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# Compatibility patch for older torch versions expected by transformers.
try:
    import torch.utils._pytree as _pytree  # type: ignore

    if not hasattr(_pytree, "register_pytree_node") and hasattr(_pytree, "_register_pytree_node"):
        def _register_pytree_node_shim(*args, **kwargs):
            # Transformers>=4.57 passes future kwargs we can safely ignore.
            kwargs.pop("serialized_type_name", None)
            kwargs.pop("serialized_parameters", None)
            return _pytree._register_pytree_node(*args, **kwargs)  # type: ignore[attr-defined]

        _pytree.register_pytree_node = _register_pytree_node_shim  # type: ignore[attr-defined]
except Exception:
    pass

from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
import evaluate
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer


def training_sequence(dataset_dir, output_dir, language, WHISPER_MODEL_CHOICE, fp16=True, save_steps=500, eval_steps=500, load_best_model_at_end=True, task_id=None):
    import os

    # Validate dataset directory exists
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # Check for required CSV files
    train_csv = os.path.join(dataset_dir, "train.csv")
    test_csv = os.path.join(dataset_dir, "test.csv")

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    # Load the datasets
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Validate CSV structure
    required_columns = ["audio", "sentence"]
    for df_name, df in [("train", train_df), ("test", test_df)]:
        if not all(col in df.columns for col in required_columns):
            available_cols = ", ".join(df.columns.tolist())
            raise ValueError(
                f"{df_name}.csv must contain columns 'audio' and 'sentence'. Found: {available_cols}")

    # Rename columns to ensure consistency
    train_df = train_df[required_columns].copy()
    test_df = test_df[required_columns].copy()

    # Validate that audio files exist
    for df_name, df in [("train", train_df), ("test", test_df)]:
        missing_files = []
        for idx, audio_path in enumerate(df["audio"]):
            full_path = os.path.join(dataset_dir, audio_path)
            if not os.path.exists(full_path):
                missing_files.append(audio_path)
        if missing_files:
            raise FileNotFoundError(
                f"Missing audio files in {df_name}: {missing_files[:5]}...")

    # convert the pandas dataframes to dataset
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # convert the sample rate of every audio files using cast_column function
    train_dataset = train_dataset.cast_column(
        "audio", Audio(sampling_rate=16000))
    test_dataset = test_dataset.cast_column(
        "audio", Audio(sampling_rate=16000))

    # import feature extractor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        f"{WHISPER_MODEL_CHOICE}")

    # Load WhisperTokenizer
    tokenizer = WhisperTokenizer.from_pretrained(
        f"{WHISPER_MODEL_CHOICE}", language=language, task="transcribe")

    # Combine To Create A WhisperProcessor
    processor = WhisperProcessor.from_pretrained(
        f"{WHISPER_MODEL_CHOICE}", language=language, task="transcribe")

    def prepare_dataset(examples):
        # compute log-Mel input features from input audio array
        audio = examples["audio"]
        examples["input_features"] = feature_extractor(
            audio["array"], sampling_rate=16000).input_features[0]
        del examples["audio"]
        sentences = examples["sentence"]

        # encode target text to label ids
        examples["labels"] = tokenizer(sentences).input_ids
        del examples["sentence"]
        return examples

    train_dataset = train_dataset.map(prepare_dataset, num_proc=1)
    test_dataset = test_dataset.map(prepare_dataset, num_proc=1)

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # split inputs and labels since they have to be of different lengths and need different padding methods
            # first treat the audio inputs by simply returning torch tensors
            input_features = [
                {"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(
                input_features, return_tensors="pt")
            # get the tokenized label sequences
            label_features = [{"input_ids": feature["labels"]}
                              for feature in features]
            # pad the labels to max length
            labels_batch = self.processor.tokenizer.pad(
                label_features, return_tensors="pt")
            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100)
            # if bos token is appended in previous tokenization step,
            # cut bos token here as it's append later anyways
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    # lets initiate the data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Load a Pre-Trained Checkpoint
    model = WhisperForConditionalGeneration.from_pretrained(
        f"{WHISPER_MODEL_CHOICE}")

    training_args = Seq2SeqTrainingArguments(
        # change to a repo name of your choice
        output_dir=f"model/{output_dir}",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=15000,
        gradient_checkpointing=True,
        fp16=fp16,
        evaluation_strategy="steps",
        per_device_eval_batch_size=1,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=100,
        report_to=["tensorboard"],
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    class ProgressReportingTrainer(Seq2SeqTrainer):
        def __init__(self, *args, **kwargs):
            self.task_id = task_id
            super().__init__(*args, **kwargs)

        def training_step(self, model, inputs):
            # Update progress if we have a task_id
            if self.task_id:
                from celery import current_task
                if current_task:
                    current_task.update_state(
                        state='PROGRESS',
                        meta={
                            'current': self.state.global_step,
                            'total': self.state.max_steps,
                            'status': f'Training step {self.state.global_step}/{self.state.max_steps}',
                            'logs': f'Training step {self.state.global_step}/{self.state.max_steps} - Loss: {self.state.log_history[-1].get("train_loss", "N/A") if self.state.log_history else "N/A"}'
                        }
                    )
            return super().training_step(model, inputs)

    trainer = ProgressReportingTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    # start the model training
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    processor.save_pretrained(f"model/{output_dir}")

    # Update progress to indicate completion
    if task_id:
        from celery import current_task
        if current_task:
            current_task.update_state(
                state='SUCCESS',
                meta={
                    'status': 'Training completed successfully',
                    'model_path': f"model/{output_dir}",
                    'logs': 'Model training completed and saved successfully'
                }
            )
