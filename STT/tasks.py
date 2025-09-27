# app/tasks.py
from celery import shared_task
from celery.utils.log import get_task_logger
from .utils import training_sequence  # Import your original function

logger = get_task_logger(__name__)


@shared_task(bind=True)
def run_training_sequence(self, **kwargs):
    """
    Celery task wrapper for the Whisper fine-tuning sequence
    """
    try:
        logger.info(f"Starting Whisper training with params: {kwargs}")

        # Add task_id to output_dir for unique logging
        if 'output_dir' in kwargs:
            kwargs['output_dir'] = f"{kwargs['output_dir']}_{self.request.id}"

        # Add task_id to kwargs for progress reporting
        kwargs['task_id'] = self.request.id

        # Run the actual training
        training_sequence(**kwargs)

        logger.info("Training completed successfully")
        return {
            'status': 'SUCCESS',
            'result': 'Training completed',
            'task_id': self.request.id
        }

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        self.retry(exc=e, countdown=60, max_retries=3)
