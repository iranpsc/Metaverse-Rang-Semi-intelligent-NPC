# Generated for the realtime agent control plane.
import uuid

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = [("LLM", "0001_initial")]

    operations = [
        migrations.CreateModel(
            name="AgentSession",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4, editable=False, primary_key=True, serialize=False
                    ),
                ),
                ("room_name", models.CharField(max_length=128, unique=True)),
                ("participant_identity", models.CharField(max_length=128, unique=True)),
                ("user_id", models.CharField(max_length=128)),
                ("vector_store", models.CharField(default="main_store", max_length=128)),
                ("dispatch_id", models.CharField(blank=True, max_length=128)),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("creating", "Creating"),
                            ("active", "Active"),
                            ("failed", "Failed"),
                            ("closed", "Closed"),
                        ],
                        default="creating",
                        max_length=16,
                    ),
                ),
                ("expires_at", models.DateTimeField()),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "voice_sample",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        to="LLM.voicesample",
                    ),
                ),
            ],
            options={"ordering": ("-created_at",)},
        )
    ]

