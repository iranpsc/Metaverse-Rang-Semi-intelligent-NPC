from django.core.management.base import BaseCommand, CommandError

from LLM.rag_service import RAGService


class Command(BaseCommand):
    help = "Rebuild a FAISS vector store from a CSV dataset using the configured embeddings."

    def add_arguments(self, parser):
        parser.add_argument("--dataset", required=True, help="Absolute path to the CSV dataset.")
        parser.add_argument("--vector-store", required=True, help="Directory where the vector store will be saved.")
        parser.add_argument("--user-id", default="system", help="User identifier for logging purposes.")

    def handle(self, *args, **options):
        dataset_path = options["dataset"]
        vector_store_path = options["vector_store"]
        user_id = options["user_id"]

        rag_service = RAGService()
        result = rag_service.build_vector_store(
            dataset_path=dataset_path,
            vector_store_path=vector_store_path,
            user_id=user_id,
        )

        if result.get("status") != "success":
            raise CommandError(result.get("message", "Failed to rebuild vector store."))

        self.stdout.write(
            self.style.SUCCESS(
                f"Vector store rebuilt at {vector_store_path} using dataset {dataset_path} (user_id={user_id})."
            )
        )

