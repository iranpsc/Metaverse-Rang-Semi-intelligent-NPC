import torch
import traceback
import threading
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from .config import CONFIG
from .rag_system import RAGSession, VectorStoreManager, MemoryManager, RSSUpdater

class RAGService:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RAGService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized: return
        
        print("Initializing RAGService...")
        self.config = CONFIG
        self.embeddings = None
        self.llm = None
        self._initialize_base_models()
        self._initialized = True
        print("RAGService ready.")

    def _initialize_base_models(self):
        device = 'cuda' if torch.cuda.is_available() and self.config['USE_GPU'] else 'cpu'
        print(f"Loading Embeddings on {device}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config['EMBEDDING_MODEL'],
            model_kwargs={'device': device},
            encode_kwargs={
                'batch_size': self.config['EMBEDDING_BATCH_SIZE'],
                'normalize_embeddings': self.config['NORMALIZE_EMBEDDINGS'],
            }
        )
        
        print("Loading LLM...")
        self.llm = OllamaLLM(
            model=self.config['LLM_MODEL'], base_url=self.config['OLLAMA_BASE_URL'],
            temperature=self.config['TEMPERATURE'], num_predict=self.config['NUM_PREDICT'],
            repeat_penalty=self.config['REPEAT_PENALTY'], top_k=self.config['TOP_K'],
            top_p=self.config['TOP_P'],
            stop=self.config['STOP_TOKENS']
        )

    # --- API Methods ---

    def process_user_rss(self, user_id: str, rss_links: list, dataset_path: str, vector_store_path: str) -> dict:
        """
        API Method: این تابع توسط بک‌کند فراخوانی می‌شود.
        لینک‌های RSS داده شده را چک می‌کند، متن کامل را استخراج می‌کند و به دیتابیس و وکتور استور کاربر اضافه می‌کند.
        
        Args:
            user_id: شناسه کاربر
            rss_links: لیست لینک‌های RSS (مثلا ['url1', 'url2'])
            dataset_path: مسیر فایل CSV کاربر
            vector_store_path: مسیر پوشه وکتور استور کاربر
        """
        if not rss_links:
            return {"status": "warning", "message": "No RSS links provided."}

        print(f"INFO: Starting RSS processing for user {user_id} with {len(rss_links)} feeds.")
        
        try:
            updater = RSSUpdater(
                embeddings=self.embeddings,
                dataset_path=dataset_path,
                vector_store_path=vector_store_path,
                user_id=user_id,
                verbose=True
            )
            
            # 1. دریافت داده‌های جدید
            new_entries = updater.fetch_new_entries(rss_links)
            
            if not new_entries:
                return {"status": "success", "message": "No new articles found in provided RSS feeds.", "new_count": 0}

            # 2. ذخیره در CSV
            updater.append_to_dataset(new_entries)
            
            # 3. ساخت داکیومنت و آپدیت وکتور استور
            new_docs = updater.create_documents(new_entries)
            if not new_docs:
                print(f"WARNING: No documents created from {len(new_entries)} entries for user {user_id}.")
                return {"status": "warning", "message": "No documents could be created from entries.", "new_count": len(new_entries)}
            
            print(f"INFO: Created {len(new_docs)} documents, updating vector store at {vector_store_path}")
            updater.vstore_manager.append_documents(new_docs, vector_store_path)
            print(f"INFO: Vector store updated successfully at {vector_store_path}")
            
            # 4. ذخیره وضعیت (تاریخ‌ها)
            updater.update_state(new_entries)
            
            count = len(new_entries)
            print(f"SUCCESS: Processed {count} new articles for user {user_id}.")
            return {"status": "success", "message": f"Successfully added {count} new articles.", "new_count": count}

        except Exception as e:
            print(f"ERROR in process_user_rss for user {user_id}: {e}")
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def build_vector_store(self, dataset_path: str, vector_store_path: str, user_id: str = "system") -> dict:
        try:
            vstore_manager = VectorStoreManager(embeddings=self.embeddings, user_id=user_id, verbose=True)
            return vstore_manager.build(dataset_path, vector_store_path)
        except Exception as e:
            print(f"FATAL ERROR: {e}")
            return {"status": "error", "message": str(e)}

    def get_answer_for_user(self, user_id: str, question: str, vector_store_path: str, memory_path: str) -> str:
        try:
            rag_session = RAGSession(
                user_id=user_id,
                llm=self.llm,
                embeddings=self.embeddings,
                config=self.config
            )
            return rag_session.get_answer(question, vector_store_path, memory_path)
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
            return "خطای داخلی."

    def clear_user_memory(self, memory_path: str, user_id: str = "system") -> dict:
        try:
            memory_manager = MemoryManager(config=self.config, user_id=user_id, verbose=True)
            return memory_manager.clear(memory_path)
        except Exception as e:
            return {"status": "error", "message": str(e)}