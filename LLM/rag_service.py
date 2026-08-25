import config
from django import conf
import torch
import traceback
import threading
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
import trafilatura
from .config import CONFIG
from .rag_system import RAGSession, VectorStoreManager, MemoryManager, RSSUpdater
from .utils import extract_text_from_file
import os

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
        
    def _create_llm(self, model_name: str = None):

        target_model = model_name if model_name else self.config['LLM_MODEL']

        return OllamaLLM(
            model=target_model,
            base_url=self.config['OLLAMA_BASE_URL'],
            temperature=self.config['TEMPERATURE'],
            num_predict=self.config['NUM_PREDICT'],
            repeat_penalty=self.config['REPEAT_PENALTY'],
            top_k=self.config['TOP_K'],
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
            updater.vstore_manager.append_documents(new_docs, vector_store_path)
            
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
            traceback.print_exc()
            return {"status": "error", "message": "Failed to build vector store due to an internal error."}

    def get_answer_for_user(self, user_id: str, question: str, vector_store_path: str, memory_path: str, model_name: str = None):
        """
        Generator function that yields answer chunks from RAG system.
        Uses yield from to delegate to the inner generator.
        """
        try:
            current_llm = self._create_llm(model_name)

            rag_session = RAGSession(
                user_id=user_id,
                llm=current_llm,
                embeddings=self.embeddings,
                config=self.config
            )
            # Use yield from to delegate to the inner generator
            yield from rag_session.get_answer(question, vector_store_path, memory_path)
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
            
            yield "خطای داخلی در پردازش درخواست."

    def process_single_url(self, user_id: str, url: str, dataset_path: str, vector_store_path: str, title: str = None) -> dict:
        """
        API Method: Process a single article URL and add it to the user's dataset and vector store.
        
        Args:
            user_id: شناسه کاربر
            url: URL of the article/webpage to process
            dataset_path: مسیر فایل CSV کاربر
            vector_store_path: مسیر پوشه وکتور استور کاربر
            title: Optional title for the article (if not provided, will try to extract from URL)
        """
        if not url or not url.strip():
            return {"status": "error", "message": "URL is required."}

        url = url.strip()
        print(f"INFO: Processing single URL for user {user_id}: {url}")
        
        try:
            # Extract content from URL using trafilatura
            print(f"INFO: Extracting content from URL...")
            
            # Download and extract content
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                return {"status": "error", "message": "Could not download the URL. The page might be inaccessible."}
            
            # Extract text content
            full_text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            
            if not full_text or not full_text.strip():
                return {"status": "error", "message": "Could not extract content from the provided URL. The page might be empty or inaccessible."}
            
            # Extract metadata (if available)
            try:
                metadata = trafilatura.extract_metadata(downloaded)
                article_title = title or (getattr(metadata, 'title', None) if metadata else None) or url
                author = getattr(metadata, 'author', '') if metadata else ""
                published_date = getattr(metadata, 'date', None) if metadata else None
                if not published_date:
                    published_date = datetime.now().isoformat()
            except Exception as meta_error:
                print(f"WARNING: Could not extract metadata: {meta_error}")
                article_title = title or url
                author = ""
                published_date = datetime.now().isoformat()
            
            # Create entry similar to RSS entry format
            entry = {
                'title': article_title,
                'link': url,
                'pubDate': published_date,
                'content': full_text.strip(),
                'feed_url': url,  # For consistency with RSS format
                'source_type': 'single_url',  # Metadata to distinguish from RSS
                'author': author
            }
            
            # Use RSSUpdater's methods to append to dataset and vector store
            updater = RSSUpdater(
                embeddings=self.embeddings,
                dataset_path=dataset_path,
                vector_store_path=vector_store_path,
                user_id=user_id,
                verbose=True
            )
            
            # Append to dataset
            updater.append_to_dataset([entry])
            
            # Create documents and update vector store
            new_docs = updater.create_documents([entry])
            updater.vstore_manager.append_documents(new_docs, vector_store_path)
            
            print(f"SUCCESS: Processed URL for user {user_id}: {article_title}")
            return {
                "status": "success", 
                "message": f"Successfully added article: {article_title}", 
                "new_count": 1,
                "title": article_title,
                "url": url
            }

        except Exception as e:
            print(f"ERROR in process_single_url for user {user_id}: {e}")
            traceback.print_exc()
            return {"status": "error", "message": str(e)}
    
    def process_user_files(
        self,
        user_id: str,
        file_paths: list,
        dataset_path: str,
        vector_store_path: str
    ) -> dict:
        """
        Process uploaded files (PDF, DOCX, TXT, CSV) and add them to user's vector store.
        """

        if not file_paths:
            return {"status": "warning", "message": "No files provided."}

        print(f"INFO: Processing {len(file_paths)} files for user {user_id}")

        updater = RSSUpdater(
            embeddings=self.embeddings,
            dataset_path=dataset_path,
            vector_store_path=vector_store_path,
            user_id=user_id,
            verbose=True
        )

        entries = []

        for file_path in file_paths:
            try:
                text = extract_text_from_file(file_path)

                if not text.strip():
                    print(f"WARNING: Empty content in {file_path}")
                    continue

                filename = os.path.basename(file_path)

                entry = {
                    "title": filename,
                    "link": file_path,
                    "pubDate": datetime.now().isoformat(),
                    "content": text,
                    "feed_url": "local_file",
                    "source_type": "file",
                    "author": user_id
                }

                entries.append(entry)

            except Exception as e:
                print(f"ERROR reading {file_path}: {e}")

        if not entries:
            return {
                "status": "warning",
                "message": "No valid content extracted from files.",
                "new_count": 0
            }

        # 1. Save to dataset CSV
        updater.append_to_dataset(entries)

        # 2. Create documents
        docs = updater.create_documents(entries)

        # 3. Update vector store
        updater.vstore_manager.append_documents(docs, vector_store_path)

        print(f"SUCCESS: Added {len(entries)} files to vector store.")

        return {
            "status": "success",
            "message": f"Added {len(entries)} files to knowledge base.",
            "new_count": len(entries)
        }


    def clear_user_memory(self, memory_path: str, user_id: str = "system") -> dict:
        try:
            memory_manager = MemoryManager(config=self.config, user_id=user_id, verbose=True)
            return memory_manager.clear(memory_path)
        except Exception as e:
            return {"status": "error", "message": str(e)}