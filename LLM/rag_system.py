from pathlib import Path
from typing import List, Optional, Dict
import pandas as pd
import shutil
import json
from datetime import datetime
from time import mktime

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import feedparser
import trafilatura

class VectorStoreManager:
    """مدیریت ساخت و بارگذاری وکتوراستورها."""
    def __init__(self, embeddings: HuggingFaceEmbeddings, user_id: str = "system", verbose: bool = False):
        self.embeddings = embeddings
        self.user_id = user_id
        self.verbose = verbose

    def _log(self, message: str):
        if self.verbose: print(f"[VectorStoreManager][{self.user_id}] {message}")

    def _create_documents(self, df: pd.DataFrame) -> List[Document]:
        documents = []
        for _, row in df.iterrows():
            # تبدیل تمام ستون‌ها به فرمت متنی یکپارچه
            # این بخش هوشمند است: اگر ستون جدیدی اضافه شود، خودکار هندل می‌شود
            content_parts = [f"{str(col).replace('_', ' ').strip().title()}: {str(val)}"
                             for col, val in row.items() if pd.notna(val) and str(val).strip() != '']
            content = "\n".join(content_parts)
            documents.append(Document(page_content=content, metadata={'user_id': self.user_id}))
        return documents

    def build(self, dataset_path: str, vector_store_path: str) -> Dict:
        try:
            self._log(f"Building vector store from '{dataset_path}' to '{vector_store_path}'")
            vs_path = Path(vector_store_path)
            
            if vs_path.exists() and any(vs_path.iterdir()):
                 shutil.rmtree(vs_path)
                 self._log("Existing vector store removed for rebuild.")

            if not Path(dataset_path).exists():
                 return {"status": "error", "message": "Dataset file not found."}

            df = pd.read_csv(dataset_path)
            documents = self._create_documents(df)
            if not documents:
                return {"status": "error", "message": "No documents created from dataset."}

            vectorstore = FAISS.from_documents(documents, self.embeddings)
            vs_path.mkdir(parents=True, exist_ok=True)
            vectorstore.save_local(str(vs_path))
            return {"status": "success", "message": f"Vector store built successfully."}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def load(self, vector_store_path: str) -> Optional[FAISS]:
        try:
            vs_path = Path(vector_store_path)
            index_file = vs_path / "index.faiss"
            self._log(f"Attempting to load vector store from: {vs_path}")
            self._log(f"Index file exists: {index_file.exists()}")
            
            if not index_file.exists():
                self._log(f"Vector store index not found at {index_file}")
                return None
            
            vectorstore = FAISS.load_local(str(vs_path), self.embeddings, allow_dangerous_deserialization=True)
            
            # Check if vectorstore has any documents
            if hasattr(vectorstore, 'index') and hasattr(vectorstore.index, 'ntotal'):
                doc_count = vectorstore.index.ntotal
                self._log(f"Vector store loaded successfully with {doc_count} documents")
                if doc_count == 0:
                    self._log("WARNING: Vector store is empty (0 documents)")
                    return None
            else:
                self._log("WARNING: Could not determine document count in vector store")
            
            return vectorstore
        except Exception as e:
            self._log(f"Error loading vector store from {vector_store_path}: {e}")
            import traceback
            self._log(f"Traceback: {traceback.format_exc()}")
            return None

    def append_documents(self, new_documents: List[Document], vector_store_path: str):
        """Documents جدید را به vectorstore موجود append می‌کند."""
        try:
            if not new_documents:
                self._log("WARNING: No documents to append.")
                return
                
            vs_path = Path(vector_store_path)
            # Ensure directory exists first
            vs_path.mkdir(parents=True, exist_ok=True)
            
            vectorstore = None
            index_file = vs_path / "index.faiss"
            
            if index_file.exists():
                vectorstore = self.load(vector_store_path)
            
            if vectorstore:
                vectorstore.add_documents(new_documents)
                self._log(f"Added {len(new_documents)} documents to existing index.")
            else:
                vectorstore = FAISS.from_documents(new_documents, self.embeddings)
                self._log(f"Created new index with {len(new_documents)} documents.")
            
            vectorstore.save_local(str(vs_path))
            self._log(f"Successfully saved/updated vector store at {vector_store_path}")
        except Exception as e:
            self._log(f"Error appending documents: {e}")
            import traceback
            traceback.print_exc()
            raise e

class MemoryManager:
    """مدیریت حافظه مکالمات کاربر."""
    def __init__(self, config: dict, user_id: str = "system", verbose: bool = False):
        self.config = config
        self.user_id = user_id
        self.verbose = verbose

    def _log(self, message: str):
        if self.verbose: print(f"[MemoryManager][{self.user_id}] {message}")
        
    def clear(self, memory_path: str) -> Dict:
        try:
            mem_path = Path(memory_path)
            if mem_path.exists():
                shutil.rmtree(mem_path)
            mem_path.mkdir(parents=True, exist_ok=True)
            return {"status": "success", "message": "Memory cleared."}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_context(self, memory_path: str) -> str:
        history_file = Path(memory_path) / "conversation_history.json"
        if not history_file.exists():
            return "هیچ تاریخچه مکالمه‌ای وجود ندارد."
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            recent_history = history[-5:] 
            context_text = "\n---\n".join([f"س: {mem['question']}\nج: {mem['answer']}" for mem in recent_history])
            return context_text
        except Exception as e:
            return "خطا در خواندن تاریخچه."

    def add_interaction(self, memory_path: str, question: str, answer: str):
        history_file = Path(memory_path) / "conversation_history.json"
        history = []
        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except: pass
        
        interaction = {"timestamp": datetime.now().isoformat(), "question": question, "answer": answer}
        history.append(interaction)
        # if len(history) > 50: 
        #     history = history[-50:]
            
        try:
            Path(memory_path).mkdir(parents=True, exist_ok=True)
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self._log(f"Error saving interaction: {e}")

class RSSUpdater:
    """
    مدیریت به‌روزرسانی داینامیک RSS برای هر کاربر.
    لینک‌ها را از ورودی تابع می‌گیرد، نه از Config.
    """
    def __init__(self, embeddings: HuggingFaceEmbeddings, dataset_path: str, vector_store_path: str, user_id: str, verbose: bool = False):
        self.embeddings = embeddings
        self.dataset_path = Path(dataset_path)
        self.vector_store_path = Path(vector_store_path)
        self.user_id = user_id
        self.verbose = verbose
        self.vstore_manager = VectorStoreManager(self.embeddings, user_id=self.user_id, verbose=self.verbose)
        
        # فایل وضعیت RSS مخصوص این دیتاست/کاربر که در کنار فایل CSV ذخیره می‌شود
        self.state_file = self.dataset_path.parent / f"rss_state_{self.user_id}.json"

    def _log(self, message: str):
        if self.verbose: print(f"[RSSUpdater][{self.user_id}] {message}")

    def _load_state(self) -> dict:
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self._log(f"Error loading state: {e}")
        return {}

    def _save_state(self, state: dict):
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self._log(f"Error saving state: {e}")

    def fetch_new_entries(self, rss_links: List[str]) -> List[dict]:
        """
        لیست لینک‌ها را می‌گیرد، چک می‌کند و موارد جدید را برمی‌گرداند.
        """
        state = self._load_state()
        new_entries = []
        
        for feed_url in rss_links:
            try:
                self._log(f"Checking feed: {feed_url}")
                feed = feedparser.parse(feed_url)
                last_pubdate = state.get(feed_url) # تاریخ آخرین چک برای این لینک خاص
                
                entries_to_process = []
                for entry in feed.entries:
                    pubdate = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pubdate = datetime.fromtimestamp(mktime(entry.published_parsed)).isoformat()
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        pubdate = datetime.fromtimestamp(mktime(entry.updated_parsed)).isoformat()
                    
                    # اگر تاریخ نداشت، از زمان حال استفاده کن (ریسک تکراری بودن دارد، اما برای RSSهای خراب چاره‌ای نیست)
                    if not pubdate:
                         # self._log(f"No pubdate for entry: {entry.title}, skipping logic based on date might be needed")
                         continue

                    # اگر جدیدتر از آخرین بار بود
                    if not last_pubdate or pubdate > last_pubdate:
                        entries_to_process.append((entry, pubdate))
                
                # پردازش متن کامل (این بخش زمان‌بر است)
                for entry, pubdate in entries_to_process:
                    try:
                        # دانلود متن کامل
                        downloaded = trafilatura.fetch_url(entry.link)
                        if downloaded:
                            full_text = trafilatura.extract(downloaded)
                            if full_text:
                                new_entries.append({
                                    'title': getattr(entry, 'title', 'No title'),
                                    'link': getattr(entry, 'link', ''),
                                    'pubDate': pubdate,
                                    'content': full_text.strip(), # متن کامل
                                    'feed_url': feed_url, # برای ردیابی منبع
                                    'source_type': 'rss_auto' # متادیتا برای تشخیص نوع داده
                                })
                                self._log(f"Extracted full text for: {entry.title}")
                    except Exception as sub_e:
                         self._log(f"Error extracting text from {entry.link}: {sub_e}")

            except Exception as e:
                self._log(f"Error fetching feed {feed_url}: {e}")
                
        return new_entries

    def append_to_dataset(self, new_entries: List[dict]):
        """داده‌های جدید را به CSV اضافه می‌کند (Concatenate)."""
        if not new_entries: return
        
        new_df = pd.DataFrame(new_entries)
        
        if self.dataset_path.exists():
            try:
                existing_df = pd.read_csv(self.dataset_path)
                # هوشمندی Pandas: اگر ستون‌ها فرق کنند، با NaN پر می‌کند
                updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            except pd.errors.EmptyDataError:
                updated_df = new_df
        else:
            updated_df = new_df
        
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        updated_df.to_csv(self.dataset_path, index=False, encoding='utf-8')
        self._log(f"Appended {len(new_entries)} rows to {self.dataset_path}")

    def create_documents(self, entries: List[dict]) -> List[Document]:
        df = pd.DataFrame(entries)
        return self.vstore_manager._create_documents(df)

    def update_state(self, new_entries: List[dict]):
        """تاریخ آخرین آیتم دریافت شده برای هر لینک را ذخیره می‌کند."""
        if not new_entries: return
        
        state = self._load_state()
        
        # پیدا کردن ماکزیمم تاریخ برای هر feed_url در داده‌های جدید
        for entry in new_entries:
            feed_url = entry['feed_url']
            current_max = state.get(feed_url, '1970-01-01T00:00:00')
            if entry['pubDate'] > current_max:
                state[feed_url] = entry['pubDate']
                
        self._save_state(state)
        self._log("RSS State updated.")

class RAGSession:
    """جلسه پرسش و پاسخ."""
    def __init__(self, user_id: str, llm: OllamaLLM, embeddings: HuggingFaceEmbeddings, config: dict):
        self.user_id = user_id
        self.llm = llm
        self.embeddings = embeddings
        self.config = config
        self.vstore_manager = VectorStoreManager(embeddings, user_id=self.user_id, verbose=True)
        self.memory_manager = MemoryManager(config, user_id=self.user_id, verbose=True)

    def _create_rag_chain(self, vectorstore: FAISS, memory_context: str):
        retriever = vectorstore.as_retriever(
            search_type=self.config['SEARCH_TYPE'],
            search_kwargs={"k": self.config.get('TOP_K_RESULTS', 2)}
        )
        prompt = PromptTemplate(
            template="""شما یک دستیار هوشمند هستید.
با استفاده از "متن مرجع" به "سوال کاربر" پاسخ دهید.
اگر پاسخ در متن نبود، بگویید "اطلاعاتی ندارم".

*** دستورالعمل: پایان پاسخ حتما [END] بنویسید. ***

متن مرجع:
{context}

تاریخچه:
{memory_context}

سوال:
{question}

پاسخ:""",
            input_variables=["context", "question", "memory_context"]
        )
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
                "memory_context": lambda x: memory_context
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain

    def _create_fallback_chain(self, memory_context: str):
        """Create a chain without vector store retrieval - uses only LLM and memory."""
        prompt = PromptTemplate(
            template="""شما یک دستیار هوشمند هستید.
به "سوال کاربر" پاسخ دهید. از دانش عمومی و تجربه خود استفاده کنید.

*** دستورالعمل: پایان پاسخ حتما [END] بنویسید. ***

تاریخچه:
{memory_context}

سوال:
{question}

پاسخ:""",
            input_variables=["question", "memory_context"]
        )
        
        fallback_chain = (
            {
                "question": RunnablePassthrough(),
                "memory_context": lambda x: memory_context
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return fallback_chain

    def get_answer(self, question: str, vector_store_path: str, memory_path: str) -> str:
        self.vstore_manager._log(f"Getting answer for question. Vector store path: {vector_store_path}")
        vectorstore = self.vstore_manager.load(vector_store_path)
        
        memory_context = ""
        if self.config.get('ENABLE_MEMORY', True):
            memory_context = self.memory_manager.get_context(memory_path)
        
        # If vector store is available, use RAG chain
        if vectorstore:
            self.vstore_manager._log(f"Using RAG chain with vector store: {vector_store_path}")
            rag_chain = self._create_rag_chain(vectorstore, memory_context)
            answer = rag_chain.invoke(question)
        else:
            # If no vector store, use fallback chain (LLM only with memory)
            self.vstore_manager._log(f"Vector store not available at {vector_store_path}, using fallback chain (LLM only)")
            vs_path = Path(vector_store_path)
            if vs_path.exists() and not (vs_path / "index.faiss").exists():
                self.vstore_manager._log(f"Vector store directory exists but is empty: {vector_store_path}")
            else:
                self.vstore_manager._log(f"Vector store path does not exist: {vector_store_path}")
            
            # Use fallback chain that doesn't require vector store
            fallback_chain = self._create_fallback_chain(memory_context)
            answer = fallback_chain.invoke(question)

        if self.config.get('ENABLE_MEMORY', True):
            self.memory_manager.add_interaction(memory_path, question, answer)
            
        return answer.strip()