from pathlib import Path
from typing import List, Optional, Dict, Generator
import pandas as pd
import shutil
import json
from datetime import datetime
from time import mktime
import time
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
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
            if not (vs_path / "index.faiss").exists():
                return None
            return FAISS.load_local(str(vs_path), self.embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            self._log(f"Error loading vector store: {e}")
            return None

    def append_documents(self, new_documents: List[Document], vector_store_path: str):
        """Documents جدید را به vectorstore موجود append می‌کند."""
        try:
            vs_path = Path(vector_store_path)
            vectorstore = None
            
            if vs_path.exists() and (vs_path / "index.faiss").exists():
                vectorstore = self.load(vector_store_path)
            
            if vectorstore:
                vectorstore.add_documents(new_documents)
                self._log("Added documents to existing index.")
            else:
                vectorstore = FAISS.from_documents(new_documents, self.embeddings)
                self._log("Created new index with documents.")
            
            vs_path.mkdir(parents=True, exist_ok=True)
            vectorstore.save_local(str(vs_path))
            self._log(f"Successfully saved/updated vector store at {vector_store_path}")
        except Exception as e:
            self._log(f"Error appending documents: {e}")
            raise e

class MemoryManager:
    """مدیریت حافظه مکالمات کاربر."""
    def __init__(self, config: dict, user_id: str = "system", verbose: bool = False):
        self.config = config
        self.user_id = user_id
        self.verbose = verbose

    def _log(self, message: str):
        if self.verbose: print(f"[MemoryManager][{self.user_id}] {message}")

    def _resolve_safe_memory_path(self, memory_path: str) -> Path:
        base_path = Path(self.config.get("MEMORY_BASE_PATH", "")).resolve()
        target_path = Path(memory_path).resolve()
        try:
            target_path.relative_to(base_path)
        except ValueError:
            raise ValueError("Invalid memory path")
        return target_path
        
    def clear(self, memory_path: str) -> Dict:
        try:
            mem_path = self._resolve_safe_memory_path(memory_path)
            if mem_path.exists():
                shutil.rmtree(mem_path)
            mem_path.mkdir(parents=True, exist_ok=True)
            return {"status": "success", "message": "Memory cleared."}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_context(self, memory_path: str) -> str:
        try:
            safe_memory_path = self._resolve_safe_memory_path(memory_path)
        except Exception as e:
            self._log(f"Invalid memory path: {e}")
            return "خطا در خواندن تاریخچه."

        history_file = safe_memory_path / "conversation_history.json"
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
        try:
            safe_memory_path = self._resolve_safe_memory_path(memory_path)
        except Exception as e:
            self._log(f"Invalid memory path: {e}")
            return

        history_file = safe_memory_path / "conversation_history.json"
        history = []
        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except: pass
        
        interaction = {"timestamp": datetime.now().isoformat(), "question": question, "answer": answer}
        history.append(interaction)
        if len(history) > 50: history = history[-50:]
            
        try:
            safe_memory_path.mkdir(parents=True, exist_ok=True)
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
                        # دانلود و استخراج متن کامل
                        downloaded = trafilatura.fetch_url(entry.link)
                        if downloaded:
                            full_text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
                            if full_text and full_text.strip():
                                new_entries.append({
                                    'title': getattr(entry, 'title', 'No title'),
                                    'link': getattr(entry, 'link', ''),
                                    'pubDate': pubdate,
                                    'content': full_text.strip(), # متن کامل
                                    'feed_url': feed_url, # برای ردیابی منبع
                                    'source_type': 'rss_auto' # متادیتا برای تشخیص نوع داده
                                })
                                self._log(f"Extracted full text for: {entry.title}")
                            else:
                                self._log(f"No content extracted from {entry.link}")
                        else:
                            self._log(f"Could not download {entry.link}")
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
    
    def get_answer(self, question: str, vector_store_path: str, memory_path: str) -> Generator[str, None, None]:

        t_total_start = time.time()

        t0 = time.time()
        vectorstore = self.vstore_manager.load(vector_store_path)
        print(f"Load VectorStore: {time.time() - t0:.4f}s")

        if not vectorstore:
            yield "هنوز دانشی برای شما ذخیره نشده است."
            return

        t1 = time.time()
        memory_context = ""
        if self.config.get('ENABLE_MEMORY', True):
            memory_context = self.memory_manager.get_context(memory_path)

        # Retrieve relevant documents
        retriever = vectorstore.as_retriever(search_type=self.config['SEARCH_TYPE'], search_kwargs={"k": self.config.get('TOP_K_RESULTS', 2)})
        docs = retriever.get_relevant_documents(question)
        print(f"Retrieved {len(docs)} documents for question: '{question}'")
        for i, doc in enumerate(docs):
            print(f"Doc {i+1}: {doc.page_content[:200]}...")

        # Prepare prompt for streaming (bypass StrOutputParser which buffers output)
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
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
        
        # Format prompt with retrieved context
        context = format_docs(docs)
        formatted_prompt = prompt.format(
            context=context,
            question=question,
            memory_context=memory_context
        )
        
        print(f"Prepare Chain & Memory: {time.time() - t1:.4f}s")

        full_answer = ""
        t_gen_start = time.time()
        first_token = True
        chunk_count = 0

        print(f"Starting to stream response for question: {question}", flush=True)
        # Stream directly from LLM to get token-by-token streaming
        try:
            import sys
            for chunk in self.llm.stream(formatted_prompt):
                chunk_count += 1
                
                # Handle different chunk types from LangChain
                if hasattr(chunk, 'content'):
                    chunk_text = chunk.content
                elif isinstance(chunk, str):
                    chunk_text = chunk
                elif hasattr(chunk, 'text'):
                    chunk_text = chunk.text
                else:
                    chunk_text = str(chunk)
                
                # Skip empty chunks
                if not chunk_text:
                    continue
                
                # Only log first few chunks and every 50th chunk to reduce overhead
                if chunk_count <= 3 or chunk_count % 50 == 0:
                    print(f"Chunk {chunk_count}: '{chunk_text[:100]}...' (len={len(chunk_text)})", flush=True)
                if first_token:
                    print(f"Time To First Token (Latency): {time.time() - t_gen_start:.4f}s", flush=True)
                    first_token = False

                full_answer += chunk_text
                # Yield immediately without any delay
                yield chunk_text
                sys.stdout.flush()  # Force flush to ensure logs are written immediately
                
        except Exception as stream_error:
            print(f"Error during streaming: {stream_error}")
            import traceback
            traceback.print_exc()
            # Fallback: try using invoke and yield character by character
            print("Falling back to non-streaming mode...")
            try:
                response = self.llm.invoke(formatted_prompt)
                response_text = response.content if hasattr(response, 'content') else str(response)
                if response_text:
                    # Yield in small chunks for better UX even in fallback mode
                    chunk_size = 10
                    for i in range(0, len(response_text), chunk_size):
                        chunk = response_text[i:i+chunk_size]
                        full_answer += chunk
                        yield chunk
            except Exception as fallback_error:
                print(f"Fallback also failed: {fallback_error}")
                traceback.print_exc()
                yield "خطا در تولید پاسخ."

        print(f"Total chunks: {chunk_count}")
        print(f"Total Text Generation Time: {time.time() - t_gen_start:.4f}s")
        print(f"Full answer: '{full_answer}'")

        if self.config.get('ENABLE_MEMORY', True):
            self.memory_manager.add_interaction(memory_path, question, full_answer.strip())

        print(f"Total Process Time: {time.time() - t_total_start:.4f}s")
        