"""
Система памяти агентов для хранения успешных примеров и семантического поиска
"""

import json
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
# from sentence_transformers import SentenceTransformer
import logging

class SimpleEmbeddingModel:
    """Simple embedding model using numpy to avoid dependency conflicts"""
    
    def __init__(self):
        pass
    
    def encode(self, texts, **kwargs):
        """Simple encoding using character n-grams"""
        import re
        
        # Handle single text case
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            # Simple character-based embedding with fixed size
            embedding = np.zeros(512)  # Fixed size embedding for all texts
            
            # Use first 512 characters or pad with zeros
            text_lower = text.lower()
            for i, char in enumerate(text_lower[:512]):
                embedding[i] = ord(char) / 255.0
            
            embeddings.append(embedding)
        
        result = np.array(embeddings)
        
        # Handle convert_to_numpy parameter
        if kwargs.get('convert_to_numpy', True):
            return result
        else:
            return result.tolist()
    
    def similarity(self, embeddings1, embeddings2):
        """Calculate cosine similarity"""
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity([embeddings1], [embeddings2])[0][0]

    def _get_embedding(self, text: str):
        """Get embedding for text using SimpleEmbeddingModel"""
        if self.embedding_model is None:
            return None
        embedding = self.embedding_model.encode([text])
        return embedding[0] if len(embedding) > 0 else None
    
    def _cosine_similarity(self, embeddings1, embeddings2):
        """Calculate cosine similarity"""
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity([embeddings1], [embeddings2])[0][0]

class AgentMemory:
    """
    Система памяти для хранения и поиска успешных примеров коррекции и суммаризации
    """
    
    def __init__(self, 
                 memory_dir: Path,
                 model_name: str = "all-MiniLM-L6-v2",
                 cache_enabled: bool = True,
                 cache_max_size: int = 100):
        """
        Инициализация системы памяти.
        
        Args:
            memory_dir: Директория для хранения файлов памяти
            model_name: Название модели для эмбеддингов
            cache_enabled: Включить кэширование промптов
            cache_max_size: Максимальный размер кэша
        """
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.cache_enabled = cache_enabled
        self.cache_max_size = cache_max_size
        
        # Файлы памяти
        self.correction_history_file = self.memory_dir / "correction_history.json"
        self.summary_history_file = self.memory_dir / "summary_history.json"
        self.few_shot_file = self.memory_dir / "few_shot_examples.json"
        self.summary_few_shot_file = self.memory_dir / "summary_few_shot_examples.json"
        self.best_prompts_file = self.memory_dir / "best_prompts.json"
        self.prompt_cache_file = self.memory_dir / "prompt_cache.json"
        self.embeddings_file = self.memory_dir / "embeddings.pkl"
        
        self.logger = logging.getLogger("agent_memory")
        
        # Загрузка данных
        self.correction_history = self._load_json(self.correction_history_file, [])
        self.summary_history = self._load_json(self.summary_history_file, [])
        self.few_shot_examples = self._load_json(self.few_shot_file, [])
        self.summary_few_shot_examples = self._load_json(self.summary_few_shot_file, [])
        self.best_prompts = self._load_json(self.best_prompts_file, {})
        self.prompt_cache = self._load_json(self.prompt_cache_file, {})
        
        # Инициализация модели эмбеддингов
        self._init_embedding_model()
        
    def _init_embedding_model(self):
        """Initialization of the embedding model"""
        try:
            # Simple numpy-based embedding to avoid dependency conflicts
            self.embedding_model = SimpleEmbeddingModel()
            self.logger.info(f"Simple embedding model loaded (replacing SentenceTransformer)")
        except Exception as e:
            self.logger.error(f"Error initializing embedding model: {e}")
            self.embedding_model = None
            
    def _load_json(self, file_path: Path, default: Any) -> Any:
        """Загрузка JSON файла"""
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Ошибка загрузки {file_path}: {e}")
        return default
        
    def _save_json(self, data: Any, file_path: Path):
        """Сохранение в JSON файл"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Ошибка сохранения {file_path}: {e}")
            
    def _get_text_hash(self, text: str) -> str:
        """Получение хэша текста"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
        
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Получение эмбеддинга текста"""
        if self.embedding_model is None:
            return None
        try:
            return self.embedding_model.encode(text, convert_to_numpy=True)
        except Exception as e:
            self.logger.error(f"Ошибка создания эмбеддинга: {e}")
            return None
            
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Вычисление косинусного сходства"""
        try:
            return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        except:
            return 0.0
            
    def learn_from_correction(self, 
                            original_text: str,
                            corrected_text: str,
                            reference_text: Optional[str],
                            metrics: Dict[str, float],
                            prompt: str,
                            model: str,
                            domain: str = "general"):
        """
        Сохранение успешной коррекции в память.
        
        Args:
            original_text: Исходный текст
            corrected_text: Исправленный текст
            reference_text: Эталонный текст (опционально)
            metrics: Метрики качества
            prompt: Использованный промпт
            model: Модель
            domain: Домен
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "original_text": original_text,
            "corrected_text": corrected_text,
            "reference_text": reference_text,
            "metrics": metrics,
            "prompt": prompt,
            "model": model,
            "domain": domain,
            "text_hash": self._get_text_hash(original_text)
        }
        
        self.correction_history.append(record)
        
        # Добавляем в few-shot примеры если качество хорошее
        if metrics.get("cor_score", 0) > 0.5:
            self.few_shot_examples.append({
                "input": original_text,
                "output": corrected_text,
                "domain": domain,
                "score": metrics.get("cor_score", 0)
            })
            
            # Ограничиваем количество примеров
            if len(self.few_shot_examples) > 100:
                self.few_shot_examples = sorted(self.few_shot_examples, 
                                              key=lambda x: x["score"], 
                                              reverse=True)[:100]
        
        # Сохраняем лучший промпт для домена
        if domain not in self.best_prompts:
            self.best_prompts[domain] = []
            
        prompt_record = {
            "prompt": prompt,
            "score": metrics.get("cor_score", 0),
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }
        
        self.best_prompts[domain].append(prompt_record)
        
        # Ограничиваем количество промптов
        if len(self.best_prompts[domain]) > 10:
            self.best_prompts[domain] = sorted(self.best_prompts[domain], 
                                              key=lambda x: x["score"], 
                                              reverse=True)[:10]
        
        # Сохранение на диск
        self._save_json(self.correction_history, self.correction_history_file)
        self._save_json(self.few_shot_examples, self.few_shot_file)
        self._save_json(self.best_prompts, self.best_prompts_file)
        
    def learn_from_summarization(self,
                               original_text: str,
                               summary_text: str,
                               reference_summary: Optional[str],
                               metrics: Dict[str, float],
                               prompt: str,
                               model: str,
                               domain: str = "general"):
        """
        Сохранение успешной суммаризации в память.
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "original_text": original_text,
            "summary_text": summary_text,
            "reference_summary": reference_summary,
            "metrics": metrics,
            "prompt": prompt,
            "model": model,
            "domain": domain,
            "text_hash": self._get_text_hash(original_text)
        }
        
        self.summary_history.append(record)
        
        # Добавляем в few-shot примеры если качество хорошее
        if metrics.get("sum_score", 0) > 0.6:
            self.summary_few_shot_examples.append({
                "input": original_text,
                "output": summary_text,
                "domain": domain,
                "score": metrics.get("sum_score", 0)
            })
            
            # Ограничиваем количество примеров
            if len(self.summary_few_shot_examples) > 100:
                self.summary_few_shot_examples = sorted(self.summary_few_shot_examples, 
                                                       key=lambda x: x["score"], 
                                                       reverse=True)[:100]
        
        # Сохранение на диск
        self._save_json(self.summary_history, self.summary_history_file)
        self._save_json(self.summary_few_shot_examples, self.summary_few_shot_file)
        
    def get_few_shot_examples(self, 
                            input_text: str,
                            domain: str = "general",
                            max_examples: int = 3,
                            similarity_threshold: float = 0.6) -> List[Dict[str, str]]:
        """
        Получение релевантных few-shot примеров для коррекции.
        
        Args:
            input_text: Входной текст
            domain: Домен
            max_examples: Максимальное количество примеров
            similarity_threshold: Порог сходства
            
        Returns:
            Список релевантных примеров
        """
        if not self.few_shot_examples or self.embedding_model is None:
            return []
            
        # Фильтруем по домену
        domain_examples = [ex for ex in self.few_shot_examples 
                          if ex.get("domain") == domain or ex.get("domain") == "general"]
        
        if not domain_examples:
            return []
            
        # Получаем эмбеддинг входного текста
        input_embedding = self._get_embedding(input_text)
        if input_embedding is None:
            return []
            
        # Вычисляем сходство с примерами
        similarities = []
        for example in domain_examples:
            example_embedding = self._get_embedding(example["input"])
            if example_embedding is not None:
                similarity = self._cosine_similarity(input_embedding, example_embedding)
                similarities.append((example, similarity))
                
        # Сортируем по сходству и фильтруем по порогу
        similarities.sort(key=lambda x: x[1], reverse=True)
        relevant_examples = [(ex, sim) for ex, sim in similarities 
                            if sim >= similarity_threshold]
        
        # Возвращаем лучшие примеры
        return [ex for ex, _ in relevant_examples[:max_examples]]
        
    def get_summary_few_shot_examples(self,
                                     input_text: str,
                                     domain: str = "general",
                                     max_examples: int = 3,
                                     similarity_threshold: float = 0.6) -> List[Dict[str, str]]:
        """
        Получение релевантных few-shot примеров для суммаризации.
        """
        if not self.summary_few_shot_examples or self.embedding_model is None:
            return []
            
        # Фильтруем по домену
        domain_examples = [ex for ex in self.summary_few_shot_examples 
                          if ex.get("domain") == domain or ex.get("domain") == "general"]
        
        if not domain_examples:
            return []
            
        # Получаем эмбеддинг входного текста
        input_embedding = self._get_embedding(input_text)
        if input_embedding is None:
            return []
            
        # Вычисляем сходство с примерами
        similarities = []
        for example in domain_examples:
            example_embedding = self._get_embedding(example["input"])
            if example_embedding is not None:
                similarity = self._cosine_similarity(input_embedding, example_embedding)
                similarities.append((example, similarity))
                
        # Сортируем по сходству и фильтруем по порогу
        similarities.sort(key=lambda x: x[1], reverse=True)
        relevant_examples = [(ex, sim) for ex, sim in similarities 
                            if sim >= similarity_threshold]
        
        # Возвращаем лучшие примеры
        return [ex for ex, _ in relevant_examples[:max_examples]]
        
    def get_best_prompt_for_domain(self, domain: str = "general") -> Optional[str]:
        """
        Получение лучшего промпта для домена.
        
        Args:
            domain: Домен
            
        Returns:
            Лучший промпт или None
        """
        if domain in self.best_prompts and self.best_prompts[domain]:
            # Возвращаем промпт с наивысшим score
            best_prompt_record = max(self.best_prompts[domain], key=lambda x: x["score"])
            return best_prompt_record["prompt"]
        return None
        
    def cache_prompt(self, prompt: str, metrics: Dict[str, float], improvement: float):
        """
        Кэширование промпта с метриками.
        
        Args:
            prompt: Промпт
            metrics: Метрики
            improvement: Улучшение
        """
        if not self.cache_enabled:
            return
            
        prompt_hash = self._get_text_hash(prompt)
        cache_record = {
            "prompt": prompt,
            "metrics": metrics,
            "improvement": improvement,
            "timestamp": datetime.now().isoformat(),
            "last_used": datetime.now().isoformat()
        }
        
        self.prompt_cache[prompt_hash] = cache_record
        
        # Ограничиваем размер кэша
        if len(self.prompt_cache) > self.cache_max_size:
            # Удаляем самые старые записи с наименьшим улучшением
            cache_items = list(self.prompt_cache.items())
            cache_items.sort(key=lambda x: (x[1]["improvement"], x[1]["timestamp"]))
            
            # Удаляем 10% записей
            to_remove = len(cache_items) // 10
            for i in range(to_remove):
                del self.prompt_cache[cache_items[i][0]]
                
        self._save_json(self.prompt_cache, self.prompt_cache_file)
        
    def get_cached_prompt(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Получение кэшированного промпта.
        
        Args:
            prompt: Промпт
            
        Returns:
            Кэшированная запись или None
        """
        if not self.cache_enabled:
            return None
            
        prompt_hash = self._get_text_hash(prompt)
        if prompt_hash in self.prompt_cache:
            # Обновляем время последнего использования
            self.prompt_cache[prompt_hash]["last_used"] = datetime.now().isoformat()
            self._save_json(self.prompt_cache, self.prompt_cache_file)
            return self.prompt_cache[prompt_hash]
        return None
    
    def _get_embedding(self, text: str):
        """Get embedding for text using SimpleEmbeddingModel"""
        if self.embedding_model is None:
            return None
        embedding = self.embedding_model.encode([text])
        return embedding[0] if len(embedding) > 0 else None
    
    def _cosine_similarity(self, embeddings1, embeddings2):
        """Calculate cosine similarity"""
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity([embeddings1], [embeddings2])[0][0]
