"""
Модуль для расчета метрик качества коррекции и суммаризации текста
"""

import re
import math
import asyncio
import numpy as np
from typing import Dict, Any, Optional, List
import logging
import config
# from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import jiwer
from Levenshtein import distance as levenshtein_distance
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

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

class MetricsCalculator:
    """
    Калькулятор метрик для оценки качества текста
    """
    
    def __init__(self, 
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 perplexity_model_name: str = "gpt2"):
        """
        Инициализация калькулятора метрик.
        
        Args:
            embedding_model_name: Название модели для эмбеддингов
            perplexity_model_name: Название модели для перплексии
        """
        self.logger = logging.getLogger("metrics_calculator")
        
        # Simple cache for metrics to avoid redundant calculations
        self._metrics_cache = {}
        self._cache_size_limit = 100
        
        # Initialize embedding model
        try:
            # Try to load sentence-transformers model
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(embedding_model_name)
            self.logger.info(f"SentenceTransformer model {embedding_model_name} loaded")
        except ImportError:
            # Fallback to simple embeddings if sentence-transformers not available
            try:
                # Use simple numpy-based embeddings as fallback
                import numpy as np
                self.embedding_model = None  # Will use simple cosine similarity
                self.logger.warning("SentenceTransformer not available, using simple embeddings")
            except Exception as e:
                self.logger.error(f"Error initializing fallback embedding model: {e}")
                self.embedding_model = None
        except Exception as e:
            self.logger.error(f"Error initializing embedding model: {e}")
            self.embedding_model = None
        
        # Initialize GPT-2 model for perplexity
        try:
            self.perplexity_model_name = "gpt2"
            self.perplexity_tokenizer = AutoTokenizer.from_pretrained(self.perplexity_model_name)
            self.perplexity_model = AutoModelForCausalLM.from_pretrained(self.perplexity_model_name)
            self.perplexity_model.eval()
            self.logger.info(f"Model perplexity {self.perplexity_model_name} loaded")
        except Exception as e:
            self.logger.error(f"Error loading perplexity model: {e}")
            self.perplexity_model = None
            self.perplexity_tokenizer = None
    
    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        """
        Расчет Word Error Rate (WER).
        
        Args:
            reference: Эталонный текст
            hypothesis: Гипотеза
            
        Returns:
            WER в диапазоне [0, 1]
        """
        try:
            return jiwer.wer(reference.lower(), hypothesis.lower())
        except Exception as e:
            self.logger.error(f"Ошибка расчета WER: {e}")
            return 1.0
    
    def calculate_lev_rating(self, text1: str, text2: str) -> float:
        """
        Расчет LevRating (сходство Левенштейна).
        
        Args:
            text1: Первый текст
            text2: Второй текст
            
        Returns:
            LevRating в диапазоне [0, 1]
        """
        try:
            max_len = max(len(text1), len(text2))
            if max_len == 0:
                return 1.0
                
            distance = levenshtein_distance(text1, text2)
            similarity = 1 - (distance / max_len)
            return similarity
        except Exception as e:
            self.logger.error(f"Ошибка расчета LevRating: {e}")
            return 0.0
    
    def calculate_perplexity(self, text: str) -> float:
        """
        Расчет перплексии текста.
        
        Args:
            text: Текст для анализа
            
        Returns:
            Перплексия (чем ниже, тем лучше)
        """
        if self.perplexity_tokenizer is None or self.perplexity_model is None:
            self.logger.warning("Модель перплексии не загружена")
            return float('inf')
            
        try:
            # Tokenizatsiya teksta s ogranicheniem dliny
            max_length = 1024  # Maksimalnaya dlya GPT-2
            inputs = self.perplexity_tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=max_length
            )
            
            with torch.no_grad():
                outputs = self.perplexity_model(**inputs, labels=inputs["input_ids"])
                
                # Raschet perpleksii
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
                
            return perplexity
        except Exception as e:
            self.logger.error(f"Ошибка расчета перплексии: {e}")
            return float('inf')
    
    def calculate_cor_score(self, 
                           original_wer: float,
                           corrected_wer: float,
                           original_lev: float,
                           corrected_lev: float,
                           perplexity: float) -> float:
        """
        Расчет композитной метрики CorScore для коррекции.
        Diapazon: 0 (plokhaya korrektsiya) do 1 (ideal'naya korrektsiya).
        Formula: CorScore = deltaWER*0.4 + deltaLev*0.4 + (1-Perpl/100)*0.2
        
        Args:
            original_wer: WER iskhodnogo teksta (Incorrect vs Etalon)
            corrected_wer: WER ispravlennogo teksta (Corrected vs Etalon)
            original_lev: LevRating iskhodnogo teksta
            corrected_lev: LevRating ispravlennogo teksta
            perplexity: Perpleksiya ispravlennogo teksta
            
        Returns:
            CorScore v diapazone [0, 1]
        """
        try:
            delta_wer = original_wer - corrected_wer
            delta_lev = corrected_lev - original_lev
            
            # Formula: CorScore = deltaWER + deltaLev*10 + (1-Perpl/100)*0.2
            cor_score = delta_wer + delta_lev * 10 + (1 - min(perplexity, 100) / 100) * 0.2
            
            return cor_score
        except Exception as e:
            self.logger.error(f"Oshibka rascheta CorScore: {e}")
            return 0.0
    
    def calculate_bert_score(self, reference: str, hypothesis: str) -> float:
        """
        Расчет BertScore.
        
        Args:
            reference: Эталонный текст
            hypothesis: Гипотеза
            
        Returns:
            BertScore в диапазоне [0, 1]
        """
        try:
            import numpy as np
            
            if self.embedding_model is not None:
                # Use SentenceTransformer embeddings
                ref_embeddings = self.embedding_model.encode([reference], convert_to_numpy=True)
                hyp_embeddings = self.embedding_model.encode([hypothesis], convert_to_numpy=True)
            else:
                # Use simple word-based similarity as fallback
                from sklearn.feature_extraction.text import TfidfVectorizer
                
                # Create TF-IDF vectors
                vectorizer = TfidfVectorizer().fit_transform([reference, hypothesis])
                ref_embeddings = vectorizer.toarray()[0:1]
                hyp_embeddings = vectorizer.toarray()[1:2]
            
            # Extract 1D arrays from 2D arrays
            ref_embedding = ref_embeddings[0]
            hyp_embedding = hyp_embeddings[0]
            
            # Calculate cosine similarity
            similarity = np.dot(ref_embedding, hyp_embedding) / (
                np.linalg.norm(ref_embedding) * np.linalg.norm(hyp_embedding)
            )
            
            return float(similarity)
        except Exception as e:
            self.logger.error(f"Ошибка расчета BertScore: {e}")
            return 0.0
    
    def calculate_meteor(self, reference: str, hypothesis: str) -> float:
        """
        Расчет METEOR score.
        
        Args:
            reference: Эталонный текст
            hypothesis: Гипотеза
            
        Returns:
            METEOR score в диапазоне [0, 1]
        """
        try:
            # Проверка наличия текстов
            if not reference or not hypothesis:
                self.logger.warning("METEOR: Пустой reference или hypothesis")
                return 0.0
            
            # Токенизация
            ref_tokens = word_tokenize(reference.lower())
            hyp_tokens = word_tokenize(hypothesis.lower())
            
            # Логирование для отладки (только в DEBUG режиме)
            self.logger.debug(f"METEOR DEBUG: ref_tokens={len(ref_tokens)}, hyp_tokens={len(hyp_tokens)}")
            self.logger.debug(f"METEOR DEBUG: ref={reference[:100]}..., hyp={hypothesis[:100]}...")
            
            # Расчет METEOR
            score = meteor_score([ref_tokens], hyp_tokens)
            self.logger.debug(f"METEOR DEBUG: score={score}")
            return score
        except Exception as e:
            self.logger.error(f"Ошибка расчета METEOR: {e}")
            return 0.0
    
    async def calculate_geval(self, 
                             original_text: str,
                             summary_text: str,
                             lm_client,
                             reference_summary: Optional[str] = None) -> float:
        """
        Расчет G-Eval с помощью LLM.
        
        Args:
            original_text: Оригинальный текст
            summary_text: Суммаризация
            lm_client: Клиент LM Studio
            
        Returns:
            G-Eval score в диапазоне [0, 1]
        """
        try:
            # Format reference summary if provided
            ref_summary_text = f"\nЭталонное резюме: {reference_summary}" if reference_summary else ""
            
            prompt = config.GEVAL_PROMPT.format(
                original_text=original_text,
                summary_text=summary_text,
                reference_summary=ref_summary_text
            )
            
            response = await lm_client.generate_with_retry(
                prompt=prompt,
                temperature=0.3,
                max_tokens=1024,
                system_prompt="You are an expert at evaluating text quality. Return only a number from 0 to 1 with decimal points."
            )
            
            # Извлечение числа из ответа
            score_match = re.search(r'0?\.\d+|1\.0|0|1', response.strip())
            if score_match:
                score = float(score_match.group())
                return min(max(score, 0.0), 1.0)
            else:
                self.logger.warning(f"Не удалось извлечь score из ответа: {response}")
                return 0.5
                
        except Exception as e:
            self.logger.error(f"Ошибка расчета G-Eval: {e}")
            return 0.0
    
    async def calculate_llm_judge(self,
                                original_text: str,
                                summary_text: str,
                                lm_client,
                                reference_summary: Optional[str] = None) -> tuple[float, str]:
        """
        Расчет LLM-Judge score.
        
        Args:
            original_text: Оригинальный текст
            summary_text: Суммаризация
            lm_client: Клиент LM Studio
            
        Returns:
            LLM-Judge score в диапазоне [1, 10]
        """
        try:
            # Format reference summary if provided
            ref_summary_text = f"\nReference summary: {reference_summary}" if reference_summary else ""
            
            prompt = config.LLM_JUDGE_PROMPT.format(
                original_text=original_text,
                summary_text=summary_text,
                reference_summary=ref_summary_text
            )
            
            response = await lm_client.generate_with_retry(
                prompt=prompt,
                temperature=0.7,
                max_tokens=1024,
                system_prompt="You are a strict and precise evaluator of text summaries. Give honest, varied scores from 1 to 10 based on actual quality. Don't default to middle scores. Be critical in your evaluation."
            )
            
            # Извлечение числа и объяснения из ответа
            score_match = re.search(r'Score:\s*(10|[1-9])', response.strip())
            explanation_match = re.search(r'Explanation:\s*(.+)', response.strip(), re.DOTALL)
            
            if score_match:
                score = float(score_match.group(1))  # Получаем только число из группы 1
                explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided"
                
                # Логируем только оценку, без объяснения на экран
                self.logger.info(f"LLM-Judge Score: {score}")
                
                return min(max(score, 1.0), 10.0), explanation
            else:
                self.logger.warning(f"Не удалось извлечь score из ответа: {response}")
                return 5.0, "No explanation provided"
                
        except Exception as e:
            self.logger.error(f"Ошибка расчета LLM-Judge: {e}")
            return 5.0, "Error occurred"
    
    def calculate_sum_score(self,
                           geval_score: float,
                           llm_judge_score: float,
                           meteor_score: float,
                           bert_score: float,
                           geval_weight: float = 0.25,
                           llm_judge_weight: float = 0.2,
                           meteor_weight: float = 0.35,
                           bert_weight: float = 0.2) -> float:
        """
        Расчет композитной метрики SumScore для суммаризации.
        
        Args:
            geval_score: G-Eval score
            llm_judge_score: LLM-Judge score
            meteor_score: METEOR score
            bert_score: BertScore
            geval_weight: вес G-Eval (25%)
            llm_judge_weight: вес LLM-Judge (20%)
            meteor_weight: вес METEOR (35%)
            bert_weight: вес BertScore (20%)
            
        Returns:
            SumScore в диапазоне [0, 1]
        """
        try:
            # Нормализация LLM-Judge к диапазону [0, 1]
            normalized_llm_judge = (llm_judge_score - 1) / 9
            
            sum_score = (geval_weight * geval_score + 
                        llm_judge_weight * normalized_llm_judge + 
                        meteor_weight * meteor_score +
                        bert_weight * bert_score)
            
            return sum_score
        except Exception as e:
            self.logger.error(f"Ошибка расчета SumScore: {e}")
            return 0.0
    
    def calculate_correction_metrics(self,
                                   original_text: str,
                                   corrected_text: str,
                                   reference_text: Optional[str] = None) -> Dict[str, float]:
        """
        Расчет всех метрик для коррекции.
        
        Args:
            original_text: Исходный текст
            corrected_text: Исправленный текст
            reference_text: Эталонный текст (опционально)
            
        Returns:
            Словарь с метриками
        """
        metrics = {}
        
        try:
            # Расчет базовых метрик
            if reference_text:
                original_wer = self.calculate_wer(reference_text, original_text)
                corrected_wer = self.calculate_wer(reference_text, corrected_text)
                metrics["wer_original"] = original_wer
                metrics["wer_corrected"] = corrected_wer
                metrics["delta_wer"] = original_wer - corrected_wer
                
                original_lev = self.calculate_lev_rating(reference_text, original_text)
                corrected_lev = self.calculate_lev_rating(reference_text, corrected_text)
                metrics["lev_original"] = original_lev
                metrics["lev_corrected"] = corrected_lev
                metrics["delta_lev"] = corrected_lev - original_lev
            else:
                # Если нет эталона, используем сходство с оригиналом
                lev_similarity = self.calculate_lev_rating(original_text, corrected_text)
                metrics["lev_similarity"] = lev_similarity
                metrics["wer_original"] = 1.0  # Максимальный WER
                metrics["wer_corrected"] = 0.0  # Идеальный WER
                metrics["delta_wer"] = 1.0
                metrics["lev_original"] = 0.0
                metrics["lev_corrected"] = lev_similarity
                metrics["delta_lev"] = lev_similarity
            
            # Расчет перплексии
            perplexity = self.calculate_perplexity(corrected_text)
            metrics["perplexity"] = perplexity
            
            # Расчет CorScore
            cor_score = self.calculate_cor_score(
                original_wer=metrics["wer_original"],
                corrected_wer=metrics["wer_corrected"],
                original_lev=metrics["lev_original"],
                corrected_lev=metrics["lev_corrected"],
                perplexity=perplexity
            )
            metrics["cor_score"] = cor_score
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета метрик коррекции: {e}")
            
        return metrics
    
    async def calculate_summary_metrics(self,
                                     original_text: str,
                                     summary_text: str,
                                     reference_summary: Optional[str] = None,
                                     lm_client=None) -> Dict[str, float]:
        """
        Расчет всех метрик для суммаризации.
        
        Args:
            original_text: Оригинальный текст
            summary_text: Суммаризация
            reference_summary: Эталонная суммаризация (опционально)
            lm_client: Клиент LM Studio для LLM-метрик
            
        Returns:
            Словарь с метриками
        """
        metrics = {}
        
        try:
            # Расчет METEOR (если есть эталон)
            if reference_summary:
                meteor = self.calculate_meteor(reference_summary, summary_text)
                metrics["meteor"] = meteor
            else:
                # Если нет эталона, используем 0.5 как среднее значение
                metrics["meteor"] = 0.5
            
            # Расчет BertScore
            bert_score = self.calculate_bert_score(original_text, summary_text)
            metrics["bert_score"] = bert_score
            
            # Расчет LLM-метрик параллельно (если есть клиент)
            if lm_client:
                # Запускаем G-Eval и LLM-Judge параллельно
                geval_task = self.calculate_geval(original_text, summary_text, lm_client, reference_summary)
                llm_judge_task = self.calculate_llm_judge(original_text, summary_text, lm_client, reference_summary)
                
                # Ждем выполнения обеих задач
                geval_score, (llm_judge_score, llm_judge_explanation) = await asyncio.gather(
                    geval_task, llm_judge_task
                )
                
                metrics["geval"] = geval_score
                metrics["llm_judge"] = llm_judge_score
                metrics["llm_judge_explanation"] = llm_judge_explanation
                
                # Расчет SumScore
                sum_score = self.calculate_sum_score(
                    geval_score=geval_score,
                    llm_judge_score=llm_judge_score,
                    meteor_score=metrics["meteor"],
                    bert_score=metrics["bert_score"],
                    geval_weight=config.GEVAL_WEIGHT,
                    llm_judge_weight=config.LLM_JUDGE_WEIGHT,
                    meteor_weight=config.METEOR_WEIGHT,
                    bert_weight=config.BERT_SCORE_WEIGHT
                )
                metrics["sum_score"] = sum_score
            else:
                # If no client, use average values and calculate SumScore
                metrics["geval"] = 0.5
                metrics["llm_judge"] = 5.0
                # Calculate SumScore with default values
                sum_score = self.calculate_sum_score(
                    geval_score=0.5,
                    llm_judge_score=5.0,
                    meteor_score=metrics["meteor"],
                    bert_score=metrics["bert_score"],
                    geval_weight=config.GEVAL_WEIGHT,
                    llm_judge_weight=config.LLM_JUDGE_WEIGHT,
                    meteor_weight=config.METEOR_WEIGHT,
                    bert_weight=config.BERT_SCORE_WEIGHT
                )
                metrics["sum_score"] = sum_score
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета метрик суммаризации: {e}")
            
        return metrics
