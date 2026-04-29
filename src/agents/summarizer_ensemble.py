"""
Ансамбль суммаризаторов для генерации нескольких вариантов суммаризации текста
"""

import asyncio
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent
from ..utils.lm_studio_client import LMStudioClient
from ..utils.agent_memory import AgentMemory
from ..metrics.metrics_calculator import MetricsCalculator
import config

class SummarizerEnsemble(BaseAgent):
    """
    Ансамбль суммаризаторов для генерации и оценки вариантов суммаризации текста
    """
    
    def __init__(self, 
                 lm_client: LMStudioClient,
                 memory: AgentMemory,
                 metrics_calculator: MetricsCalculator):
        """
        Инициализация ансамбля суммаризаторов.
        
        Args:
            lm_client: Клиент LM Studio
            memory: Система памяти агентов
            metrics_calculator: Калькулятор метрик
        """
        super().__init__("SummarizerEnsemble")
        self.lm_client = lm_client
        self.memory = memory
        self.metrics_calculator = metrics_calculator
        
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Основной метод выполнения агента.
        
        Args:
            state: Текущее состояние системы
            
        Returns:
            Обновленное состояние системы
        """
        self.log_execution("Начало работы ансамбля суммаризаторов")
        
        # Проверка обязательных ключей
        if not self.validate_state(state, ["corrected_text", "prompt_summary"]):
            return state
            
        input_text = self.get_from_state(state, "corrected_text")
        reference_summary = self.get_from_state(state, "reference_summary")
        prompt_summary = self.get_from_state(state, "prompt_summary")
        domain = self.get_from_state(state, "domain", "general")
        
        # Генерация вариантов суммаризации
        variants_data = await self._generate_summary_variants(
            input_text=input_text,
            prompt_summary=prompt_summary,
            reference_summary=reference_summary
        )
        
        # Оценка вариантов и выбор лучшего
        best_variant = await self._select_best_variant(variants_data, input_text, reference_summary)
        
        # Адаптивные попытки если качество низкое
        if best_variant["metrics"].get("sum_score", 0) < config.SUMMARY_THRESHOLDS["satisfactory"]:
            self.log_execution(f"Низкое качество (SumScore={best_variant['metrics'].get('sum_score', 0):.3f}), "
                             f"запускаем адаптивные попытки")
            best_variant = await self._adaptive_summary_attempts(
                input_text=input_text,
                reference_summary=reference_summary,
                current_best=best_variant
            )
        
        # Сохранение в память
        self._learn_from_summarization(
            input_text=input_text,
            summary_text=best_variant["text"],
            reference_summary=reference_summary,
            metrics=best_variant["metrics"],
            prompt=prompt_summary
        )
        
        # Обновление состояния
        self.add_to_state(state,
                         ensemble_summary_outputs=[v["text"] for v in variants_data],
                         ensemble_summary_prompts=[v["prompt_type"] for v in variants_data],
                         ensemble_summary_temperatures=[v["temperature"] for v in variants_data],
                         ensemble_summary_metrics=[v["metrics"] for v in variants_data],
                         summary=best_variant["text"],
                         best_summary_prompt=prompt_summary,
                         best_summary_prompt_type=best_variant.get("prompt_type", "basic"),
                         best_summary_temperature=best_variant.get("temperature", 0.7),
                         summary_metrics=best_variant["metrics"])
        
        self.log_execution(f"Выбран лучший вариант с SumScore={best_variant['metrics'].get('sum_score', 0):.3f}")
        return state
        
    async def _generate_summary_variants(self, 
                                      input_text: str,
                                      prompt_summary: str,
                                      reference_summary: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Генерация вариантов суммаризации текста.
        
        Args:
            input_text: Входной текст
            prompt_summary: Промпт суммаризации
            
        Returns:
            Список вариантов с метриками
        """
        variants = []
        tasks = []
        
        # 3 Basic prompts with different temperatures
        for i, temp in enumerate(config.SUMMARY_TEMPERATURES):
            task = self._generate_single_summary(
                input_text=input_text,
                prompt=prompt_summary,
                temperature=temp,
                prompt_type="basic",
                reference_summary=reference_summary
            )
            tasks.append(task)
        
        # 1 Best saved prompt from memory
        saved_prompts = self._get_saved_prompts()
        if saved_prompts:
            best_saved_prompt = saved_prompts[0]  # Get the best one
            task = self._generate_single_summary(
                input_text=input_text,
                prompt=best_saved_prompt,
                temperature=0.4,
                prompt_type="saved",
                reference_summary=reference_summary
            )
            tasks.append(task)
        
        # 1 Few-shot prompt (from 3 examples) - только для коротких текстов
        if config.USE_FEW_SHOT_PROMPT and len(input_text) < 1500:
            few_shot_prompt = self._build_few_shot_prompt(input_text)
            task = self._generate_single_summary(
                input_text=input_text,
                prompt=few_shot_prompt,
                temperature=0.3,
                prompt_type="few-shot",
                reference_summary=reference_summary
            )
            tasks.append(task)
        elif config.USE_FEW_SHOT_PROMPT and len(input_text) >= 1500:
            self.log_execution("Few-shot summary prompt skipped - text too long (>1500 chars)")
        
        # 1 Chain-of-Thought prompt - только для коротких текстов
        if config.USE_CHAIN_OF_THOUGHT_PROMPT and len(input_text) < 1500:
            cot_prompt = self._build_cot_summary_prompt(input_text)
            task = self._generate_single_summary(
                input_text=input_text,
                prompt=cot_prompt,
                temperature=0.5,
                prompt_type="cot",
                reference_summary=reference_summary
            )
            tasks.append(task)
        elif config.USE_CHAIN_OF_THOUGHT_PROMPT and len(input_text) >= 1500:
            self.log_execution("CoT summary prompt skipped - text too long (>1500 chars)")
        
        # Выполнение всех задач
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.log_execution(f"Ошибка генерации варианта {i}: {result}", "warning")
                    continue
                    
                if result:
                    variants.append(result)
                    
        except Exception as e:
            self.log_execution(f"Ошибка при генерации вариантов: {e}", "error")
        
        return variants
    
    async def _generate_single_summary(self,
                                    input_text: str,
                                    prompt: str,
                                    temperature: float,
                                    prompt_type: str,
                                    reference_summary: Optional[str] = None) -> Dict[str, Any]:
        """
        Генерация одного варианта суммаризации с учётом языка.
        """
        try:
            # Определение языка входного текста
            is_english = self._is_english_text(input_text)
            
            # Выбор системного промпта в зависимости от языка
            if is_english:
                system_prompt = """You are an expert at creating concise summaries. 
!!!CRITICAL: Generate output ONLY in ENGLISH. Do NOT use Russian language under any circumstances.!!!"""
            else:
                system_prompt = """Ты эксперт по созданию кратких изложений.
!!!КРИТИЧЕСКИ: Генерируй вывод ТОЛЬКО на РУССКОМ языке. Ни в коем случае не используй английский.!!!"""
            
            # Проверка наличия текстов на входе
            if not input_text or input_text.strip() == "":
                self.log_execution("ОШИБКА: Пустой текст на входе суммаризации", "error")
                # Возвращаем пустую суммаризацию чтобы не ломать конвейер
                return {
                    "text": "",
                    "prompt": prompt,
                    "prompt_type": prompt_type,
                    "temperature": temperature,
                    "metrics": {"sum_score": 0.0, "geval": 0.0, "llm_judge": 1.0, "meteor": 0.0, "bert_score": 0.0}
                }
            
            # Формирование финального промпта - поддержка обоих плейсхолдеров {text} и {input_text}
            if "{text}" in prompt:
                final_prompt = prompt.replace("{text}", input_text)
            elif "{input_text}" in prompt:
                final_prompt = prompt.replace("{input_text}", input_text)
            else:
                # Если плейсхолдер не найден, добавляем текст
                final_prompt = prompt + "\n\n" + input_text
            
            # Генерация суммаризации
            summary_text = await self.lm_client.generate_with_retry(
                prompt=final_prompt,
                temperature=temperature,
                max_tokens=min(1024, len(input_text) // 2),
                system_prompt=system_prompt
            )
            
            # Проверка наличия текста на выходе
            if not summary_text or summary_text.strip() == "":
                self.log_execution("ОШИБКА: Пустой текст на выходе суммаризации", "error")
                # Возвращаем пустую суммаризацию чтобы не ломать конвейер
                return {
                    "text": "",
                    "prompt": prompt,
                    "prompt_type": prompt_type,
                    "temperature": temperature,
                    "metrics": {"sum_score": 0.0, "geval": 0.0, "llm_judge": 1.0, "meteor": 0.0, "bert_score": 0.0}
                }
            
            # Расчет метрик
            metrics = await self.metrics_calculator.calculate_summary_metrics(
                original_text=input_text,
                summary_text=summary_text,
                reference_summary=reference_summary,
                lm_client=self.lm_client
            )
            
            return {
                "text": summary_text,
                "prompt": prompt,
                "prompt_type": prompt_type,
                "temperature": temperature,
                "metrics": metrics
            }
            
        except Exception as e:
            self.log_execution(f"Ошибка генерации суммаризации: {e}", "error")
            return None
    
    def _build_few_shot_prompt(self, input_text: str) -> str:
        """
        Построение few-shot промпта для суммаризации.
        
        Args:
            input_text: Входной текст
            
        Returns:
            Few-shot промпт
        """
        # Получаем примеры из памяти
        examples = self.memory.get_summary_few_shot_examples(
            input_text=input_text,
            max_examples=3
        )
        
        prompt_parts = ["Создай краткое изложение следующих примеров:"]
        
        for example in examples:
            prompt_parts.append(f"Текст: {example['input'][:200]}...")
            prompt_parts.append(f"Изложение: {example['output']}")
            prompt_parts.append("")
        
        prompt_parts.append(f"Теперь создай изложение этого текста:")
        prompt_parts.append("{text}")
        
        return "\n".join(prompt_parts)
    
    def _is_english_text(self, text: str) -> bool:
        """Определение языка текста."""
        import re
        cyrillic = re.compile(r'[а-яёА-ЯЁ]')
        latin = re.compile(r'[a-zA-Z]')
        cyrillic_count = len(cyrillic.findall(text))
        latin_count = len(latin.findall(text))
        # Если кириллицы больше 30% от латиницы - русский
        if cyrillic_count > latin_count * 0.3:
            return False
        return True
    
    def _build_extractive_prompt(self, input_text: str) -> str:
        """
        Построение extractive промпта (извлечение ключевых предложений).
        
        Args:
            input_text: Входной текст
            
        Returns:
            Extractive промпт
        """
        return f"""Выдели самые важные предложения из текста и объедини их в краткое изложение. Сохраняй ключевые факты и идеи.

Текст: {{text}}

Ключевые предложения в виде изложения:"""
    
    def _build_abstractive_prompt(self, input_text: str) -> str:
        """
        Построение abstractive промпта (абстрактная суммаризация).
        
        Args:
            input_text: Входной текст
            
        Returns:
            Abstractive промпт
        """
        return f"""Перескажи основное содержание текста своими словами. Создай новое, более краткое изложение, которое передает суть оригинала.

Текст: {{text}}

Изложение своими словами:"""
    
    def _get_saved_prompts(self) -> List[str]:
        """
        Получение сохраненных промптов из памяти.
        
        Returns:
            Список промптов
        """
        domain = "general"  # Можно получать из состояния
        
        prompts = []
        
        # Добавляем стандартные промпты как запасные (промпт №2 первым)
        prompts.extend([
            "Промпт №2: Создай краткую, но информативную суммаризацию текста, сохраняя все ключевые факты, имена, даты и числа. Особое внимание удели структуре и логике изложения. Суммаризация должна быть на 70% короче оригинала: {text}",
            "Создай краткое изложение основных идей текста: {text}",
            "Выдели главное из текста и представь в сжатой форме: {text}",
            "Суммаризируй ключевые моменты текста: {text}"
        ])
        
        return prompts
    
    async def _select_best_variant(self, 
                                 variants: List[Dict[str, Any]], 
                                 input_text: str,
                                 reference_summary: str) -> Dict[str, Any]:
        """
        Выбор лучшего варианта на основе SumScore.
        
        Args:
            variants: Список вариантов
            input_text: Исходный текст
            reference_summary: Эталонная суммаризация
            
        Returns:
            Лучший вариант
        """
        if not variants:
            # Возвращаем заглушку если нет вариантов
            return {
                "text": "",
                "prompt_type": "none",
                "temperature": 0.0,
                "metrics": {"sum_score": 0.0}
            }
        
        # Расчет метрик для каждого варианта
        for variant in variants:
            if "sum_score" not in variant["metrics"]:
                # Расчет SumScore если отсутствует
                metrics = await self.metrics_calculator.calculate_summary_metrics(
                    original_text=input_text,
                    summary_text=variant["text"],
                    reference_summary=reference_summary,
                    lm_client=self.lm_client
                )
                variant["metrics"].update(metrics)
        
        # Выбор варианта с максимальным SumScore
        best_variant = max(variants, key=lambda v: v["metrics"].get("sum_score", 0))
        
        self.log_execution(f"Лучший вариант: {best_variant['prompt_type']} "
                         f"(SumScore={best_variant['metrics'].get('sum_score', 0):.3f})")
        
        return best_variant
    
    async def _adaptive_summary_attempts(self,
                                     input_text: str,
                                     reference_summary: str,
                                     current_best: Dict[str, Any]) -> Dict[str, Any]:
        """
        Адаптивные попытки улучшения суммаризации.
        
        Args:
            input_text: Входной текст
            reference_summary: Эталонная суммаризация
            current_best: Текущий лучший вариант
            
        Returns:
            Улучшенный вариант
        """
        attempts = 0
        best_variant = current_best
        
        # Последовательность попыток
        attempt_strategies = [
            ("few_shot", self._build_few_shot_prompt(input_text), 0.3),
            ("extractive", self._build_extractive_prompt(input_text), 0.2),
            ("abstractive", self._build_abstractive_prompt(input_text), 0.6),
            ("high_temp", "Создай лучшую суммаризацию: {text}", 0.9),
            ("low_temp", "Создай точную суммаризацию: {text}", 0.1)
        ]
        
        for strategy_name, prompt, temp in attempt_strategies:
            if attempts >= config.SUMMARIZATION_ATTEMPTS:
                break
                
            if best_variant["metrics"].get("sum_score", 0) >= config.SUMMARY_THRESHOLDS["satisfactory"]:
                break
            
            self.log_execution(f"Адаптивная попытка {attempts + 1}: {strategy_name}")
            
            try:
                result = await self._generate_single_summary(
                    input_text=input_text,
                    prompt=prompt,
                    temperature=temp,
                    prompt_type=strategy_name
                )
                
                if result:
                    # Расчет полных метрик
                    metrics = await self.metrics_calculator.calculate_summary_metrics(
                        original_text=input_text,
                        summary_text=result["text"],
                        reference_summary=reference_summary,
                        lm_client=self.lm_client
                    )
                    result["metrics"] = metrics
                    
                    if metrics.get("sum_score", 0) > best_variant["metrics"].get("sum_score", 0):
                        best_variant = result
                        self.log_execution(f"Улучшение найдено (SumScore={metrics.get('sum_score', 0):.3f})")
                
            except Exception as e:
                self.log_execution(f"Ошибка в адаптивной попытке {strategy_name}: {e}", "warning")
            
            attempts += 1
        
        return best_variant
    
    def _build_cot_summary_prompt(self, input_text: str) -> str:
        """
        Build Chain-of-Thought prompt for summarization.
        
        Args:
            input_text: Input text
            
        Returns:
            CoT prompt for summarization
        """
        return f"""Analyze the text step by step and create a summary:

Text: {input_text}

Step 1: Identify the main topic and key points
Step 2: Determine the most important information
Step 3: Eliminate redundant details
Step 4: Formulate a concise summary

Summary:"""

    def _learn_from_summarization(self,
                                input_text: str,
                                summary_text: str,
                                reference_summary: str,
                                metrics: Dict[str, float],
                                prompt: str):
        """
        Save successful summarization to memory.
        
        Args:
            input_text: Original text
            summary_text: Summarization
            reference_summary: Reference summarization
            metrics: Metrics
            prompt: Prompt used
        """
        # Сохраняем только если качество хорошее
        if metrics.get("sum_score", 0) > config.PROMPT_CACHE_MIN_IMPROVEMENT:
            self.memory.learn_from_summarization(
                original_text=input_text,
                summary_text=summary_text,
                reference_summary=reference_summary,
                metrics=metrics,
                prompt=prompt,
                model=config.MODEL_NAME
            )
