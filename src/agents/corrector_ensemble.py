"""
Ансамбль корректоров для генерации нескольких вариантов исправления текста
"""

import asyncio
from typing import Dict, Any, List, Optional
import random

from .base_agent import BaseAgent
from ..utils.lm_studio_client import LMStudioClient
from ..utils.agent_memory import AgentMemory
from ..metrics.metrics_calculator import MetricsCalculator
import config

class CorrectorEnsemble(BaseAgent):
    """
    Ансамбль корректоров для генерации и оценки вариантов исправления текста
    """
    
    def __init__(self, 
                 lm_client: LMStudioClient,
                 memory: AgentMemory,
                 metrics_calculator: MetricsCalculator):
        """
        Инициализация ансамбля корректоров.
        
        Args:
            lm_client: Клиент LM Studio
            memory: Система памяти агентов
            metrics_calculator: Калькулятор метрик
        """
        super().__init__("CorrectorEnsemble")
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
        self.log_execution("Начало работы ансамбля корректоров")
        
        # Проверка обязательных ключей
        if not self.validate_state(state, ["input_text", "prompt_correction"]):
            return state
            
        input_text = self.get_from_state(state, "input_text")
        reference_text = self.get_from_state(state, "reference_text")
        prompt_correction = self.get_from_state(state, "prompt_correction")
        domain = self.get_from_state(state, "domain", "general")
        
        # Вычисление исходных метрик
        if reference_text:
            # Для исходных метрик сравниваем Incorrect vs Etalon
            original_wer = self.metrics_calculator.calculate_wer(reference_text, input_text)
            original_lev = self.metrics_calculator.calculate_lev_rating(reference_text, input_text)
            
            original_metrics = {
                "wer_original": original_wer,
                "wer_corrected": original_wer,  # Начальное значение, будет обновлено после коррекции
                "lev_original": original_lev,
                "lev_corrected": original_lev,   # Начальное значение, будет обновлено после коррекции
                "delta_wer": 0.0,
                "delta_lev": 0.0,
                "perplexity": float('inf'),
                "cor_score": 0.0
            }
        else:
            # Нет эталона - используем пустые метрики
            original_metrics = {
                "wer_original": 1.0,
                "wer_corrected": 1.0,
                "lev_original": 0.0,
                "lev_corrected": 0.0,
                "delta_wer": 0.0,
                "delta_lev": 0.0,
                "perplexity": float('inf'),
                "cor_score": 0.0
            }
        
        self.log_execution(f"Исходные метрики: WER={original_metrics.get('wer_original', 0):.3f}, "
                         f"Lev={original_metrics.get('lev_original', 0):.3f}")
        
        # Генерация вариантов исправления
        variants_data = await self._generate_correction_variants(
            input_text=input_text,
            prompt_correction=prompt_correction,
            original_metrics=original_metrics,
            reference_text=reference_text
        )
        
        # Оценка вариантов и выбор лучшего
        best_variant = self._select_best_variant(variants_data, original_metrics)
        
        # Адаптивные попытки если качество низкое
        if best_variant["delta_lev"] < config.DELTA_LEV_THRESHOLD:
            self.log_execution(f"Низкое качество (delta_lev={best_variant['delta_lev']:.3f}), "
                             f"запускаем адаптивные попытки")
            best_variant = await self._adaptive_correction_attempts(
                input_text=input_text,
                reference_text=reference_text,
                original_metrics=original_metrics,
                current_best=best_variant
            )
        
        # Сохранение в память
        self._learn_from_correction(
            input_text=input_text,
            corrected_text=best_variant["text"],
            reference_text=reference_text,
            metrics=best_variant["metrics"],
            prompt=prompt_correction
        )
        
        # Обновление состояния
        self.add_to_state(state,
                         ensemble_outputs=[v["text"] for v in variants_data],
                         ensemble_prompts=[v["prompt_type"] for v in variants_data],
                         ensemble_temperatures=[v["temperature"] for v in variants_data],
                         ensemble_metrics=[v["metrics"] for v in variants_data],
                         corrected_text=best_variant["text"],
                         best_correction_prompt=best_variant.get("prompt", ""),
                         metrics_correction=best_variant["metrics"])
        
        self.log_execution(f"Выбран лучший вариант с CorScore={best_variant['metrics']['cor_score']:.3f}")
        return state
        
    async def _generate_correction_variants(self, 
                                         input_text: str,
                                         prompt_correction: str,
                                         original_metrics: Dict[str, float],
                                         reference_text: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Генерация вариантов исправления текста.
        
        Args:
            input_text: Входной текст
            prompt_correction: Промпт коррекции
            original_metrics: Исходные метрики
            reference_text: Эталонный текст
            
        Returns:
            Список вариантов с метриками
        """
        variants = []
        tasks = []
        
        # 3 Basic prompts with different temperatures
        for i, temp in enumerate(config.CORRECTION_TEMPERATURES):
            task = self._generate_single_correction(
                input_text=input_text,
                prompt=prompt_correction,
                temperature=temp,
                prompt_type="basic",
                reference_text=reference_text,
                original_metrics=original_metrics
            )
            tasks.append(task)
        
        # 1 Best saved prompt from memory
        saved_prompts = self._get_saved_prompts()
        if saved_prompts:
            best_saved_prompt = saved_prompts[0]  # Get the best one
            task = self._generate_single_correction(
                input_text=input_text,
                prompt=best_saved_prompt,
                temperature=0.4,
                prompt_type="saved",
                reference_text=reference_text,
                original_metrics=original_metrics
            )
            tasks.append(task)
        
        # 1 Few-shot prompt (from 3 examples)
        if config.USE_FEW_SHOT_PROMPT:
            few_shot_prompt = self._build_few_shot_prompt(input_text)
            task = self._generate_single_correction(
                input_text=input_text,
                prompt=few_shot_prompt,
                temperature=0.3,
                prompt_type="few-shot",
                reference_text=reference_text,
                original_metrics=original_metrics
            )
            tasks.append(task)
        
        # 1 Chain-of-Thought prompt
        if config.USE_CHAIN_OF_THOUGHT_PROMPT:
            cot_prompt = self._build_cot_prompt(input_text)
            task = self._generate_single_correction(
                input_text=input_text,
                prompt=cot_prompt,
                temperature=0.5,
                prompt_type="cot",
                reference_text=reference_text,
                original_metrics=original_metrics
            )
            tasks.append(task)
        
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
        
        # Self-consistency дополнительные варианты
        if config.SELF_CONSISTENCY_ENABLED and variants:
            best_temp = variants[0]["temperature"]
            for i in range(config.SELF_CONSISTENCY_EXTRA_COUNT):
                task = self._generate_single_correction(
                    input_text=input_text,
                    prompt=prompt_correction,
                    temperature=best_temp,
                    prompt_type="self-consistency",
                    reference_text=reference_text,
                    original_metrics=original_metrics
                )
                try:
                    result = await task
                    if result:
                        variants.append(result)
                except Exception as e:
                    self.log_execution(f"Ошибка self-consistency варианта: {e}", "warning")
        
        return variants
    
    async def _generate_single_correction(self,
                                       input_text: str,
                                       prompt: str,
                                       temperature: float,
                                       prompt_type: str,
                                       reference_text: Optional[str] = None,
                                       original_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Генерация одного варианта исправления.
        
        Args:
            input_text: Входной текст
            prompt: Промпт
            temperature: Температура
            prompt_type: Тип промпта
            reference_text: Эталонный текст
            
        Returns:
            Словарь с вариантом и метриками
        """
        try:
            # Проверка наличия текстов на входе
            if not input_text or input_text.strip() == "":
                self.log_execution("ОШИБКА: Пустой текст на входе коррекции", "error")
                # Возвращаем оригинальный текст чтобы не ломать конвейер
                return {
                    "text": input_text,
                    "prompt": prompt,
                    "prompt_type": prompt_type,
                    "temperature": temperature,
                    "metrics": {"cor_score": 0.0, "wer": 1.0, "delta_lev": 0.0}
                }
            
            # Формирование финального промпта
            final_prompt = prompt.replace("{text}", input_text)
            
            # Генерация исправленного текста
            corrected_text = await self.lm_client.generate_with_retry(
                prompt=final_prompt,
                temperature=temperature,
                max_tokens=2048,
                system_prompt=config.CORRECTION_SYSTEM_PROMPT
            )
            
            # Проверка наличия текста на выходе
            if not corrected_text or corrected_text.strip() == "":
                self.log_execution("ОШИБКА: Пустой текст на выходе коррекции", "error")
                # Возвращаем оригинальный текст чтобы не ломать конвейер
                return {
                    "text": input_text,
                    "prompt": prompt,
                    "prompt_type": prompt_type,
                    "temperature": temperature,
                    "metrics": {"cor_score": 0.0, "wer": 1.0, "delta_lev": 0.0}
                }
            
            # Calculation of metrics
            try:
                # Always calculate metrics for this specific variant
                metrics = self.metrics_calculator.calculate_correction_metrics(
                    original_text=input_text,
                    corrected_text=corrected_text,
                    reference_text=reference_text
                )
            except Exception as e:
                self.log_execution(f"Error calculating metrics: {e}", "error")
                metrics = {"cor_score": 0.0, "delta_wer": 0.0, "delta_lev": 0.0, "perplexity": float('inf')}
            
            return {
                "text": corrected_text,
                "prompt": prompt,
                "prompt_type": prompt_type,
                "temperature": temperature,
                "metrics": metrics,
                "delta_wer": metrics.get("delta_wer", 0),
                "delta_lev": metrics.get("delta_lev", 0),
                "perplexity": metrics.get("perplexity", float('inf'))
            }
            
        except Exception as e:
            self.log_execution(f"Ошибка генерации исправления: {e}", "error")
            return None
    
    def _get_dynamic_temperatures(self, input_text: str, original_metrics: Dict[str, float]) -> List[float]:
        """
        Определение динамических температур на основе длины текста и качества.
        
        Args:
            input_text: Входной текст
            original_metrics: Исходные метрики
            
        Returns:
            Список температур
        """
        if not config.DYNAMIC_TEMPERATURES_ENABLED:
            return config.CORRECTION_TEMPERATURES
        
        text_length = len(input_text)
        wer = original_metrics.get("wer_original", 0.5)
        
        # Адаптация температур
        if text_length < 50:
            # Короткий текст - более низкие температуры
            base_temps = [0.2, 0.4, 0.6]
        elif text_length < 200:
            # Средний текст
            base_temps = [0.3, 0.5, 0.7]
        else:
            # Длинный текст - более высокие температуры
            base_temps = [0.4, 0.6, 0.8]
        
        # Если WER высокий, увеличиваем температуры для большего разнообразия
        if wer > 0.5:
            base_temps = [min(t + 0.2, 1.0) for t in base_temps]
        
        return base_temps
    
    def _build_few_shot_prompt(self, input_text: str) -> str:
        """
        Построение few-shot промпта.
        
        Args:
            input_text: Входной текст
            
        Returns:
            Few-shot промпт
        """
        # Получаем примеры из памяти
        examples = self.memory.get_few_shot_examples(
            input_text=input_text,
            max_examples=3
        )
        
        prompt_parts = ["Исправь ошибки в следующих примерах:"]
        
        for example in examples:
            prompt_parts.append(f"Неверно: {example['input']}")
            prompt_parts.append(f"Верно: {example['output']}")
            prompt_parts.append("")
        
        prompt_parts.append(f"Теперь исправь этот текст:")
        prompt_parts.append("{text}")
        
        return "\n".join(prompt_parts)
    
    def _build_cot_prompt(self, input_text: str) -> str:
        """
        Построение Chain-of-Thought промпта.
        
        Args:
            input_text: Входной текст
            
        Returns:
            CoT промпт
        """
        return f"""Проанализируй текст по шагам и исправь ошибки:

1. Найди орфографические ошибки
2. Проверь грамматику
3. Исправь пунктуацию
4. Убедись, что смысл сохранен

Текст для анализа: {{text}}

Исправленный текст:"""
    
    def _get_saved_prompts(self) -> List[str]:
        """
        Получение сохраненных промптов из памяти.
        
        Returns:
            Список промптов
        """
        domain = "general"  # Можно получать из состояния
        best_prompt = self.memory.get_best_prompt_for_domain(domain)
        
        prompts = []
        if best_prompt:
            prompts.append(best_prompt)
        
        # Добавляем стандартные промпты как запасные
        prompts.extend([
            "Исправь все ошибки в тексте, сохраняя смысл: {text}",
            "Отредактируй текст для исправления ошибок: {text}"
        ])
        
        return prompts
    
    def _select_best_variant(self, 
                           variants: List[Dict[str, Any]], 
                           original_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Выбор лучшего варианта на основе композитной оценки.
        
        Args:
            variants: Список вариантов
            original_metrics: Исходные метрики
            
        Returns:
            Лучший вариант
        """
        if not variants:
            # Возвращаем заглушку если нет вариантов
            return {
                "text": "",
                "prompt_type": "none",
                "temperature": 0.0,
                "metrics": {"cor_score": 0.0},
                "delta_wer": 0.0,
                "delta_lev": 0.0,
                "perplexity": float('inf')
            }
        
        # Расчет композитной оценки для каждого варианта
        for variant in variants:
            metrics = variant["metrics"]
            
            # Kompozitnaya otsenka - ispol'zuem yedinuyu formulu CorScore
            original_wer = variant["metrics"].get("wer_original", 1.0)
            corrected_wer = variant["metrics"].get("wer_corrected", 1.0)
            original_lev = variant["metrics"].get("lev_original", 0.0)
            corrected_lev = variant["metrics"].get("lev_corrected", 0.0)
            perplexity = variant["metrics"].get("perplexity", float('inf'))
            
            # Debug logging
            delta_wer = original_wer - corrected_wer
            delta_lev = corrected_lev - original_lev
            expected_score = delta_wer + delta_lev * 10 + (1 - min(perplexity, 100) / 100) * 0.2
            
                        
            composite_score = self.metrics_calculator.calculate_cor_score(
                original_wer=original_wer,
                corrected_wer=corrected_wer,
                original_lev=original_lev,
                corrected_lev=corrected_lev,
                perplexity=perplexity
            )
            
            variant["composite_score"] = composite_score
        
        # Выбор варианта с максимальной оценкой
        best_variant = max(variants, key=lambda v: v["composite_score"])
        
        self.log_execution(f"Лучший вариант: {best_variant['prompt_type']} "
                         f"(score={best_variant['composite_score']:.3f})")
        
        return best_variant
    
    async def _adaptive_correction_attempts(self,
                                        input_text: str,
                                        reference_text: str,
                                        original_metrics: Dict[str, float],
                                        current_best: Dict[str, Any]) -> Dict[str, Any]:
        """
        Адаптивные попытки улучшения коррекции.
        
        Args:
            input_text: Входной текст
            reference_text: Эталонный текст
            original_metrics: Исходные метрики
            current_best: Текущий лучший вариант
            
        Returns:
            Улучшенный вариант
        """
        attempts = 0
        best_variant = current_best
        
        # Последовательность попыток
        attempt_strategies = [
            ("few-shot", self._build_few_shot_prompt(input_text), 0.3),
            ("CoT", self._build_cot_prompt(input_text), 0.5),
            ("high_temp", f"Исправь ошибки: {{text}}", 0.9),
            ("saved", self._get_saved_prompts()[0] if self._get_saved_prompts() else f"Исправь: {{text}}", 0.4)
        ]
        
        for strategy_name, prompt, temp in attempt_strategies:
            if attempts >= config.MAX_LEV_RETRY_ATTEMPTS:
                break
                
            if best_variant["delta_lev"] >= config.DELTA_LEV_THRESHOLD:
                break
            
            self.log_execution(f"Адаптивная попытка {attempts + 1}: {strategy_name}")
            
            try:
                result = await self._generate_single_correction(
                    input_text=input_text,
                    prompt=prompt,
                    temperature=temp,
                    prompt_type=f"adaptive_{strategy_name}",
                    reference_text=reference_text,
                    original_metrics=original_metrics
                )
                
                if result and result["delta_lev"] > best_variant["delta_lev"]:
                    best_variant = result
                    self.log_execution(f"Улучшение найдено (delta_lev={result['delta_lev']:.3f})")
                
            except Exception as e:
                self.log_execution(f"Ошибка в адаптивной попытке {strategy_name}: {e}", "warning")
            
            attempts += 1
        
        return best_variant
    
    def _learn_from_correction(self,
                             input_text: str,
                             corrected_text: str,
                             reference_text: str,
                             metrics: Dict[str, float],
                             prompt: str):
        """
        Сохранение успешной коррекции в память.
        
        Args:
            input_text: Исходный текст
            corrected_text: Исправленный текст
            reference_text: Эталонный текст
            metrics: Метрики
            prompt: Использованный промпт
        """
        # Сохраняем только если качество хорошее
        if metrics.get("cor_score", 0) > config.PROMPT_CACHE_MIN_IMPROVEMENT:
            self.memory.learn_from_correction(
                original_text=input_text,
                corrected_text=corrected_text,
                reference_text=reference_text,
                metrics=metrics,
                prompt=prompt,
                model=config.MODEL_NAME
            )
            
            # Кэшируем промпт
            improvement = metrics.get("cor_score", 0)
            self.memory.cache_prompt(prompt, metrics, improvement)
