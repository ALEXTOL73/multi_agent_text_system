"""
Агрегатор суммаризаций для улучшения лучшего варианта через LLM
"""

from typing import Dict, Any, List
from Levenshtein import distance as levenshtein_distance

from .base_agent import BaseAgent
from ..utils.lm_studio_client import LMStudioClient
from ..metrics.metrics_calculator import MetricsCalculator
import config

class SummarizerAggregator(BaseAgent):
    """
    Агрегатор суммаризаций для улучшения лучшего варианта через LLM
    """
    
    def __init__(self, 
                 lm_client: LMStudioClient,
                 metrics_calculator: MetricsCalculator):
        """
        Инициализация агрегатора суммаризаций.
        
        Args:
            lm_client: Клиент LM Studio
            metrics_calculator: Калькулятор метрик
        """
        super().__init__("SummarizerAggregator")
        self.lm_client = lm_client
        self.metrics_calculator = metrics_calculator
        
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Основной метод выполнения агента.
        
        Args:
            state: Текущее состояние системы
            
        Returns:
            Обновленное состояние системы
        """
        self.log_execution("Начало работы агрегатора суммаризаций")
        
        # Проверка обязательных ключей
        if not self.validate_state(state, ["ensemble_summary_outputs", "summary"]):
            return state
            
        ensemble_outputs = self.get_from_state(state, "ensemble_summary_outputs", [])
        ensemble_prompts = self.get_from_state(state, "ensemble_summary_prompts", [])
        summary = self.get_from_state(state, "summary")
        input_text = self.get_from_state(state, "corrected_text")
        reference_summary = self.get_from_state(state, "reference_summary")
        
        # Если недостаточно вариантов для агрегации, пропускаем
        if len(ensemble_outputs) < 2:
            self.add_to_state(state,
                           aggregator_used=False,
                           aggregation_reason="insufficient_variants")
            return state
        
        # Выбор лучшего базового варианта из ансамбля
        basic_variants = []
        basic_metrics = []
        
        # Find basic variants and their metrics
        for i, (output, prompt_type) in enumerate(zip(ensemble_outputs, ensemble_prompts)):
            if prompt_type == "basic":
                basic_variants.append(output)
                # Calculate metrics for this basic variant
                metrics = await self.metrics_calculator.calculate_summary_metrics(
                    original_text=input_text,
                    summary_text=output,
                    reference_summary=reference_summary,
                    lm_client=self.lm_client
                )
                basic_metrics.append(metrics)
        
        if basic_variants:
            # Select best basic variant by SumScore
            best_idx = max(range(len(basic_variants)), 
                          key=lambda i: basic_metrics[i].get("sum_score", 0))
            best_variant = basic_variants[best_idx]
            best_metrics = basic_metrics[best_idx]
            self.log_execution(f"Selected best basic variant (index {best_idx}) with SumScore={best_metrics.get('sum_score', 0):.3f}")
        else:
            # Fallback to summary if no basic variants
            best_variant = summary
            best_metrics = await self.metrics_calculator.calculate_summary_metrics(
                original_text=input_text,
                summary_text=best_variant,
                reference_summary=reference_summary,
                lm_client=self.lm_client
            )
        
        # Получение альтернативных вариантов
        alternatives = self._get_alternatives(ensemble_outputs, best_variant)
        
        if not alternatives:
            self.log_execution("Нет альтернативных вариантов для агрегации")
            self.add_to_state(state,
                           aggregator_used=False,
                           aggregation_reason="no_alternatives")
            return state
        
        self.log_execution(f"Получено {len(alternatives)} альтернативных вариантов")
        
        # Генерация агрегированного варианта
        aggregated_text = await self._generate_aggregated_summary(
            best_variant=best_variant,
            alternatives=alternatives,
            original_text=input_text
        )
        
        if not aggregated_text:
            self.log_execution("Не удалось сгенерировать агрегированную суммаризацию")
            self.add_to_state(state,
                           aggregator_used=False,
                           aggregation_reason="generation_failed")
            return state
        
        # Оценка агрегированного варианта
        aggregated_metrics = await self.metrics_calculator.calculate_summary_metrics(
            original_text=input_text,
            summary_text=aggregated_text,
            reference_summary=reference_summary,
            lm_client=self.lm_client
        )
        
        # Сравнение и выбор финального варианта
        final_text, final_metrics, used_aggregation = self._select_final_variant(
            best_variant=best_variant,
            best_metrics=best_metrics,
            aggregated_text=aggregated_text,
            aggregated_metrics=aggregated_metrics,
            original_text=input_text
        )
        
        # Обновление состояния
        self.add_to_state(state,
                         summary=final_text,
                         summary_metrics=final_metrics,
                         aggregator_used=used_aggregation,
                         aggregation_reason=self._get_aggregation_reason(used_aggregation, final_text, best_variant))
        
        self.log_execution(f"Агрегация {'использована' if used_aggregation else 'не использована'}")
        return state
    
    def _get_alternatives(self, ensemble_outputs: List[str], best_variant: str) -> List[str]:
        """
        Получение альтернативных вариантов для агрегации.
        
        Args:
            ensemble_outputs: Все варианты из ансамбля
            best_variant: Лучший вариант
            
        Returns:
            Список альтернативных вариантов
        """
        alternatives = []
        
        for output in ensemble_outputs:
            if output != best_variant and output.strip():
                alternatives.append(output)
                if len(alternatives) >= 2:  # Берем максимум 2 альтернативы
                    break
        
        return alternatives
    
    async def _generate_aggregated_summary(self,
                                         best_variant: str,
                                         alternatives: List[str],
                                         original_text: str) -> str:
        """
        Генерация улучшенной суммаризации через LLM.
        
        Args:
            best_variant: Лучший вариант
            alternatives: Альтернативные варианты
            original_text: Оригинальный текст
            
        Returns:
            Агрегированная суммаризация
        """
        try:
            # Detect language of the best variant
            is_english = self._is_english_text(best_variant)
            
            if is_english:
                aggregation_prompt = f"""Improve the following summary by using the best elements from alternative variants:

Main variant: {best_variant}

Alternative variant 1: {alternatives[0] if len(alternatives) > 0 else ""}
Alternative variant 2: {alternatives[1] if len(alternatives) > 1 else ""}

Original text for context: {original_text[:500]}...

Create a final summary that:
1. Preserves key ideas from all variants
2. Improves style and clarity
3. Remains concise and informative
4. Accurately reflects the original content
5. CRITICAL: Output must be in ENGLISH language only

Return only the improved summary without additional comments."""
                
                system_prompt = "You are an expert at improving texts. Create the best summary in ENGLISH by combining the strengths of different versions."
            else:
                aggregation_prompt = f"""Улучши следующую суммаризацию, используя лучшие элементы из альтернативных вариантов:

Основной вариант: {best_variant}

Альтернативный вариант 1: {alternatives[0] if len(alternatives) > 0 else ""}
Альтернативный вариант 2: {alternatives[1] if len(alternatives) > 1 else ""}

Оригинальный текст для контекста: {original_text[:500]}...

Создай финальную суммаризацию, которая:
1. Сохраняет ключевые идеи из всех вариантов
2. Улучшает стиль и ясность
3. Остается краткой и информативной
4. Точно отражает содержание оригинала
5. КРИТИЧНО: Вывод должен быть на РУССКОМ языке только

Верни только улучшенную суммаризацию без дополнительных комментариев."""
                
                system_prompt = "You are an expert at improving texts. Create the best summary in RUSSIAN by combining the strengths of different versions."
            
            # Generate improved summary
            aggregated_summary = await self.lm_client.generate_with_retry(
                prompt=aggregation_prompt,
                temperature=config.AGGREGATION_TEMPERATURE,
                max_tokens=1024,
                system_prompt=system_prompt
            )
            
            return aggregated_summary.strip()
            
        except Exception as e:
            self.log_execution(f"Ошибка генерации агрегированной суммаризации: {e}", "error")
            return ""
    
    def _select_final_variant(self,
                           best_variant: str,
                           best_metrics: Dict[str, float],
                           aggregated_text: str,
                           aggregated_metrics: Dict[str, float],
                           original_text: str) -> tuple:
        """
        Выбор финального варианта на основе метрик.
        
        Args:
            best_variant: Лучший вариант из ансамбля
            best_metrics: Метрики лучшего варианта
            aggregated_text: Агрегированный текст
            aggregated_metrics: Метрики агрегированного текста
            original_text: Исходный текст
            
        Returns:
            Кортеж (финальный текст, финальные метрики, использована ли агрегация)
        """
        # Расчет композитных оценок
        best_score = best_metrics.get("sum_score", 0)
        aggregated_score = aggregated_metrics.get("sum_score", 0)
        
        self.log_execution(f"Оценки: лучший={best_score:.3f}, агрегированный={aggregated_score:.3f}")
        
        # Проверка сходства с лучшим вариантом
        similarity = self._calculate_similarity(best_variant, aggregated_text)
        self.log_execution(f"Сходство с лучшим вариантом: {similarity:.3f}")
        
        # Проверка длины
        length_ratio = len(aggregated_text) / len(original_text) if len(original_text) > 0 else 0
        best_length_ratio = len(best_variant) / len(original_text) if len(original_text) > 0 else 0
        self.log_execution(f"Соотношение длины: лучший={best_length_ratio:.3f}, агрегированный={length_ratio:.3f}")
        
        # Дополнительные проверки для суммаризации
        length_ok = 0.1 <= length_ratio <= 0.8  # Суммаризация должна быть 10-80% от оригинала
        
        # Принятие решения
        if (aggregated_score > best_score and 
            similarity >= config.MIN_SIMILARITY and 
            length_ok):
            self.log_execution("Выбран агрегированный вариант")
            return aggregated_text, aggregated_metrics, True
        else:
            self.log_execution("Выбран лучший вариант из ансамбля")
            return best_variant, best_metrics, False
    
    def _is_english_text(self, text: str) -> bool:
        """
        Detect if text is in English language.
        
        Args:
            text: Text to analyze
            
        Returns:
            True if text is in English, False if Russian
        """
        import re
        
        # Check for Cyrillic characters (more comprehensive pattern)
        cyrillic_pattern = re.compile(r'[à-ÿÀ-ß]')
        latin_pattern = re.compile(r'[a-zA-Z]')
        
        # Count characters
        cyrillic_count = len(cyrillic_pattern.findall(text))
        latin_count = len(latin_pattern.findall(text))
        
        # Determine language based on character distribution
        if cyrillic_count > latin_count * 0.3:  # If 30%+ Cyrillic
            return False
        elif latin_count > cyrillic_count * 0.3:  # If 30%+ Latin
            return True
        else:  # Mixed content - prefer English
            return True
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate Levenshtein similarity between texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity in range [0, 1]
        """
        try:
            max_len = max(len(text1), len(text2))
            if max_len == 0:
                return 1.0
                
            distance = levenshtein_distance(text1, text2)
            similarity = 1 - (distance / max_len)
            return similarity
        except Exception as e:
            self.log_execution(f"Error calculating similarity: {e}", "error")
            return 0.0
    
    def _get_aggregation_reason(self, 
                              used_aggregation: bool,
                              final_text: str,
                              best_variant: str) -> str:
        """
        Получение причины выбора агрегации или ее отклонения.
        
        Args:
            used_aggregation: Была ли использована агрегация
            final_text: Финальный текст
            best_variant: Лучший вариант из ансамбля
            
        Returns:
            Строка с причиной
        """
        if not used_aggregation:
            if final_text == best_variant:
                return "best_variant_better"
            else:
                return "aggregation_failed"
        else:
            return "aggregation_improved"
