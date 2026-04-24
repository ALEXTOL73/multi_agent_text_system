"""
Агрегатор коррекций для улучшения лучшего варианта через LLM
"""

from typing import Dict, Any, List
from Levenshtein import distance as levenshtein_distance

from .base_agent import BaseAgent
from ..utils.lm_studio_client import LMStudioClient
from ..metrics.metrics_calculator import MetricsCalculator
import config

class CorrectorAggregator(BaseAgent):
    """
    Агрегатор коррекций для улучшения лучшего варианта через LLM
    """
    
    def __init__(self, 
                 lm_client: LMStudioClient,
                 metrics_calculator: MetricsCalculator):
        """
        Инициализация агрегатора коррекций.
        
        Args:
            lm_client: Клиент LM Studio
            metrics_calculator: Калькулятор метрик
        """
        super().__init__("CorrectorAggregator")
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
        self.log_execution("Начало работы агрегатора коррекций")
        
        # Проверка обязательных ключей
        if not self.validate_state(state, ["ensemble_outputs", "corrected_text"]):
            return state
            
        ensemble_outputs = self.get_from_state(state, "ensemble_outputs", [])
        ensemble_prompts = self.get_from_state(state, "ensemble_prompts", [])
        corrected_text = self.get_from_state(state, "corrected_text")
        input_text = self.get_from_state(state, "input_text")
        reference_text = self.get_from_state(state, "reference_text")
        
        # Если недостаточно вариантов для агрегации, пропускаем
        if len(ensemble_outputs) < 2:
            self.log_execution("Недостаточно вариантов для агрегации, используем лучший вариант")
            self.add_to_state(state,
                           aggregator_used=False,
                           aggregation_reason="insufficient_variants")
            return state
        
        # Select best basic variant from ensemble
        basic_variants = []
        basic_metrics = []
        
        # Find basic variants and their metrics
        for i, (output, prompt_type) in enumerate(zip(ensemble_outputs, ensemble_prompts)):
            if prompt_type == "basic":
                basic_variants.append(output)
                # Calculate metrics for this basic variant
                metrics = self.metrics_calculator.calculate_correction_metrics(
                    original_text=input_text,
                    corrected_text=output,
                    reference_text=reference_text
                )
                basic_metrics.append(metrics)
        
        if basic_variants:
            # Select best basic variant with priority deltaLev>0.05 and deltaWER>0.1
            priority_indices = []
            for i, metrics in enumerate(basic_metrics):
                delta_wer = metrics.get("wer_original", 1.0) - metrics.get("wer_corrected", 1.0)
                delta_lev = metrics.get("lev_corrected", 0.0) - metrics.get("lev_original", 0.0)
                
                if delta_lev > 0.05 and delta_wer > 0.1:
                    priority_indices.append(i)
            
            if priority_indices:
                # Choose best among priority variants by CorScore
                best_idx = max(priority_indices, key=lambda i: basic_metrics[i].get("cor_score", 0))
                best_metrics = basic_metrics[best_idx]
                delta_wer = best_metrics.get("wer_original", 1.0) - best_metrics.get("wer_corrected", 1.0)
                delta_lev = best_metrics.get("lev_corrected", 0.0) - best_metrics.get("lev_original", 0.0)
                self.log_execution(f"Selected priority variant (index {best_idx}) with deltaLev={delta_lev:.3f}, deltaWER={delta_wer:.3f}, CorScore={best_metrics.get('cor_score', 0):.3f}")
            else:
                # Fallback to best by CorScore
                best_idx = max(range(len(basic_variants)), 
                              key=lambda i: basic_metrics[i].get("cor_score", 0))
                best_metrics = basic_metrics[best_idx]
                self.log_execution(f"No priority variants found, selected best by CorScore (index {best_idx}) with CorScore={best_metrics.get('cor_score', 0):.3f}")
            
            best_variant = basic_variants[best_idx]
        else:
            # Fallback to corrected_text if no basic variants
            best_variant = corrected_text
            best_metrics = self.metrics_calculator.calculate_correction_metrics(
                original_text=input_text,
                corrected_text=best_variant,
                reference_text=reference_text
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
        aggregated_text = await self._generate_aggregated_text(
            best_variant=best_variant,
            alternatives=alternatives
        )
        
        if not aggregated_text:
            self.log_execution("Не удалось сгенерировать агрегированный текст")
            self.add_to_state(state,
                           aggregator_used=False,
                           aggregation_reason="generation_failed")
            return state
        
        # Оценка агрегированного варианта
        aggregated_metrics = self.metrics_calculator.calculate_correction_metrics(
            original_text=input_text,
            corrected_text=aggregated_text,
            reference_text=reference_text
        )
        
        # Сравнение и выбор финального варианта
        final_text, final_metrics, used_aggregation = self._select_final_variant(
            best_variant=best_variant,
            best_metrics=best_metrics,
            aggregated_text=aggregated_text,
            aggregated_metrics=aggregated_metrics,
            input_text=input_text
        )
        
        # Обновление состояния
        self.add_to_state(state,
                         corrected_text=final_text,
                         metrics_correction=final_metrics,
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
    
    async def _generate_aggregated_text(self,
                                    best_variant: str,
                                    alternatives: List[str]) -> str:
        """
        Генерация улучшенного текста через LLM.
        
        Args:
            best_variant: Лучший вариант
            alternatives: Альтернативные варианты
            
        Returns:
            Агрегированный текст
        """
        try:
            # Формирование промпта агрегации
            alt_1 = alternatives[0] if len(alternatives) > 0 else ""
            alt_2 = alternatives[1] if len(alternatives) > 1 else ""
            
            aggregation_prompt = config.AGGREGATION_PROMPT.format(
                best_text=best_variant,
                alt_1=alt_1,
                alt_2=alt_2
            )
            
            # Генерация улучшенного текста
            aggregated_text = await self.lm_client.generate_with_retry(
                prompt=aggregation_prompt,
                temperature=config.AGGREGATION_TEMPERATURE,
                max_tokens=2048,
                system_prompt="Ты - эксперт по улучшению текстов. Создай лучший вариант, объединяя сильные стороны разных версий."
            )
            
            return aggregated_text.strip()
            
        except Exception as e:
            self.log_execution(f"Ошибка генерации агрегированного текста: {e}", "error")
            return ""
    
    def _select_final_variant(self,
                            best_variant: str,
                            best_metrics: Dict[str, float],
                            aggregated_text: str,
                            aggregated_metrics: Dict[str, float],
                            input_text: str) -> tuple:
        """
        Выбор финального варианта на основе метрик.
        
        Args:
            best_variant: Лучший вариант из ансамбля
            best_metrics: Метрики лучшего варианта
            aggregated_text: Агрегированный текст
            aggregated_metrics: Метрики агрегированного текста
            input_text: Исходный текст
            
        Returns:
            Кортеж (финальный текст, финальные метрики, использована ли агрегация)
        """
        # Расчет композитных оценок
        best_score = self._calculate_composite_score(best_metrics)
        aggregated_score = self._calculate_composite_score(aggregated_metrics)
        
        self.log_execution(f"Оценки: лучший={best_score:.3f}, агрегированный={aggregated_score:.3f}")
        
        # Проверка сходства с лучшим вариантом
        similarity = self._calculate_similarity(best_variant, aggregated_text)
        self.log_execution(f"Сходство с лучшим вариантом: {similarity:.3f}")
        
        # Проверка длины
        length_ratio = len(aggregated_text) / len(input_text) if len(input_text) > 0 else 0
        self.log_execution(f"Соотношение длины: {length_ratio:.3f}")
        
        # Принятие решения
        if (aggregated_score > best_score and 
            similarity >= config.MIN_SIMILARITY and 
            length_ratio >= 0.3):
            self.log_execution("Выбран агрегированный вариант")
            return aggregated_text, aggregated_metrics, True
        else:
            self.log_execution("Выбран лучший вариант из ансамбля")
            return best_variant, best_metrics, False
    
    def _calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """
        Расчет композитной оценки для сравнения вариантов.
        
        Args:
            metrics: Метрики варианта
            
        Returns:
            Композитная оценка
        """
        try:
            # Получаем исходные метрики для расчета CorScore
            original_wer = metrics.get("wer_original", 1.0)
            corrected_wer = metrics.get("wer_corrected", 1.0)
            original_lev = metrics.get("lev_original", 0.0)
            corrected_lev = metrics.get("lev_corrected", 0.0)
            perplexity = metrics.get("perplexity", float('inf'))
            
            # Используем единую формулу CorScore
            composite_score = self.metrics_calculator.calculate_cor_score(
                original_wer=original_wer,
                corrected_wer=corrected_wer,
                original_lev=original_lev,
                corrected_lev=corrected_lev,
                perplexity=perplexity
            )
            
            return composite_score
            
        except Exception as e:
            self.log_execution(f"Ошибка расчета композитной оценки: {e}", "error")
            return 0.0
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Расчет сходства Левенштейна между текстами.
        
        Args:
            text1: Первый текст
            text2: Второй текст
            
        Returns:
            Сходство в диапазоне [0, 1]
        """
        try:
            max_len = max(len(text1), len(text2))
            if max_len == 0:
                return 1.0
                
            distance = levenshtein_distance(text1, text2)
            similarity = 1 - (distance / max_len)
            
            return similarity
            
        except Exception as e:
            self.log_execution(f"Ошибка расчета сходства: {e}", "error")
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
