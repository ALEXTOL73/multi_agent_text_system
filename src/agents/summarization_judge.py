"""
Судья суммаризации для итоговой оценки качества суммаризации текста
"""

from typing import Dict, Any

from .base_agent import BaseAgent
import config

class SummarizationJudge(BaseAgent):
    """
    Судья суммаризации для итоговой оценки качества и принятия решения о повторе
    """
    
    def __init__(self):
        """
        Инициализация судьи суммаризации.
        """
        super().__init__("SummarizationJudge")
        
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Основной метод выполнения агента.
        
        Args:
            state: Текущее состояние системы
            
        Returns:
            Обновленное состояние системы
        """
        self.log_execution("Начало оценки качества суммаризации")
        
        # Проверка обязательных ключей
        if not self.validate_state(state, ["corrected_text", "summary", "summary_metrics"]):
            return state
            
        input_text = self.get_from_state(state, "corrected_text")
        summary_text = self.get_from_state(state, "summary")
        metrics = self.get_from_state(state, "summary_metrics", {})
        
        # Получение текущего количества попыток
        summary_attempts = self.get_from_state(state, "summary_attempts", 0)
        
        # Оценка качества
        quality_assessment = self._assess_quality(metrics, input_text, summary_text)
        summary_quality = quality_assessment["quality"]
        retry_needed = quality_assessment["retry_needed"]
        
        # Проверка лимита попыток
        if retry_needed and summary_attempts >= config.SUMMARIZATION_ATTEMPTS:
            retry_needed = False
            self.log_execution(f"Достигнут лимит попыток ({config.SUMMARIZATION_ATTEMPTS}), повтор невозможен")
        
        # Логирование результатов
        self.log_execution(f"Оценка качества: {summary_quality}")
        self.log_execution(f"Потребуется повтор: {'да' if retry_needed else 'нет'}")
        
        # Обновление состояния
        self.add_to_state(state,
                         summary_quality=summary_quality,
                         summary_retry_needed=retry_needed,
                         summary_attempts=summary_attempts + 1)
        
        return state
    
    def _assess_quality(self, 
                        metrics: Dict[str, float],
                        input_text: str,
                        summary_text: str) -> Dict[str, Any]:
        """
        Оценка качества на основе метрик.
        
        Args:
            metrics: Метрики суммаризации
            input_text: Исходный текст
            summary_text: Суммаризация
            
        Returns:
            Словарь с оценкой качества и решением о повторе
        """
        sum_score = metrics.get("sum_score", 0)
        geval = metrics.get("geval", 0)
        llm_judge = metrics.get("llm_judge", 5)
        meteor = metrics.get("meteor", 0)
        bert_score = metrics.get("bert_score", 0)
        
        # Определение качества на основе порогов SumScore
        thresholds = config.SUMMARY_THRESHOLDS
        
        if sum_score >= thresholds["excellent"]:
            quality = "ОТЛИЧНО"
            retry_needed = False
        elif sum_score >= thresholds["good"]:
            quality = "ХОРОШО"
            retry_needed = False
        elif sum_score >= thresholds["satisfactory"]:
            quality = "УДОВЛЕТВОРИТЕЛЬНО"
            retry_needed = False
        else:
            quality = "ТРЕБУЕТСЯ УЛУЧШЕНИЕ"
            retry_needed = True
        
        # Дополнительные проверки
        # Проверка длины суммаризации
        length_ratio = len(summary_text) / len(input_text) if len(input_text) > 0 else 0
        if length_ratio < 0.05:  # Слишком короткая
            quality = "ТРЕБУЕТСЯ УЛУЧШЕНИЕ"
            retry_needed = True
        elif length_ratio > 0.9:  # Слишком длинная
            quality = "ТРЕБУЕТСЯ УЛУЧШЕНИЕ"
            retry_needed = True
        
        # Проверка на пустую суммаризацию
        if len(summary_text.strip()) < 10:
            quality = "ТРЕБУЕТСЯ УЛУЧШЕНИЕ"
            retry_needed = True
        
        # Проверка на копирование оригинала
        if length_ratio > 0.8 and meteor < 0.3:
            quality = "ТРЕБУЕТСЯ УЛУЧШЕНИЕ"
            retry_needed = True
        
        # Улучшение оценки если отдельные метрики очень хорошие
        if (geval > 0.8 and llm_judge > 7 and 
            quality in ["УДОВЛЕТВОРИТЕЛЬНО", "ТРЕБУЕТСЯ УЛУЧШЕНИЕ"]):
            quality = "ХОРОШО"
            retry_needed = False
        
        # Логирование метрик для анализа
        self.log_execution(f"Метрики: SumScore={sum_score:.3f}, G-Eval={geval:.3f}, "
                         f"LLM-Judge={llm_judge:.1f}, METEOR={meteor:.3f}, "
                         f"BertScore={bert_score:.3f}, LengthRatio={length_ratio:.3f}")
        
        return {
            "quality": quality,
            "retry_needed": retry_needed,
            "sum_score": sum_score,
            "geval": geval,
            "llm_judge": llm_judge,
            "meteor": meteor,
            "bert_score": bert_score,
            "length_ratio": length_ratio
        }
    
    def get_quality_color(self, quality: str) -> str:
        """
        Получение цвета для отображения качества в интерфейсе.
        
        Args:
            quality: Оценка качества
            
        Returns:
            Цвет в формате hex
        """
        color_map = {
            "ОТЛИЧНО": "#4CAF50",      # Зеленый
            "ХОРОШО": "#8BC34A",       # Светло-зеленый
            "УДОВЛЕТВОРИТЕЛЬНО": "#FF9800",  # Оранжевый
            "ТРЕБУЕТСЯ УЛУЧШЕНИЕ": "#F44336"  # Красный
        }
        return color_map.get(quality, "#9E9E9E")  # Серый по умолчанию
    
    def get_quality_description(self, quality: str) -> str:
        """
        Получение описания оценки качества.
        
        Args:
            quality: Оценка качества
            
        Returns:
            Описание оценки
        """
        descriptions = {
            "ОТЛИЧНО": "Суммаризация практически идеальна, точно передает основное содержание",
            "ХОРОШО": "Суммаризация хорошего качества, передает основные идеи",
            "УДОВЛЕТВОРИТЕЛЬНО": "Суммаризация приемлемого качества, но можно улучшить",
            "ТРЕБУЕТСЯ УЛУЧШЕНИЕ": "Качество суммаризации низкое, требуется повторная обработка"
        }
        return descriptions.get(quality, "Неизвестное качество")
    
    def get_improvement_suggestions(self, 
                                   metrics: Dict[str, float],
                                   input_text: str,
                                   summary_text: str) -> list:
        """
        Получение предложений по улучшению суммаризации.
        
        Args:
            metrics: Метрики суммаризации
            input_text: Исходный текст
            summary_text: Суммаризация
            
        Returns:
            Список предложений по улучшению
        """
        suggestions = []
        
        sum_score = metrics.get("sum_score", 0)
        length_ratio = len(summary_text) / len(input_text) if len(input_text) > 0 else 0
        geval = metrics.get("geval", 0)
        meteor = metrics.get("meteor", 0)
        
        if length_ratio < 0.1:
            suggestions.append("Суммаризация слишком короткая, нужно добавить больше деталей")
        elif length_ratio > 0.8:
            suggestions.append("Суммаризация слишком длинная, нужно сделать ее более краткой")
        
        if geval < 0.5:
            suggestions.append("Нужно улучшить точность передачи содержания")
        
        if meteor < 0.3:
            suggestions.append("Нужно улучшить лексическое сходство с оригиналом")
        
        if sum_score < 0.3:
            suggestions.append("Требуется значительное улучшение качества суммаризации")
        
        return suggestions
