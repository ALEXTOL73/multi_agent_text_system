"""
Судья коррекции для итоговой оценки качества исправления текста
"""

from typing import Dict, Any

from .base_agent import BaseAgent
import config

class CorrectionJudge(BaseAgent):
    """
    Судья коррекции для итоговой оценки качества и принятия решения о повторе
    """
    
    def __init__(self):
        """
        Инициализация судьи коррекции.
        """
        super().__init__("CorrectionJudge")
        
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Основной метод выполнения агента.
        
        Args:
            state: Текущее состояние системы
            
        Returns:
            Обновленное состояние системы
        """
        self.log_execution("Начало оценки качества коррекции")
        
        # Проверка обязательных ключей
        if not self.validate_state(state, ["input_text", "corrected_text", "metrics_correction"]):
            return state
            
        input_text = self.get_from_state(state, "input_text")
        corrected_text = self.get_from_state(state, "corrected_text")
        metrics = self.get_from_state(state, "metrics_correction", {})
        
        # Получение текущего количества попыток
        correction_attempts = self.get_from_state(state, "correction_attempts", 0)
        
        # Оценка качества
        quality_assessment = self._assess_quality(metrics)
        correction_quality = quality_assessment["quality"]
        retry_needed = quality_assessment["retry_needed"]
        
        # Проверка лимита попыток
        if retry_needed and correction_attempts >= config.MAX_CORRECTION_ATTEMPTS:
            retry_needed = False
            self.log_execution(f"Достигнут лимит попыток ({config.MAX_CORRECTION_ATTEMPTS}), повтор невозможен")
        
        # Логирование результатов
        self.log_execution(f"Оценка качества: {correction_quality}")
        self.log_execution(f"Потребуется повтор: {'да' if retry_needed else 'нет'}")
        
        # Обновление состояния
        self.add_to_state(state,
                         correction_quality=correction_quality,
                         correction_retry_needed=retry_needed,
                         correction_attempts=correction_attempts + 1)
        
        return state
    
    def _assess_quality(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Оценка качества на основе метрик.
        
        Args:
            metrics: Метрики коррекции
            
        Returns:
            Словарь с оценкой качества и решением о повторе
        """
        delta_wer = metrics.get("delta_wer", 0)
        delta_lev = metrics.get("delta_lev", 0)
        lev_rating = metrics.get("lev_corrected", metrics.get("lev_rating", 0))
        cor_score = metrics.get("cor_score", 0)
        
        # Новая логика определения качества на основе LevRating и улучшений
        has_improvements = delta_wer > 0 or delta_lev > 0
        
        if has_improvements and lev_rating > 0.95:
            quality = "ОТЛИЧНО"
            retry_needed = False
        elif has_improvements and 0.85 <= lev_rating <= 0.95:
            quality = "ХОРОШО"
            retry_needed = False
        elif has_improvements and 0.75 <= lev_rating <= 0.85:
            quality = "УДОВЛЕТВОРИТЕЛЬНО"
            retry_needed = False
        elif has_improvements and lev_rating < 0.75:
            quality = "ТРЕБУЕТСЯ УЛУЧШЕНИЕ"
            retry_needed = True
        elif not has_improvements and lev_rating > 0.75:
            quality = "ТРЕБУЕТСЯ УЛУЧШЕНИЕ"
            retry_needed = True
        else:  # not has_improvements and lev_rating < 0.75
            quality = "ПЛОХО"
            retry_needed = True
        
        # Логирование метрик для анализа
        self.log_execution(f"Метрики: ΔWER={delta_wer:.3f}, ΔLev={delta_lev:.3f}, CorScore={cor_score:.3f}")
        
        return {
            "quality": quality,
            "retry_needed": retry_needed,
            "delta_wer": delta_wer,
            "delta_lev": delta_lev,
            "cor_score": cor_score
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
            "ОТЛИЧНО": "Текст исправлен практически идеально, ошибки практически отсутствуют",
            "ХОРОШО": "Текст исправлен хорошо, остались незначительные ошибки",
            "УДОВЛЕТВОРИТЕЛЬНО": "Текст исправлен частично, остались заметные ошибки",
            "ТРЕБУЕТСЯ УЛУЧШЕНИЕ": "Качество исправления низкое, требуется повторная коррекция"
        }
        return descriptions.get(quality, "Неизвестное качество")
