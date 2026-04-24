"""
Базовый класс для всех агентов в мультиагентной системе CorSumAgentsAI
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
from datetime import datetime

class BaseAgent(ABC):
    """
    Абстрактный базовый класс для всех агентов.
    Каждый агент должен реализовать метод execute().
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация агента.
        
        Args:
            name: Имя агента для логирования
            config: Конфигурация агента (опционально)
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"agent.{name}")
        
    @abstractmethod
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Основной метод выполнения агента.
        
        Args:
            state: Текущее состояние системы
            
        Returns:
            Обновленное состояние системы
        """
        pass
    
    def log_execution(self, message: str, level: str = "info"):
        """
        Логирование выполнения агента.
        
        Args:
            message: Сообщение для логирования
            level: Уровень логирования (info, warning, error, debug)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {self.name}: {message}"
        
        if level == "info":
            self.logger.info(log_message)
        elif level == "warning":
            self.logger.warning(log_message)
        elif level == "error":
            self.logger.error(log_message)
        elif level == "debug":
            self.logger.debug(log_message)
        else:
            self.logger.info(log_message)
    
    def validate_state(self, state: Dict[str, Any], required_keys: list) -> bool:
        """
        Проверка наличия необходимых ключей в состоянии.
        
        Args:
            state: Текущее состояние
            required_keys: Список обязательных ключей
            
        Returns:
            True если все ключи присутствуют, иначе False
        """
        missing_keys = [key for key in required_keys if key not in state]
        if missing_keys:
            self.log_execution(f"Отсутствуют обязательные ключи: {missing_keys}", "error")
            return False
        return True
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Получение значения из конфигурации агента.
        
        Args:
            key: Ключ конфигурации
            default: Значение по умолчанию
            
        Returns:
            Значение из конфигурации или default
        """
        return self.config.get(key, default)
    
    def add_to_state(self, state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Добавление значений в состояние.
        
        Args:
            state: Текущее состояние
            **kwargs: Пары ключ-значение для добавления
            
        Returns:
            Обновленное состояние
        """
        state.update(kwargs)
        return state
    
    def get_from_state(self, state: Dict[str, Any], key: str, default: Any = None) -> Any:
        """
        Получение значения из состояния.
        
        Args:
            state: Текущее состояние
            key: Ключ для получения
            default: Значение по умолчанию
            
        Returns:
            Значение из состояния или default
        """
        return state.get(key, default)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __repr__(self) -> str:
        return self.__str__()
