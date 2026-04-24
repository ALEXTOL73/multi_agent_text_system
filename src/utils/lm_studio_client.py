"""
Асинхронный клиент для взаимодействия с LM Studio API
"""

import aiohttp
import asyncio
import logging
from typing import Optional, Dict, Any
import json
from datetime import datetime

class LMStudioClient:
    """
    Асинхронный клиент для общения с локальным сервером LM Studio
    """
    
    def __init__(self, 
                 base_url: str = "http://localhost:1234/v1",
                 model_name: str = "gemma-3-12b-it",
                 max_retries: int = 3,
                 timeout: int = 300):
        """
        Инициализация клиента LM Studio.
        
        Args:
            base_url: Базовый URL LM Studio API
            model_name: Имя модели
            max_retries: Максимальное количество повторных попыток
            timeout: Таймаут запроса в секундах
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.max_retries = max_retries
        self.timeout = aiohttp.ClientTimeout(total=timeout, connect=30)
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger("lm_studio_client")
        
    async def __aenter__(self):
        """Контекстный менеджер для входа"""
        await self._ensure_session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Контекстный менеджер для выхода"""
        await self.close()
        
    async def _ensure_session(self):
        """Создание сессии если она не существует"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
            
    async def close(self):
        """Закрытие сессии"""
        if self.session and not self.session.closed:
            await self.session.close()
            
    async def generate(self, 
                      prompt: str, 
                      temperature: float = 0.7,
                      max_tokens: int = 2048,
                      system_prompt: Optional[str] = None) -> str:
        """
        Generate text using LM Studio.
        
        Args:
            prompt: Prompt for generation
            temperature: Generation temperature
            max_tokens: Maximum number of tokens
            system_prompt: System prompt (optional)
            
        Returns:
            Generated text
        """
        result = await self.generate_with_metadata(prompt, temperature, max_tokens, system_prompt)
        return result["text"]
    
    async def generate_with_metadata(self, 
                                   prompt: str, 
                                   temperature: float = 0.7,
                                   max_tokens: int = 2048,
                                   system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate text using LM Studio with metadata.
        
        Args:
            prompt: Prompt for generation
            temperature: Generation temperature
            max_tokens: Maximum number of tokens
            system_prompt: System prompt (optional)
            
        Returns:
            Dictionary with 'text' and 'metadata' including request parameters
        """
        # Use context manager for each request to ensure proper cleanup
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            # Check connection first
            try:
                async with session.get(f"{self.base_url}/models", timeout=10) as response:
                    if response.status != 200:
                        raise Exception(f"LM Studio not responding: status {response.status}")
            except Exception as e:
                self.logger.error(f"LM Studio connection check failed: {e}")
                raise Exception(f"LM Studio is not available: {e}")
            
            for attempt in range(self.max_retries):
                try:
                    # Формирование запроса
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": prompt})
                    
                    # Add random seed to ensure variability with temperature
                    import random
                    seed = random.randint(1, 10000)
                    
                    payload = {
                        "model": self.model_name,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "stream": False,
                        "seed": seed
                    }
                    
                    self.logger.debug(f"LM Studio запрос (попытка {attempt + 1}): temp={temperature}, seed={seed}, max_tokens={max_tokens}")
                    
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            generated_text = result["choices"][0]["message"]["content"].strip()
                            self.logger.debug(f"Успешный ответ от LM Studio: {generated_text[:100]}...")
                            
                            # Add metadata to response
                            metadata = {
                                "temperature": temperature,
                                "seed": seed,
                                "max_tokens": max_tokens,
                                "model": self.model_name
                            }
                            
                            return {
                                "text": generated_text,
                                "metadata": metadata
                            }
                        else:
                            error_text = await response.text()
                            self.logger.warning(f"Ошибка LM Studio (статус {response.status}): {error_text}")
                            
                except asyncio.TimeoutError:
                    self.logger.warning(f"Таймаут запроса к LM Studio (попытка {attempt + 1})")
                except aiohttp.ClientError as e:
                    self.logger.warning(f"Ошибка клиента LM Studio (попытка {attempt + 1}): {e}")
                except Exception as e:
                    self.logger.error(f"Неожиданная ошибка (попытка {attempt + 1}): {e}")
                    
                # Пауза между попытками
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Экспоненциальная задержка
                    
            raise Exception(f"Не удалось получить ответ от LM Studio после {self.max_retries} попыток")
        
    async def generate_with_retry(self, 
                                 prompt: str, 
                                 temperature: float = 0.7,
                                 max_tokens: int = 2048,
                                 system_prompt: Optional[str] = None) -> str:
        """
        Генерация текста с автоматическими повторными попытками.
        
        Args:
            prompt: Промпт для генерации
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов
            system_prompt: Системный промпт (опционально)
            
        Returns:
            Сгенерированный текст
        """
        return await self.generate(prompt, temperature, max_tokens, system_prompt)
        
    async def test_connection(self) -> bool:
        """
        Проверка соединения с LM Studio.
        
        Returns:
            True если соединение установлено, иначе False
        """
        try:
            await self._ensure_session()
            
            # Пробуем получить список моделей
            async with self.session.get(f"{self.base_url}/models") as response:
                if response.status == 200:
                    models = await response.json()
                    self.logger.info(f"Соединение с LM Studio установлено. Доступные модели: {models}")
                    return True
                else:
                    self.logger.error(f"Не удалось получить список моделей. Статус: {response.status}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Ошибка проверки соединения с LM Studio: {e}")
            return False
        finally:
            # Закрываем сессию после проверки соединения
            if self.session and not self.session.closed:
                await self.session.close()
            
    async def get_model_info(self) -> Dict[str, Any]:
        """
        Получение информации о текущей модели.
        
        Returns:
            Словарь с информацией о модели
        """
        try:
            await self._ensure_session()
            
            async with self.session.get(f"{self.base_url}/models") as response:
                if response.status == 200:
                    models_data = await response.json()
                    # Ищем нашу модель в списке
                    for model in models_data.get("data", []):
                        if self.model_name in model.get("id", ""):
                            return model
                            
            return {"error": f"Модель {self.model_name} не найдена"}
            
        except Exception as e:
            self.logger.error(f"Ошибка получения информации о модели: {e}")
            return {"error": str(e)}
