"""
Агент генерации промптов для коррекции текста
"""

import asyncio
from typing import Dict, Any, List
import re

from .base_agent import BaseAgent
from ..utils.lm_studio_client import LMStudioClient
from ..utils.agent_memory import AgentMemory
import config

class CorrectionPromptGenerator(BaseAgent):
    """
    Агент для генерации промптов коррекции текста
    """
    
    def __init__(self, memory: AgentMemory, lm_client: LMStudioClient):
        """
        Инициализация агента.
        
        Args:
            memory: Система памяти агентов
            lm_client: Клиент LM Studio
        """
        super().__init__("CorrectionPromptGenerator")
        self.memory = memory
        self.lm_client = lm_client
        
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Основной метод выполнения агента.
        
        Args:
            state: Текущее состояние системы
            
        Returns:
            Обновленное состояние системы
        """
        self.log_execution("Начало генерации промптов коррекции")
        
        # Проверка обязательных ключей
        if not self.validate_state(state, ["input_text"]):
            return state
            
        input_text = self.get_from_state(state, "input_text")
        domain = self.get_from_state(state, "domain", "general")
        
        # Определение языка
        detected_language = self._detect_language(input_text)
        self.log_execution(f"Определен язык: {detected_language}")
        
        # Получение few-shot примеров
        few_shot_examples = []
        if config.DYNAMIC_FEW_SHOT_ENABLED:
            few_shot_examples = self.memory.get_few_shot_examples(
                input_text=input_text,
                domain=domain,
                max_examples=config.MAX_FEW_SHOT_EXAMPLES,
                similarity_threshold=config.FEW_SHOT_SIMILARITY_THRESHOLD
            )
            self.log_execution(f"Получено {len(few_shot_examples)} few-shot примеров")
        
        # Формирование пользовательского промпта
        user_prompt = self._build_user_prompt(input_text, detected_language, few_shot_examples)
        
        # Генерация вариантов промптов
        prompt_variants = await self._generate_prompt_variants(user_prompt)
        
        # Выбор основного промпта
        main_prompt = prompt_variants[0] if prompt_variants else self._get_default_prompt(detected_language)
        
        # Добавление плейсхолдера если отсутствует
        if "{text}" not in main_prompt:
            main_prompt += "\n\n{text}"
            
        # Обновление состояния
        self.add_to_state(state,
                         prompt_correction_variants=prompt_variants,
                         prompt_correction=main_prompt,
                         detected_language=detected_language,
                         few_shot_examples_used=len(few_shot_examples))
        
        self.log_execution(f"Сгенерировано {len(prompt_variants)} вариантов промптов")
        
        return state
        
    def _detect_language(self, text: str) -> str:
        """
        Определение языка текста по наличию кириллицы.
        
        Args:
            text: Входной текст
            
        Returns:
            'ru' или 'en'
        """
        # Проверяем наличие кириллицы
        cyrillic_pattern = re.compile(r'[а-яёА-ЯЁ]')
        if cyrillic_pattern.search(text):
            return "ru"
        return "en"
        
    def _build_user_prompt(self, input_text: str, language: str, few_shot_examples: List[Dict]) -> str:
        """
        Формирование пользовательского промпта для генерации промпта коррекции.
        
        Args:
            input_text: Входной текст
            language: Определенный язык
            few_shot_examples: Few-shot примеры
            
        Returns:
            Пользовательский промпт
        """
        prompt_parts = []
        
        # Добавляем статические примеры
        if language == "ru":
            prompt_parts.append("Создай эффективный промпт для исправления ошибок в русском тексте.")
            prompt_parts.append("Примеры хороших промптов:")
            
            for i, example in enumerate(config.STATIC_CORRECTION_EXAMPLES, 1):
                prompt_parts.append(f"{i}. Вход: '{example['input']}' -> Выход: '{example['output']}'")
        else:
            prompt_parts.append("Create an effective prompt for correcting errors in English text.")
            prompt_parts.append("Examples of good prompts:")
            
            for i, example in enumerate(config.STATIC_CORRECTION_EXAMPLES, 1):
                prompt_parts.append(f"{i}. Input: '{example['input']}' -> Output: '{example['output']}'")
        
        # Добавляем динамические few-shot примеры
        if few_shot_examples:
            prompt_parts.append(f"\nДополнительные релевантные примеры:")
            for i, example in enumerate(few_shot_examples, len(config.STATIC_CORRECTION_EXAMPLES) + 1):
                if language == "ru":
                    prompt_parts.append(f"{i}. Вход: '{example['input']}' -> Выход: '{example['output']}'")
                else:
                    prompt_parts.append(f"{i}. Input: '{example['input']}' -> Output: '{example['output']}'")
        
        # Добавляем инструкции
        if language == "ru":
            prompt_parts.append(f"""
            
            Текст для исправления: '{input_text}'
            
            Создай промпт, который поможет языковой модели исправить орфографические, грамматические и пунктуационные ошибки в этом тексте.
            Промпт должен:
            1. Быть четким и конкретным
            2. Сохранять смысл и стиль текста
            3. Включать инструкции по исправлению ошибок
            4. Содержать плейсхолдер {input_text} для вставки текста
            
            Верни только текст промпта без дополнительных комментариев.
            """)
        else:
            prompt_parts.append(f"""
            
            Text to correct: '{input_text}'
            
            Create a prompt that will help a language model correct spelling, grammar, and punctuation errors in this text.
            The prompt should:
            1. Be clear and specific
            2. Preserve meaning and style
            3. Include correction instructions
            4. Content placeholder {input_text} for text insertion
            
            Return only the prompt text without additional comments.
            """)
        
        return "\n".join(prompt_parts)
        
    async def _generate_prompt_variants(self, user_prompt: str) -> List[str]:
        """
        Генерация вариантов промптов с разными температурами.
        
        Args:
            user_prompt: Пользовательский промпт
            
        Returns:
            Список вариантов промптов
        """
        variants = []
        temperatures = [0.3, 0.6, 0.9]
        
        # Different system prompts for each temperature to ensure variety
        system_prompts = [
            "Generate a minimal correction prompt. Focus ONLY on basic spelling and grammar fixes. Keep it under 50 words. Use {input_text} placeholder.",
            "Generate a style-focused correction prompt. Emphasize readability, flow, and professional tone. Include punctuation rules. Avoid listing basic errors. Use {input_text} placeholder.", 
            "Generate a technical editing prompt. Focus on formatting, consistency, and coherence. Mention specific elements like numbers, abbreviations, and structure. Use {input_text} placeholder."
        ]
        
        tasks = []
        for i, temp in enumerate(temperatures):
            task = self.lm_client.generate_with_retry(
                prompt=user_prompt,
                temperature=temp,
                max_tokens=512,
                system_prompt=system_prompts[i]
            )
            tasks.append(task)
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.log_execution(f"Ошибка генерации промпта с температурой {temperatures[i]}: {result}", "warning")
                    continue
                    
                # Очистка результата
                cleaned_prompt = self._clean_prompt(result)
                if cleaned_prompt:
                    variants.append(cleaned_prompt)
                    
        except Exception as e:
            self.log_execution(f"Ошибка при генерации вариантов промптов: {e}", "error")
        
        return variants
        
    def _clean_prompt(self, prompt: str) -> str:
        """
        Очистка сгенерированного промпта.
        
        Args:
            prompt: Исходный промпт
            
        Returns:
            Очищенный промпт
        """
        # Удаляем лишние пробелы и переносы строк
        prompt = re.sub(r'\s+', ' ', prompt.strip())
        
        # Удаляем кавычки если они есть
        if prompt.startswith('"') and prompt.endswith('"'):
            prompt = prompt[1:-1]
        elif prompt.startswith("'") and prompt.endswith("'"):
            prompt = prompt[1:-1]
            
        return prompt
        
    def _get_default_prompt(self, language: str) -> str:
        """
        Получение промпта по умолчанию.
        
        Args:
            language: Язык
            
        Returns:
            Промпт по умолчанию
        """
        if language == "ru":
            return """Исправь орфографические, грамматические и пунктуационные ошибки в следующем тексте. Сохраняй исходный смысл и стиль.

{input_text}

Верни только исправленный текст без дополнительных комментариев."""
        else:
            return """Correct spelling, grammar, and punctuation errors in the following text. Preserve the original meaning and style.

{input_text}

Return only the corrected text without additional comments."""
