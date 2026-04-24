"""
Агент генерации промптов для суммаризации текста
"""

import asyncio
from typing import Dict, Any, List
import re

from .base_agent import BaseAgent
from ..utils.lm_studio_client import LMStudioClient
from ..utils.agent_memory import AgentMemory
import config

class SummarizationPromptGenerator(BaseAgent):
    """
    Агент для генерации промптов суммаризации текста
    """
    
    def __init__(self, memory: AgentMemory, lm_client: LMStudioClient):
        """
        Инициализация агента.
        
        Args:
            memory: Система памяти агентов
            lm_client: Клиент LM Studio
        """
        super().__init__("SummarizationPromptGenerator")
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
        self.log_execution("Начало генерации промптов суммаризации")
        
        # Проверка обязательных ключей
        if not self.validate_state(state, ["corrected_text"]):
            return state
            
        input_text = self.get_from_state(state, "corrected_text")
        domain = self.get_from_state(state, "domain", "general")
        
        # Detect language from corrected text
        detected_language = self._detect_language(input_text)
        
        # Получение few-shot примеров
        few_shot_examples = []
        if config.SUMMARY_DYNAMIC_FEW_SHOT_ENABLED:
            few_shot_examples = self.memory.get_summary_few_shot_examples(
                input_text=input_text,
                domain=domain,
                max_examples=config.SUMMARY_MAX_FEW_SHOT_EXAMPLES,
                similarity_threshold=config.FEW_SHOT_SIMILARITY_THRESHOLD
            )
        
        # Add static examples for detected language
        static_examples = config.STATIC_SUMMARY_EXAMPLES
        few_shot_examples.extend(static_examples)
        
        self.log_execution(f"Получено {len(few_shot_examples)} few-shot примеров суммаризации")
        
        # Формирование пользовательского промпта
        user_prompt = self._build_user_prompt(input_text, detected_language, few_shot_examples)
        
        # Генерация вариантов промптов
        prompt_variants = await self._generate_prompt_variants(user_prompt, detected_language)
        
        # Выбор основного промпта
        main_prompt = prompt_variants[0] if prompt_variants else self._get_default_prompt(detected_language)
        
        # Добавление плейсхолдера если отсутствует
        if "{input_text}" not in main_prompt:
            main_prompt += "\n\n{input_text}"
            
        # Обновление состояния
        self.add_to_state(state,
                         prompt_summary_variants=prompt_variants,
                         prompt_summary=main_prompt,
                         summary_few_shot_examples_used=len(few_shot_examples))
        
        self.log_execution(f"Сгенерировано {len(prompt_variants)} вариантов промптов суммаризации")
        
        return state
        
    def _detect_language(self, text: str) -> str:
        """
        Detect language of text by checking for Cyrillic characters.
        
        Args:
            text: Input text
            
        Returns:
            'ru' or 'en'
        """
        # Check for Cyrillic characters (more comprehensive pattern)
        cyrillic_pattern = re.compile(r'[а-яёА-ЯЁ]')
        latin_pattern = re.compile(r'[a-zA-Z]')
        
        # Count characters
        cyrillic_count = len(cyrillic_pattern.findall(text))
        latin_count = len(latin_pattern.findall(text))
        
        # Determine language based on character distribution
        if cyrillic_count > latin_count * 0.3:  # If 30%+ Cyrillic
            return "ru"
        elif latin_count > cyrillic_count * 0.3:  # If 30%+ Latin
            return "en"
        else:  # Mixed content - prefer the language of reference if available
            return "en"  # Default to English for mixed content
        
    def _build_user_prompt(self, input_text: str, detected_language: str, few_shot_examples: List[Dict]) -> str:
        """
        Формирование пользовательского промпта для генерации промпта суммаризации.
        
        Args:
            input_text: Входной текст
            language: Определенный язык
            few_shot_examples: Few-shot примеры
            
        Returns:
            Пользовательский промпт
        """
        prompt_parts = []
        
        # Добавляем статические примеры
        static_examples = config.STATIC_SUMMARY_EXAMPLES
        if static_examples:
            if detected_language == "ru":
                prompt_parts.append("Создай эффективный промпт для суммаризации русского текста.")
                prompt_parts.append("Резюме должно содержать не более 5 предложений.")
                prompt_parts.append("Примеры хороших промптов:")
                
                for i, example in enumerate(static_examples, 1):
                    prompt_parts.append(f"{i}. Вход: '{example['input'][:100]}...' -> Выход: '{example['output']}'")
            else:
                prompt_parts.append("Create an effective prompt for text summarization.")
                prompt_parts.append("Summary should contain 3-4 sentences.")
                prompt_parts.append("Examples of good prompts:")
                
                for i, example in enumerate(static_examples, 1):
                    prompt_parts.append(f"{i}. Input: '{example['input'][:100]}...' -> Output: '{example['output']}'")
        else:
            if detected_language == "ru":
                prompt_parts.append("Создай эффективный промпт для суммаризации русского текста.")
                prompt_parts.append("Резюме должно содержать не более 5 предложений.")
                prompt_parts.append("Примеры хороших промптов:")
            else:
                prompt_parts.append("Create an effective prompt for text summarization.")
                prompt_parts.append("Summary should contain 3-4 sentences.")
                prompt_parts.append("Examples of good prompts:")
                
            for i, example in enumerate(static_examples, 1):
                if detected_language == "ru":
                    prompt_parts.append(f"{i}. Вход: '{example['input'][:100]}...' -> Выход: '{example['output']}'")
                else:
                    prompt_parts.append(f"{i}. Input: '{example['input'][:100]}...' -> Output: '{example['output']}'")
        
        # Добавляем динамические few-shot примеры
        if few_shot_examples:
            prompt_parts.append(f"\nДополнительные релевантные примеры:")
            for i, example in enumerate(few_shot_examples, len(static_examples) + 1):
                if detected_language == "ru":
                    prompt_parts.append(f"{i}. Вход: '{example['input'][:100]}...' -> Выход: '{example['output']}'")
                else:
                    prompt_parts.append(f"{i}. Input: '{example['input'][:100]}...' -> Output: '{example['output']}'")
        
        # Определение целевой длины суммаризации
        target_length = int(len(input_text) * config.SUMMARY_FEW_SHOT_LENGTH_RATIO)
        
        # Добавляем инструкции
        if detected_language == "ru":
            prompt_parts.append(f"""
            
            Текст для суммаризации: '{input_text[:200]}...'
            
            Создай промпт, который поможет языковой модели создать краткое, но информативное изложение основного содержания этого текста.
            Промпт должен:
            1. Быть четким и конкретным
            2. Сохранять ключевые идеи и факты
            3. Создавать суммаризацию ровно из 5 предложений
            4. Содержать плейсхолдер {input_text} для вставки текста
            5. НИКОГДА не упоминать ограничения по количеству символов, знаков или байт
            
            Верни только текст промпта без дополнительных комментариев.
            """)
        else:
            prompt_parts.append(f"""
            
            Text to summarize: '{input_text[:200]}...'
            
            Create a prompt that will help a language model create a concise but informative summary of this text's main content.
            The prompt should:
            1. Be clear and specific
            2. Preserve key ideas and facts
            3. Create a summary of 3-4 sentences
            4. Include placeholder {input_text} for text insertion
            5. NEVER mention character limits, symbols, or byte counts
            
            Return only the prompt text without additional comments.
            """)
        
        return "\n".join(prompt_parts)
        
# ... (rest of the code remains the same)
    async def _generate_prompt_variants(self, user_prompt: str, detected_language: str) -> List[str]:
        """
        Генерация вариантов промптов с разными температурами.
        
        Args:
            user_prompt: Пользовательский промпт
            detected_language: Определенный язык текста
            
        Returns:
            Список вариантов промптов
        """
        variants = []
        temperatures = [0.3, 0.6, 0.9]
        
        # Different system prompts for each temperature to ensure variety
        # Use language-appropriate system prompts
        if detected_language == "ru":
            system_prompts = [
                "You are an expert at creating effective summarization prompts in RUSSIAN. Generate a clear, concise prompt that instructs to create a summary in 3-4 sentences in RUSSIAN. Focus on extracting key information. The prompt should use {input_text} as placeholder. NEVER mention character limits or symbols. OUTPUT MUST BE IN RUSSIAN.",
                "You are a specialist in prompt engineering for text summarization in RUSSIAN. Create a comprehensive prompt that generates summaries in 3-4 sentences in RUSSIAN. Emphasize thorough coverage of main points. Use {input_text} placeholder. ABSOLUTELY NO character limits mentioned. OUTPUT MUST BE IN RUSSIAN.", 
                "You are a master prompt designer for summarization tasks in RUSSIAN. Generate a balanced prompt that creates summaries in 3-4 sentences in RUSSIAN. Combine conciseness with completeness. Use {input_text} placeholder. NEVER reference character counts or symbols. OUTPUT MUST BE IN RUSSIAN."
            ]
        else:
            system_prompts = [
                "You are an expert at creating effective summarization prompts. Generate a clear, concise prompt that instructs to create a summary in 3-4 sentences. Focus on extracting key information. The prompt should use {input_text} as placeholder. NEVER mention character limits or symbols.",
                "You are a specialist in prompt engineering for text summarization. Create a comprehensive prompt that generates summaries in 3-4 sentences. Emphasize thorough coverage of main points. Use {input_text} placeholder. ABSOLUTELY NO character limits mentioned.", 
                "You are a master prompt designer for summarization tasks. Generate a balanced prompt that creates summaries in 3-4 sentences. Combine conciseness with completeness. Use {input_text} placeholder. NEVER reference character counts or symbols."
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
    
    tasks = []
    for temp in temperatures:
        task = self.lm_client.generate_with_retry(
            prompt=user_prompt,
            temperature=temp,
            max_tokens=512,
            system_prompt=config.PROMPT_GENERATION_SYSTEM_PROMPT
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
        
    def _get_default_prompt(self, detected_language: str) -> str:
        """
        Get default prompt template with placeholder.
        
        Args:
            detected_language: Detected language ('ru' or 'en')
            
        Returns:
            Default prompt template with {input_text} placeholder
        """
        if detected_language == "en":
            return """You are an expert at creating concise summaries of texts. Your task is to create a brief but informative summary of the main content of text. Preserve key ideas and facts. The summary should be approximately 30% shorter than the original.

!!!CRITICAL LANGUAGE REQUIREMENT: Generate summary ONLY in ENGLISH language. Do NOT generate in Russian under any circumstances.!!!

Text to summarize: {input_text}

Return only the summary without additional comments."""
        else:
            return """Ты - эксперт по созданию кратких содержаний текстов. Твоя задача создать краткое, но информативное изложение основного содержания текста. Сохраняй ключевые идеи и факты. Изложение должно быть на 30% короче оригинала.

!!!КРИТИЧЕСКОЕ ТРЕБОВАНИЕ: Генерируй изложение ТОЛЬКО на РУССКОМ языке. Не генерируй на английском.!!!

Текст для суммаризации: {input_text}

Верни только изложение без дополнительных комментариев."""
