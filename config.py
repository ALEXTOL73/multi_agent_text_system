"""
Конфигурация мультиагентной системы CorSumAgentsAI
"""

from pathlib import Path
from typing import List

# --- Общие ---
MODEL_NAME = "gemma-3-12b-it"
LMSTUDIO_URL = "http://localhost:1234/v1"
REQUEST_TIMEOUT = 150
MAX_RETRIES = 3

# --- Logging ---
LOGS_DIR = Path("logs")
LOG_LEVEL = "INFO"
ENABLE_DATE_FOLDERS = True
DATE_FORMAT = "DDMMYY"  # Format: 220426 for April 22, 2026

# --- Коррекция ---
LEV_WEIGHT = 0.3
PERPLEXITY_WEIGHT = 0.2
DELTA_LEV_THRESHOLD = 0.02
MAX_LEV_RETRY_ATTEMPTS = 3
LEV_RETRY_TEMPS = [0.2, 0.5, 0.8]
USE_FEW_SHOT_PROMPT = True
USE_CHAIN_OF_THOUGHT_PROMPT = True
DYNAMIC_TEMPERATURES_ENABLED = False
SELF_CONSISTENCY_ENABLED = False
SELF_CONSISTENCY_EXTRA_COUNT = 0
MAX_CORRECTION_ATTEMPTS = 1
CORRECTION_TEMPERATURES = [0.7]

# --- Суммаризация ---
SUMMARY_TEMPERATURES = [0.7]
GEVAL_WEIGHT = 0.25
LLM_JUDGE_WEIGHT = 0.2
METEOR_WEIGHT = 0.35
BERT_SCORE_WEIGHT = 0.2
SUMMARIZATION_ATTEMPTS = 1

# --- Simplified prompts ---
USE_FEW_SHOT_PROMPT = True
DYNAMIC_TEMPERATURES_ENABLED = False

# --- Few-shot ---
DYNAMIC_FEW_SHOT_ENABLED = False
MAX_FEW_SHOT_EXAMPLES = 3
FEW_SHOT_SIMILARITY_THRESHOLD = 0.6
SUMMARY_DYNAMIC_FEW_SHOT_ENABLED = False
SUMMARY_MAX_FEW_SHOT_EXAMPLES = 3
SUMMARY_FEW_SHOT_LENGTH_RATIO = 0.3
SUMMARY_FEW_SHOT_SIMILARITY_THRESHOLD = 0.3

# --- Память ---
PROMPT_AGGREGATION_ENABLED = False
PROMPT_CACHE_ENABLED = True
PROMPT_CACHE_MAX_SIZE = 100
PROMPT_CACHE_MIN_IMPROVEMENT = 0.05

# --- Пути ---
DATA_DIR = Path("data")
CORRECTION_METRICS_DIR = DATA_DIR / "correction_metrics"
SUMMARY_METRICS_DIR = DATA_DIR / "summary_metrics"
CORRECTION_DIR = DATA_DIR / "correction"
SUMMARY_DIR = DATA_DIR / "summary"
LOGS_DIR = Path("logs")
MEMORY_DIR = DATA_DIR / "memory"
INPUT_DIR = Path("inputs")

# --- Оркестратор ---
STOP_ON_ERROR = True
SAVE_INTERMEDIATE_STATES = False

# --- Статические промпты для коррекции ---
STATIC_CORRECTION_EXAMPLES = [
    {
        "input": "првиет как дила?",
        "output": "привет как дела?"
    },
    {
        "input": "я иду в шклу",
        "output": "я иду в школу"
    },
    {
        "input": "сегодная хорошая погода",
        "output": "сегодня хорошая погода"
    }
]

# --- Статические промпты для суммаризации ---
STATIC_SUMMARY_EXAMPLES = [
    {
        "input": "Hello! How are you? I went to the store today and bought milk. Then I met a friend and we had lunch together at a cafe. In the evening I watched an interesting movie.",
        "output": "A person shared about their day: going to the store, meeting a friend, having lunch at a cafe, and watching a movie in the evening."
    },
    {
        "input": "The conference discussed recent advances in artificial intelligence. Speakers talked about neural networks, machine learning, and natural language processing. Participants asked many questions and shared experiences.",
        "output": "The conference covered modern AI technologies: neural networks, machine learning, and NLP. Speakers and participants actively discussed achievements and exchanged experiences."
    }
]

# --- Системные промпты ---
CORRECTION_SYSTEM_PROMPT = """You are an expert at correcting errors in texts. Your task is to correct spelling, grammar, and punctuation errors in given text. Preserve the original meaning and style of the text. Return only the corrected text without additional comments."""

SUMMARIZATION_SYSTEM_PROMPT = """You are an expert at creating concise summaries of texts. Your task is to create a brief but informative summary of the main content of text. Preserve key ideas and facts. The summary should be approximately 30% shorter than the original."""

PROMPT_GENERATION_SYSTEM_PROMPT = """You are an expert at creating effective prompts for language models. Create a prompt that will help the model perform the specified task with maximum quality. The prompt should be clear, specific, and contain necessary instructions."""

CHAIN_OF_THOUGHT_PROMPT = {
    "en": """You are an expert at creating step-by-step solutions to complex problems. Break down the problem into smaller steps and provide a clear explanation for each step. The goal is to create a chain of thought that leads to the final solution.""",
    "ru": """Ты - эксперт по созданию пошаговых решений сложных проблем. Разбей проблему на более мелкие шаги и предоставь четкое объяснение для каждого шага. Цель - создать цепочку мыслей, которая приводит к окончательному решению."""
}

# --- Промпты для агрегации ---
AGGREGATION_PROMPT = """Улучшь следующий текст, используя лучшие элементы из альтернативных вариантов:

Основной вариант: {best_text}

Альтернативный вариант 1: {alt_1}
Альтернативный вариант 2: {alt_2}

Создай финальный вариант, который сохраняет смысл, но улучшает стиль, грамматику и ясность. Возвращай только улучшенный текст."""

# --- Промпты для оценки ---
GEVAL_PROMPT = """Оцени качество следующего суммаризационного текста по шкале от 0 до 1, где 1 - отлично.

Оригинальный текст: {original_text}
Суммаризация: {summary_text}
{reference_summary}

Критерии оценки:
1. Точность - насколько точно суммаризация отражает содержание оригинала
2. Полнота - насколько хорошо охвачены основные идеи
3. Краткость - насколько лаконично изложено содержание
4. Ясность - насколько понятен и читаем текст
5. Соответствие эталону - насколько близко к эталонному резюме (если есть)
6. Именованные сущности - сохранены ли важные имена, даты, места, организации из оригинала

Особое внимание:
- Сравни с эталонным резюме если оно предоставлено
- Проверь наличие ключевых именованных сущностей из оригинала
- Оцени информативность и фактическую точность

Верни только число от 0 до 1."""

LLM_JUDGE_PROMPT = """Evaluate the quality of the following summary text CRITICALLY on a scale of 1 to 10, where 10 is excellent.

Original text: {original_text}
Summary: {summary_text}
{reference_summary}

SCORING GUIDELINES - BE PRECISE:
1-3: Poor summary - major inaccuracies, missing key information, poorly written
4-6: Average summary - some inaccuracies, moderate coverage, decent writing
7-8: Good summary - mostly accurate, good coverage, well-written
9-10: Excellent summary - highly accurate, comprehensive, perfectly written

Evaluation criteria:
1. Accuracy - how accurately the summary reflects the original content
2. Completeness - how well the main ideas are covered
3. Conciseness - how briefly the content is presented
4. Clarity - how clear and readable the text is
5. Reference alignment - how well it matches the reference summary (if provided)
6. Named entities - preservation of important names, dates, places, organizations

CRITICAL EVALUATION REQUIREMENTS:
- Do NOT give default middle scores (7-8) unless truly deserved
- Be strict about factual accuracy and completeness
- Penalize summaries that miss important named entities
- Compare carefully with reference summary if provided
- Consider if summary adds incorrect information

Provide your evaluation in this format:
Score: [number from 1 to 10]
Explanation: [specific explanation of why you gave this score, mentioning strengths and weaknesses]"""

# --- Пороги качества ---
CORRECTION_THRESHOLDS = {
    "excellent": {"delta_wer": 0.3, "delta_lev": -0.05},
    "good": {"delta_wer": 0.15, "delta_lev": -0.1},
    "satisfactory": {"delta_wer": 0}
}

SUMMARY_THRESHOLDS = {
    "excellent": 0.8,
    "good": 0.6,
    "satisfactory": 0.4
}

# --- Метрические веса ---
AGGREGATION_TEMPERATURE = 0.3
MIN_SIMILARITY = 0.6
