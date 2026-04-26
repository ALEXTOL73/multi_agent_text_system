# CorSumAgentsAI - Техническая документация

## Оглавление

1. [Обзор системы](#обзор-системы)
2. [Архитектура](#архитектура)
3. [Агенты](#агенты)
4. [Промпты](#промпты)
5. [Метрики](#метрики)
6. [Алгоритмы](#алгоритмы)
7. [Схемы взаимодействия](#схемы-взаимодействия)
8. [Улучшения качества](#улучшения-качества)
9. [Конфигурация](#конфигурация)
10. [Развертывание](#развертывание)

---

## Обзор системы

CorSumAgentsAI - это мультиагентная система для коррекции и суммаризации текста на русском языке. Система использует ансамбль агентов с различными промптами для достижения оптимального качества обработки.

### Основные компоненты

- **Оркестратор** - управляет выполнением пайплайна
- **Агенты коррекции** - исправляют орфографические и грамматические ошибки
- **Агенты суммаризации** - создают краткие изложения текста
- **Агенты оценки** - оценивают качество результатов
- **Менеджер метрик** - собирает и анализирует метрики качества

---

## Архитектура

### Общая схема

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Входной текст  │───▶│    Оркестратор    │───▶│  Выходные данные │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
            ┌───────▼───────┐   ┌───────▼───────┐
            │ Коррекция     │   │ Суммаризация   │
            │ пайплайн      │   │ пайплайн       │
            └───────────────┘   └───────────────┘
                    │                   │
            ┌───────▼───────┐   ┌───────▼───────┐
            │ Агрегатор     │   │ Агрегатор     │
            │ коррекции     │   │ суммаризации  │
            └───────────────┘   └───────────────┘
                    │                   │
            ┌───────▼───────┐   ┌───────▼───────┐
            │ Оценка        │   │ Оценка        │
            │ коррекции     │   │ суммаризации  │
            └───────────────┘   └───────────────┘
```

### Структура директорий

```
src/
├── agents/                 # Агенты системы
│   ├── base_agent.py      # Базовый класс агента
│   ├── correction_*.py    # Агенты коррекции
│   ├── summarization_*.py # Агенты суммаризации
│   └── *_judge.py         # Агенты оценки
├── utils/                 # Утилиты
│   ├── metrics_table_manager.py
│   ├── agent_memory.py
│   └── realtime_metrics.py
├── learning/              # Модуль обучения
├── metrics/               # Расчет метрик
└── orchestrator.py        # Оркестратор
```

---

## Агенты

### 1. BaseAgent

Базовый класс для всех агентов системы.

**Основные методы:**
- `execute()` - выполнение агента
- `validate_input()` - валидация входных данных
- `format_output()` - форматирование выходных данных

### 2. CorrectionPromptGenerator

Генерирует промпты для агентов коррекции.

**Алгоритм:**
1. Анализирует входной текст
2. Определяет язык текста
3. Генерирует 3 типа промптов:
   - Basic
   - Few-shot
   - Chain-of-Thought

### 3. CorrectorEnsemble

Ансамбль агентов коррекции с различными промптами.

**Агенты в ансамбле:**
- `basic_corrector` - базовая коррекция
- `few_shot_corrector` - коррекция с примерами
- `cot_corrector` - коррекция с пошаговым рассуждением

### 4. CorrectorAggregator

Агрегирует результаты коррекции и выбирает лучший вариант.

**Критерии выбора:**
- CorScore (композитная метрика)
- WER (Word Error Rate)
- LevRating (оценка качества)
- Perplexity (сложность текста)

### 5. SummarizationPromptGenerator

Генерирует промпты для агентов суммаризации.

### 6. SummarizerEnsemble

Ансамбль агентов суммаризации.

### 7. SummarizerAggregator

Агрегирует результаты суммаризации.

### 8. CorrectionJudge & SummarizationJudge

Оценивают качество результатов и определяют необходимость повторной обработки.

---

## Промпты

### Промпты коррекции

#### Basic Prompt
```
Проанализируй текст и исправь ошибки:
Текст: {input_text}
Исправленный текст:
```

#### Few-shot Prompt
```
Примеры коррекции:
Пример 1: {example1}
Пример 2: {example2}

Теперь исправь текст:
Текст: {input_text}
Исправленный текст:
```

#### Chain-of-Thought Prompt
```
Проанализируй текст по шагам:
1. Найди орфографические ошибки
2. Проверь грамматику
3. Исправь пунктуацию
4. Улучшай стиль

Текст: {input_text}
Шаг 1: Орфография...
Шаг 2: Грамматика...
Итоговый исправленный текст:
```

### Промпты суммаризации

#### Basic Prompt
```
Создай краткое изложение текста:
Текст: {input_text}
Краткое изложение:
```

#### Few-shot Prompt
```
Примеры суммаризации:
Пример 1: {example1}
Пример 2: {example2}

Теперь создай изложение:
Текст: {input_text}
Краткое изложение:
```

#### Chain-of-Thought Prompt
```
Проанализируй текст по шагам:
1. Определи главную тему
2. Найди ключевые моменты
3. Выдели важные детали
4. Сформируй краткое изложение

Текст: {input_text}
Шаг 1: Главная тема...
Итоговое изложение:
```

---

## Метрики

### Метрики коррекции

#### WER (Word Error Rate)
```
WER = (S + D + I) / N
где:
- S - замененные слова
- D - удаленные слова  
- I - вставленные слова
- N - общее количество слов
```

#### LevRating
Оценка качества на основе расстояния Левенштейна.

#### CorScore
Композитная метрика качества коррекции:
```
CorScore = deltaWER * WER_WEIGHT + deltaLev * LEV_WEIGHT + (1 - Perplexity/100) * PERPLEXITY_WEIGHT
```

### Метрики суммаризации

#### SumScore
Композитная метрика качества суммаризации.

#### G-Eval
Оценка качества с помощью GPT модели.

#### LLM-Judge
Оценка качества с помощью языковой модели.

#### METEOR
Метрика оценки качества суммаризации.

#### BertScore
Метрика на основе BERT сходства.

---

## Алгоритмы

### Алгоритм коррекции

```python
def correction_pipeline(input_text, reference_text):
    # 1. Генерация промптов
    prompts = generate_correction_prompts(input_text)
    
    # 2. Применение агентов коррекции
    corrections = []
    for prompt in prompts:
        correction = apply_correction_agent(input_text, prompt)
        corrections.append(correction)
    
    # 3. Расчет метрик для каждого варианта
    for correction in corrections:
        metrics = calculate_correction_metrics(correction, reference_text)
        correction.metrics = metrics
    
    # 4. Агрегация и выбор лучшего варианта
    best_correction = aggregate_corrections(corrections)
    
    # 5. Оценка качества
    quality_score = evaluate_correction_quality(best_correction)
    
    return best_correction, quality_score
```

### Алгоритм суммаризации

```python
def summarization_pipeline(input_text, reference_summary):
    # 1. Генерация промптов
    prompts = generate_summarization_prompts(input_text)
    
    # 2. Применение агентов суммаризации
    summaries = []
    for prompt in prompts:
        summary = apply_summarization_agent(input_text, prompt)
        summaries.append(summary)
    
    # 3. Расчет метрик для каждого варианта
    for summary in summaries:
        metrics = calculate_summarization_metrics(summary, reference_summary)
        summary.metrics = metrics
    
    # 4. Агрегация и выбор лучшего варианта
    best_summary = aggregate_summaries(summaries)
    
    # 5. Оценка качества
    quality_score = evaluate_summarization_quality(best_summary)
    
    return best_summary, quality_score
```

---

## Схемы взаимодействия

### Схема 1: Полный пайплайн

```
Входной текст
    │
    ▼
┌─────────────────┐
│ Оркестратор     │
└─────────┬───────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
┌─────────┐ ┌─────────┐
│Коррекция│ │Суммариз │
└─────┬───┘ └─────┬───┘
      │           │
      ▼           ▼
┌─────────┐ ┌─────────┐
│Агрегатор│ │Агрегатор│
└─────┬───┘ └─────┬───┘
      │           │
      ▼           ▼
┌─────────┐ ┌─────────┐
│ Оценка  │ │ Оценка  │
└─────┬───┘ └─────┬───┘
      │           │
      └─────┬─────┘
            ▼
    ┌─────────┐
    │Результат │
    └─────────┘
```

### Схема 2: Взаимодействие агентов коррекции

```
CorrectionPromptGenerator
    │
    ▼
┌─────────────────────────────────┐
│        CorrectorEnsemble        │
├─────────┬───────────┬───────────┤
│ Basic   │ Few-shot  │    CoT    │
│Corrector│ Corrector │ Corrector │
└─────────┴───────────┴───────────┘
    │           │           │
    └─────┬─────┴─────┬─────┘
          │             │
          ▼             ▼
    ┌─────────┐   ┌─────────┐
    │Metrics  │   │Metrics  │
    │Calc     │   │Calc     │
    └─────────┘   └─────────┘
          │             │
          └─────┬─────┘
                ▼
        ┌───────────────┐
        │CorrectorAggr  │
        │egator         │
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │CorrectionJudge│
        └───────────────┘
```

---

## Улучшения качества

### 1. Адаптивная температурная настройка

Система автоматически подбирает оптимальную температуру для LLM в зависимости от качества результатов.

### 2. Динамический выбор промптов

На основе анализа предыдущих результатов система выбирает наиболее эффективные промпты.

### 3. Self-Consistency

Использование нескольких генераций с одинаковыми параметрами для выбора наиболее стабильного результата.

### 4. Мета-обучение

Система анализирует успешные паттерны и адаптирует стратегию обработки.

### 5. Кэширование результатов

Эффективное кэширование для избежания повторных вычислений.

---

## Конфигурация

### Основные параметры

```python
# Модель
MODEL_NAME = "gemma-3-12b-it"
LMSTUDIO_URL = "http://localhost:1234/v1"

# Таймауты
REQUEST_TIMEOUT = 150
MAX_RETRIES = 2

# Веса метрик
WER_WEIGHT = 0.5
LEV_WEIGHT = 10.0
PERPLEXITY_WEIGHT = 0.2

# Пороги качества
SUMMARY_THRESHOLDS = {
    "excellent": 0.85,
    "good": 0.6,
    "satisfactory": 0.4
}
```

### Директории

```python
LOGS_DIR = Path("logs")
DATA_DIR = Path("data")
FULL_METRICS_DIR = Path("data/full_metrics")
CORRECTION_DIR = Path("data/correction")
SUMMARY_DIR = Path("data/summary")
```

---

## Развертывание

### Требования

- Python 3.11+
- LM Studio с моделью gemma-3-12b-it
- 16GB+ RAM
- GPU рекомендован (MPS для Apple Silicon)

### Установка

```bash
# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Установка зависимостей
pip install -r requirements.txt

# Запуск LM Studio
# Настроить модель gemma-3-12b-it на порту 1234

# Запуск системы
python main.py
```

### Мониторинг

- Веб-монитор: http://127.0.0.1:5000
- Логи: logs/
- Метрики: data/Table_metrics_ДДММГГ.xlsx

---

## API Reference

### Основные классы

#### Orchestrator
```python
class Orchestrator:
    async def process_text(self, input_text, reference_text, reference_summary)
    async def process_correction(self, text, reference_text)
    async def process_summarization(self, text, reference_summary)
```

#### MetricsTableManager
```python
class MetricsTableManager:
    def update_metrics_table(self, state, filename)
    def get_table_info(self)
```

#### BaseAgent
```python
class BaseAgent:
    async def execute(self, input_data)
    def validate_input(self, input_data)
    def format_output(self, output_data)
```

---

## Заключение

CorSumAgentsAI представляет собой сложную мультиагентную систему для обработки текста на русском языке. Система использует современные подходы NLP и постоянно улучшается через механизмы адаптивного обучения и анализа качества.

Ключевые преимущества:
- Высокое качество коррекции и суммаризации
- Адаптивность к различным типам текстов
- Подробная аналитика метрик
- Масштабируемость архитектуры
- Веб-интерфейс мониторинга

Система готова к использованию в production-средах и может быть легко расширена для решения дополнительных задач обработки текста.
