# CorSumAgentsAI - Мультиагентная система коррекции и суммаризации текста

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

**CorSumAgentsAI** - это продвинутая мультиагентная система для автоматической коррекции орфографических, грамматических и пунктуационных ошибок в русскоязычных/англоязычных текстах с последующей суммаризацией исправленного текста.

## 🚀 Ключевые особенности

- 🤖 **Агентно-ориентированная архитектура** - каждый компонент является независимым агентом
- 🧠 **Использование локальных LLM** - работа через LM Studio без отправки данных в облако
- 📊 **Комплексная оценка качества** - WER, LevRating, Perplexity, G-Eval, METEOR, BertScore
- 🎯 **Адаптивные температуры** и self-consistency для повышения качества
- 💾 **Система памяти агентов** с семантическим поиском успешных примеров
- 🔄 **Динамическое few-shot обучение** на основе релевантных примеров
- 📈 **Метрики качества** с настраиваемыми порогами
- 💻 **CLI интерфейс** и возможность экспорта результатов

## 📋 Содержание

- [Установка](#установка)
- [Требования](#требования)
- [Быстрый старт](#быстрый-старт)
- [Архитектура системы](#архитектура-системы)
- [Использование](#использование)
- [Конфигурация](#конфигурация)
- [Метрики качества](#метрики-качества)
- [Примеры использования](#примеры-использования)
- [Устранение проблем](#устранение-проблем)

## 🛠️ Установка

### 1. Клонирование репозитория

```bash
git clone https://github.com/your-repo/multi_agent_text_system.git
cd multi_agent_text_system
```

### 2. Создание виртуального окружения

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Установка и настройка LM Studio

1. Скачайте [LM Studio](https://lmstudio.ai/)
2. Установите и запустите приложение
3. Загрузите модель (рекомендуется `gemma-3-12b-it` или `llama-3-8b-instruct`)
4. Запустите локальный сервер на порту `1234`
5. Убедитесь, что сервер доступен по адресу `http://localhost:1234`

## 📋 Требования

### Программное обеспечение
- **Python 3.10+**
- **LM Studio** с загруженной моделью
- **Git** (для клонирования репозитория)

### Железо
- **Рекомендуется**: GPU с 8+ ГБ видеопамяти (NVIDIA CUDA)
- **Минимум**: CPU с 16 ГБ ОЗУ
- **Диск**: ~10 ГБ свободного пространства

## 🚀 Быстрый старт

### Базовое использование

```bash
# Простая коррекция и суммаризация
python main.py --input "првиет как дила? я иду в шклу"

# Только коррекция
python main.py --input "текст с ошибками" --no-summarization

# Из файла
python main.py --file input.txt --output results.json
```

### С эталонными данными

```bash
python main.py --file input.txt --reference correct.txt --ref-summary summary.txt
```

### Указание домена

```bash
python main.py --input "медицинский текст" --domain medical
```

## 🏗️ Архитектура системы

### Общая схема

```
[Input Text] → Оркестратор → [Агент генерации промпта коррекции] → [Ансамбль корректоров] → [Агрегатор коррекций] → [Судья коррекции] → (возврат при низком качестве) → [Агент генерации промпта суммаризации] → [Ансамбль суммаризаторов] → [Агрегатор суммаризаций] → [Судья суммаризации] → [Output]
```

### Компоненты системы

#### 🤖 Агенты
1. **CorrectionPromptGenerator** - генерация промптов для коррекции
2. **CorrectorEnsemble** - ансамбль корректоров с разными температурами
3. **CorrectorAggregator** - агрегация лучших вариантов коррекции
4. **CorrectionJudge** - оценка качества коррекции
5. **SummarizationPromptGenerator** - генерация промптов для суммаризации
6. **SummarizerEnsemble** - ансамбль суммаризаторов
7. **SummarizerAggregator** - агрегация вариантов суммаризации
8. **SummarizationJudge** - оценка качества суммаризации

#### 🔧 Утилиты
- **LMStudioClient** - асинхронный клиент для LM Studio
- **AgentMemory** - система памяти с семантическим поиском
- **MetricsCalculator** - расчет метрик качества

## 📖 Использование

### Командная строка

#### Основные параметры

```bash
python main.py [OPTIONS]

Обязательные параметры:
  --input TEXT, -i TEXT     Текст для обработки
  --file FILE, -f FILE      Файл с текстом для обработки

Опциональные параметры:
  --reference TEXT, -r TEXT    Эталонный текст для оценки коррекции
  --ref-summary TEXT          Эталонная суммаризация для оценки
  --domain TEXT, -d TEXT     Домен текста (default: general)
  --output FILE, -o FILE      Файл для сохранения результатов
  --no-correction            Отключить коррекцию текста
  --no-summarization         Отключить суммаризацию текста
  --info                     Показать информацию о системе
  --version                  Показать версию
```

#### Примеры

```bash
# Базовый пример
python main.py --input "првиет как дила?"

# Только коррекция
python main.py --input "текст с ошибками" --no-summarization

# Обработка файла с сохранением результатов
python main.py --file document.txt --output results.json

# С эталонными данными для оценки качества
python main.py --file input.txt --reference correct.txt --ref-summary summary.txt

# Медицинский домен
python main.py --input "пациент жалуется на головную боль" --domain medical

# Показать информацию о системе
python main.py --info
```

### Структура результатов

```json
{
  "input_text": "исходный текст",
  "corrected_text": "исправленный текст",
  "summary": "суммаризация",
  "metrics_correction": {
    "wer_corrected": 0.0,
    "lev_corrected": 1.0,
    "perplexity": 15.2,
    "cor_score": 0.95
  },
  "summary_metrics": {
    "geval": 0.85,
    "llm_judge": 8.2,
    "meteor": 0.72,
    "bert_score": 0.89,
    "sum_score": 0.81
  },
  "correction_quality": "ОТЛИЧНО",
  "summary_quality": "ХОРОШО",
  "processing_time": 12.34
}
```

## ⚙️ Конфигурация

Основные настройки находятся в файле `config.py`:

### LM Studio
```python
LMSTUDIO_URL = "http://localhost:1234/v1"
MODEL_NAME = "gemma-3-12b-it"
REQUEST_TIMEOUT = 120
MAX_RETRIES = 3
```

### Коррекция
```python
LEV_WEIGHT = 0.3
PERPLEXITY_WEIGHT = 0.2
DELTA_LEV_THRESHOLD = 0.02
MAX_CORRECTION_ATTEMPTS = 2
```

### Суммаризация
```python
SUMMARY_TEMPERATURES = [0.4, 0.6, 0.8]
GEVAL_WEIGHT = 0.3
LLM_JUDGE_WEIGHT = 0.2
METEOR_WEIGHT = 0.5
SUMMARIZATION_ATTEMPTS = 2
```

### Few-shot обучение
```python
DYNAMIC_FEW_SHOT_ENABLED = True
MAX_FEW_SHOT_EXAMPLES = 3
FEW_SHOT_SIMILARITY_THRESHOLD = 0.6
```

## 📊 Метрики качества

### Для коррекции

| Метрика | Диапазон | Описание |
|----------|----------|----------|
| **WER** | 0-1 | Word Error Rate (ниже лучше) |
| **LevRating** | 0-1 | Сходство Левенштейна (выше лучше) |
| **Perplexity** | >0 | Перплексия (ниже лучше) |
| **CorScore** | -∞...+∞ | Композитная метрика (выше лучше) |

### Для суммаризации

| Метрика | Диапазон | Описание |
|----------|----------|----------|
| **G-Eval** | 0-1 | LLM-оценка качества |
| **LLM-Judge** | 1-10 | Прямая оценка LLM |
| **METEOR** | 0-1 | Схожесть с эталоном |
| **BertScore** | 0-1 | Семантическое сходство |
| **SumScore** | 0-1 | Композитная метрика |

### Пороги качества

#### Коррекция
- **ОТЛИЧНО**: ΔWER > 0.3 и ΔLev > -0.05
- **ХОРОШО**: ΔWER > 0.15 и ΔLev > -0.1  
- **УДОВЛЕТВОРИТЕЛЬНО**: ΔWER > 0
- **ТРЕБУЕТСЯ УЛУЧШЕНИЕ**: иначе

#### Суммаризация
- **ОТЛИЧНО**: SumScore > 0.8
- **ХОРОШО**: SumScore > 0.6
- **УДОВЛЕТВОРИТЕЛЬНО**: SumScore > 0.4
- **ТРЕБУЕТСЯ УЛУЧШЕНИЕ**: иначе

## 💡 Примеры использования

### Пример 1: Базовая коррекция

```bash
python main.py --input "првиет как дила? я иду в шклу"
```

**Результат:**
```
Исходный текст: првиет как дила? я иду в шклу
Исправленный текст: привет как дела? я иду в школу
Суммаризация: Приветствие и сообщение о походе в школу.
Метрики коррекции: WER=0.0, LevRating=1.0, CorScore=0.95
Качество коррекции: ОТЛИЧНО
```

### Пример 2: Обработка длинного текста

```bash
python main.py --file long_document.txt --output analysis.json
```

### Пример 3: С эталонными данными

```bash
python main.py --file input.txt --reference reference.txt --ref-summary ref_summary.txt
```

## 🔧 Устранение проблем

### Частые проблемы

#### 1. Ошибка соединения с LM Studio
```
❌ Не удалось установить соединение с LM Studio
```

**Решение:**
- Убедитесь, что LM Studio запущен
- Проверьте, что сервер работает на порту 1234
- Проверьте URL в `config.py`

#### 2. Недостаточно памяти
```
❌ CUDA out of memory
```

**Решение:**
- Используйте меньшую модель
- Уменьшите `max_tokens` в конфигурации
- Используйте CPU вместо GPU

#### 3. Медленная обработка
```
⏱️ Время обработки: 120+ секунд
```

**Решение:**
- Используйте GPU
- Уменьшите количество агентов
- Отключите self-consistency

#### 4. Низкое качество коррекции
```
🎯 Качество коррекции: ТРЕБУЕТСЯ УЛУЧШЕНИЕ
```

**Решение:**
- Проверьте качество модели в LM Studio
- Увеличьте `MAX_CORRECTION_ATTEMPTS`
- Добавьте эталонный текст для оценки

### Логирование

Система создает подробные логи в директории `logs/`:
```
logs/
├── corsum_agents_20240420.log
├── intermediate_correction_judge_20240420_143022.json
└── intermediate_summarizer_ensemble_20240420_143025.json
```

### Структура данных

```
data/
├── correction/           # Результаты коррекции
├── summary/             # Результаты суммаризации
├── correction_metrics/   # Метрики коррекции
├── summary_metrics/     # Метрики суммаризации
└── memory/             # Память агентов
```

## 🤝 Вклад в проект

1. Fork репозитория
2. Создайте ветку (`git checkout -b feature/AmazingFeature`)
3. Commit изменения (`git commit -m 'Add some AmazingFeature'`)
4. Push в ветку (`git push origin feature/AmazingFeature`)
5. Откройте Pull Request

## 📄 Лицензия

Этот проект лицензирован под MIT License - см. файл [LICENSE](LICENSE) для деталей.

## 🙏 Благодарности

- **LM Studio** за отличную платформу локальных LLM
- **Hugging Face** за модели и инструменты
- **Sentence Transformers** за эмбеддинги
- **NLTK** за метрики качества

## 📞 Контакты

- Проект: [GitHub Repository](https://github.com/your-repo/multi_agent_text_system)
- Вопросы: [Issues](https://github.com/your-repo/multi_agent_text_system/issues)
- Документация: [Wiki](https://github.com/your-repo/multi_agent_text_system/wiki)

---

**CorSumAgentsAI** - Сделаем тексты лучше вместе! 🚀
