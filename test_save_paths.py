#!/usr/bin/env python3
"""
Тестовый скрипт для проверки сохранения метрик в правильную директорию
"""
import asyncio
import sys
import os
from pathlib import Path

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent))

from main import process_text_file
from config import FULL_METRICS_DIR

async def test_save_paths():
    """Тест сохранения путей"""
    print("=== Тест сохранения путей метрик ===")
    
    # Создаем тестовый файл
    test_file = "test_file.txt"
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write("Это тестовый текст для проверки сохранения метрик в правильную директорию.")
    
    # Очищаем директорию full_metrics перед тестом
    if FULL_METRICS_DIR.exists():
        import shutil
        shutil.rmtree(FULL_METRICS_DIR)
        print(f"Очищена директория: {FULL_METRICS_DIR}")
    
    FULL_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Создана директория: {FULL_METRICS_DIR}")
    
    try:
        # Запускаем обработку файла
        results = await process_text_file(
            input_file=test_file,
            output_file="test_output",
            enable_correction=True,
            enable_summarization=True
        )
        
        print(f"Результаты обработки: {results}")
        
        # Проверяем где создались файлы
        print("\n=== Проверка созданных файлов ===")
        
        # Проверяем data/full_metrics/
        full_metrics_files = list(FULL_METRICS_DIR.glob("*.json"))
        print(f"Файлы в {FULL_METRICS_DIR}:")
        for f in full_metrics_files:
            print(f"  - {f.name}")
        
        # Проверяем что нет файлов в data/
        data_files = list(Path("data").glob("*.json"))
        print(f"\nФайлы в data/:")
        for f in data_files:
            print(f"  - {f.name}")
        
        # Проверяем что нет файлов в data/correction_metrics/
        correction_metrics_dir = Path("data/correction_metrics")
        if correction_metrics_dir.exists():
            correction_files = list(correction_metrics_dir.glob("*.json"))
            print(f"\nФайлы в {correction_metrics_dir}:")
            for f in correction_files:
                print(f"  - {f.name}")
        
        # Проверяем что нет файлов в data/summary_metrics/
        summary_metrics_dir = Path("data/summary_metrics")
        if summary_metrics_dir.exists():
            summary_files = list(summary_metrics_dir.glob("*.json"))
            print(f"\nФайлы в {summary_metrics_dir}:")
            for f in summary_files:
                print(f"  - {f.name}")
        
        # Выводим результат
        if len(full_metrics_files) > 0 and len(data_files) == 0:
            print("\n✅ УСПЕХ: Все файлы метрик сохраняются в data/full_metrics/")
        else:
            print("\n❌ ОШИБКА: Файлы метрик сохраняются не в ту директорию!")
        
    except Exception as e:
        print(f"Ошибка при выполнении теста: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Удаляем тестовый файл
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"\nУдален тестовый файл: {test_file}")

if __name__ == "__main__":
    asyncio.run(test_save_paths())
