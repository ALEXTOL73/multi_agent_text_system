#!/usr/bin/env python3
"""
Тест для проверки оптимизаций: кэширование и асинхронная обработка
"""

import asyncio
import sys
import time
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.lm_studio_client import LMStudioClient
from main import process_batch_files_async, process_batch_files

async def test_caching():
    """Тест кэширования LLM запросов"""
    print("🧪 Тест кэширования LLM запросов")
    print("=" * 50)
    
    try:
        # Создаем клиент с включенным кэшем
        client = LMStudioClient(
            base_url="http://localhost:1234/v1",
            model_name="gemma-3-12b-it",
            cache_enabled=True,
            cache_size=50
        )
        
        # Тестовый промпт
        test_prompt = "Исправь ошибку: Я пошол в школ"
        test_temperature = 0.7
        test_max_tokens = 100
        test_system_prompt = "Ты учитель русского языка"
        
        print(f"📝 Тестовый промпт: {test_prompt}")
        print(f"🌡️  Температура: {test_temperature}")
        print(f"📊 Макс токенов: {test_max_tokens}")
        print()
        
        # Первый запрос (должен зайти в LM Studio)
        print("🔄 Первый запрос (без кэша)...")
        start_time = time.time()
        
        try:
            async with client:
                result1 = await client.generate_with_metadata(
                    prompt=test_prompt,
                    temperature=test_temperature,
                    max_tokens=test_max_tokens,
                    system_prompt=test_system_prompt
                )
            first_request_time = time.time() - start_time
            print(f"✅ Первый запрос выполнен за {first_request_time:.2f} сек")
            print(f"📄 Результат: {result1['text'][:100]}...")
            
            # Проверяем статистику кэша
            cache_stats = client.get_cache_stats()
            print(f"📊 Статистика кэша после первого запроса:")
            print(f"   - Hit rate: {cache_stats['hit_rate']:.2%}")
            print(f"   - Cache hits: {cache_stats['cache_hits']}")
            print(f"   - Cache misses: {cache_stats['cache_misses']}")
            print()
            
            # Второй запрос с теми же параметрами (должен взять из кэша)
            print("🔄 Второй запрос (с кэшем)...")
            start_time = time.time()
            
            async with client:
                result2 = await client.generate_with_metadata(
                    prompt=test_prompt,
                    temperature=test_temperature,
                    max_tokens=test_max_tokens,
                    system_prompt=test_system_prompt
                )
            second_request_time = time.time() - start_time
            print(f"✅ Второй запрос выполнен за {second_request_time:.2f} сек")
            print(f"📄 Результат: {result2['text'][:100]}...")
            
            # Проверяем статистику кэша
            cache_stats = client.get_cache_stats()
            print(f"📊 Статистика кэша после второго запроса:")
            print(f"   - Hit rate: {cache_stats['hit_rate']:.2%}")
            print(f"   - Cache hits: {cache_stats['cache_hits']}")
            print(f"   - Cache misses: {cache_stats['cache_misses']}")
            print()
            
            # Проверяем что результаты одинаковы
            if result1['text'] == result2['text']:
                print("✅ Результаты идентичны - кэш работает правильно!")
            else:
                print("❌ Результаты различаются - проблема с кэшем!")
            
            # Проверяем ускорение
            if second_request_time < first_request_time:
                speedup = first_request_time / second_request_time
                print(f"🚀 Ускорение: {speedup:.1f}x")
            else:
                print("⚠️  Ускорение не обнаружено")
            
            print()
            print("🎉 Тест кэширования завершен!")
            
        except Exception as e:
            print(f"❌ Ошибка при тестировании кэширования: {e}")
            print("💡 Убедитесь что LM Studio запущен на http://localhost:1234")
            
    except Exception as e:
        print(f"❌ Ошибка инициализации клиента: {e}")

async def test_async_processing():
    """Тест асинхронной обработки файлов"""
    print("🧪 Тест асинхронной обработки файлов")
    print("=" * 50)
    
    try:
        # Проверяем наличие файлов для теста
        inputs_dir = Path("inputs/incorrect")
        if not inputs_dir.exists():
            print("❌ Директория inputs/incorrect не найдена")
            print("💡 Создайте тестовые файлы в inputs/incorrect/")
            return
        
        test_files = list(inputs_dir.glob("*.txt"))
        if len(test_files) < 2:
            print("❌ Нужно минимум 2 файла для теста асинхронной обработки")
            print(f"💡 Найдено файлов: {len(test_files)}")
            return
        
        print(f"📁 Найдено файлов для теста: {len(test_files)}")
        for file in test_files[:3]:  # Показываем первые 3 файла
            print(f"   - {file.name}")
        
        if len(test_files) > 3:
            print(f"   ... и еще {len(test_files) - 3} файлов")
        print()
        
        # Тестируем асинхронную обработку
        print("🔄 Тест асинхронной обработки...")
        start_time = time.time()
        
        try:
            # Используем небольшое количество файлов для теста
            max_files = min(3, len(test_files))
            
            # Создаем временную директорию для теста
            test_dir = Path("test_async_processing")
            test_dir.mkdir(exist_ok=True)
            
            # Копируем несколько файлов для теста
            import shutil
            test_inputs_dir = test_dir / "inputs" / "incorrect"
            test_inputs_dir.mkdir(parents=True, exist_ok=True)
            
            for i, file in enumerate(test_files[:max_files]):
                shutil.copy2(file, test_inputs_dir / file.name)
            
            # Временно меняем рабочую директорию для теста
            original_inputs = Path("inputs/incorrect")
            backup_inputs = original_inputs.rename("inputs/incorrect_backup") if original_inputs.exists() else None
            
            try:
                # Копируем тестовые файлы в основную директорию
                if backup_inputs:
                    original_inputs.mkdir(parents=True, exist_ok=True)
                    for file in test_inputs_dir.glob("*.txt"):
                        shutil.copy2(file, original_inputs / file.name)
                
                # Запускаем асинхронную обработку
                results = await process_batch_files_async(
                    domain="general",
                    enable_correction=True,
                    enable_summarization=False,  # Отключаем суммаризацию для быстрого теста
                    metrics_manager=None,
                    max_concurrent_files=2
                )
                
                async_time = time.time() - start_time
                print(f"✅ Асинхронная обработка завершена за {async_time:.2f} сек")
                print(f"📊 Обработано файлов: {len(results)}/{max_files}")
                
                if results:
                    print("📄 Результаты:")
                    for filename, result in results.items():
                        status = "✅" if result else "❌"
                        print(f"   {status} {filename}")
                
            finally:
                # Восстанавливаем оригинальные файлы
                if backup_inputs:
                    if original_inputs.exists():
                        shutil.rmtree(original_inputs)
                    backup_inputs.rename(original_inputs)
                
                # Удаляем тестовую директорию
                if test_dir.exists():
                    shutil.rmtree(test_dir)
            
        except Exception as e:
            print(f"❌ Ошибка при асинхронной обработке: {e}")
            return
        
        print()
        print("🎉 Тест асинхронной обработки завершен!")
        
    except Exception as e:
        print(f"❌ Ошибка подготовки теста: {e}")

async def main():
    """Главная функция тестирования"""
    print("🚀 Тестирование оптимизаций системы")
    print("=" * 60)
    print()
    
    # Тест кэширования
    await test_caching()
    print()
    
    # Тест асинхронной обработки
    await test_async_processing()
    print()
    
    print("🎊 Все тесты завершены!")

if __name__ == "__main__":
    asyncio.run(main())
