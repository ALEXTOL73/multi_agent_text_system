#!/usr/bin/env python3
"""
Тестовый пример для проверки передачи текста в систему
"""

import asyncio
import sys
import os

# Добавляем src в путь
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.orchestrator import Orchestrator
from src.utils.agent_memory import AgentMemory
from src.utils.lm_studio_client import LMStudioClient
from src.metrics.metrics_calculator import MetricsCalculator
from src.agents.correction_prompt_generator import CorrectionPromptGenerator
from src.agents.summarization_prompt_generator import SummarizationPromptGenerator
from src.agents.corrector_ensemble import CorrectorEnsemble
from src.agents.summarizer_ensemble import SummarizerEnsemble
from src.agents.corrector_aggregator import CorrectorAggregator
from src.agents.summarizer_aggregator import SummarizerAggregator
from src.agents.correction_judge import CorrectionJudge
from src.agents.summarization_judge import SummarizationJudge
import config

async def test_text_flow():
    """Тестовый пример проверки потока текста"""
    
    print("🧪 ТЕСТОВЫЙ ПРИМЕР: Проверка потока текста")
    print("="*60)
    
    # Тестовый текст
    test_text = """ПЕКИН, 26 ноября, /ТАСС/. Суд в КНР приговорил бывшего партсекретаря и председателя совета директоров Bank of China Лю Ляньгэ к смертной казни за коррупцию и незаконную выдачу кредитов. Как сообщила Верховная прокуратура КНР, он получил взятки на сумму около 1,21 млн юаней ($78,16 млн). Приговор вынесен с отсрочкой в исполнении на два года. Суд постановил пожизненно лишить Лю Ляньгэ политических прав и конфисковать все незаконно полученные активы в пользу государства."""
    
    test_reference = """Бывший глава Bank of China Лю Ляньгэ приговорен к смертной казни с отсрочкой за коррупцию на сумму $78 млн."""
    
    print(f"📝 ВХОДНОЙ ТЕКСТ: {test_text[:100]}...")
    print(f"📋 ЭТАЛОННАЯ СУММАРИЗАЦИЯ: {test_reference}")
    print()
    
    # Инициализация компонентов
    try:
        lm_client = LMStudioClient()
        agent_memory = AgentMemory(memory_dir="data/memory")
        metrics_calculator = MetricsCalculator()
        
        # Creating orchestrator
        orchestrator = Orchestrator()
        await orchestrator.initialize()
        
        # Creating only essential agents for testing
        agents = {
            "correction_prompt_generator": CorrectionPromptGenerator(lm_client, agent_memory),
            "corrector_ensemble": CorrectorEnsemble(lm_client, agent_memory, metrics_calculator),
            "summarization_prompt_generator": SummarizationPromptGenerator(lm_client, agent_memory),
            "summarizer_ensemble": SummarizerEnsemble(lm_client, agent_memory, metrics_calculator)
        }
        
        # Начальное состояние
        state = {
            "input_text": test_text,
            "reference_text": test_text,  # Для коррекции
            "reference_summary": test_reference,  # Для суммаризации
            "domain": "news"
        }
        
        print("🔄 ЗАПУСК КОРРЕКЦИИ...")
        print("-"*40)
        
        # Запуск коррекции
        correction_state = await orchestrator._run_correction_pipeline(state)
        
        print("✅ КОРРЕКЦИЯ ЗАВЕРШЕНА")
        print(f"📊 ИСПРАВЛЕННЫЙ ТЕКСТ: {correction_state.get('corrected_text', 'ОТСУТСТВУЕТ')[:100]}...")
        print()
        
        print("🔄 ЗАПУСК СУММАРИЗАЦИИ...")
        print("-"*40)
        
        # Запуск суммаризации
        final_state = await orchestrator._run_summarization_pipeline(correction_state)
        
        print("✅ СУММАРИЗАЦИЯ ЗАВЕРШЕНА")
        print(f"📋 ИТОГОВАЯ СУММАРИЗАЦИЯ: {final_state.get('summary', 'ОТСУТСТВУЕТ')}")
        print()
        
        # Проверка результатов
        print("🔍 ПРОВЕРКА РЕЗУЛЬТАТОВ:")
        print("-"*40)
        
        corrected_text = correction_state.get('corrected_text', '')
        summary = final_state.get('summary', '')
        
        print(f"✅ Текст на входе коррекции: {'ПРИСУТСТВУЕТ' if test_text else 'ОТСУТСТВУЕТ'}")
        print(f"✅ Текст на выходе коррекции: {'ПРИСУТСТВУЕТ' if corrected_text else 'ОТСУТСТВУЕТ'}")
        print(f"✅ Текст на входе суммаризации: {'ПРИСУТСТВУЕТ' if corrected_text else 'ОТСУТСТВУЕТ'}")
        print(f"✅ Текст на выходе суммаризации: {'ПРИСУТСТВУЕТ' if summary else 'ОТСУТСТВУЕТ'}")
        
        if corrected_text:
            print(f"📏 Длина исправленного текста: {len(corrected_text)} символов")
        
        if summary:
            print(f"📏 Длина суммаризации: {len(summary)} символов")
        
        print()
        print("🎯 ТЕСТ ЗАВЕРШЕН!")
        
    except Exception as e:
        print(f"❌ ОШИБКА В ТЕСТЕ: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_text_flow())
