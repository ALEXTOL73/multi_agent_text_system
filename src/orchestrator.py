"""
Оркестратор для управления последовательностью агентов в мультиагентной системе
"""

import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from .agents.base_agent import BaseAgent
from .agents.correction_prompt_generator import CorrectionPromptGenerator
from .agents.corrector_ensemble import CorrectorEnsemble
from .agents.corrector_aggregator import CorrectorAggregator
from .agents.correction_judge import CorrectionJudge
from .agents.summarization_prompt_generator import SummarizationPromptGenerator
from .agents.summarizer_ensemble import SummarizerEnsemble
from .agents.summarizer_aggregator import SummarizerAggregator
from .agents.summarization_judge import SummarizationJudge
from .utils.lm_studio_client import LMStudioClient
from .utils.agent_memory import AgentMemory
from .metrics.metrics_calculator import MetricsCalculator
import config

class Orchestrator:
    """
    Оркестратор для управления последовательностью агентов
    """
    
    def __init__(self):
        """
        Инициализация оркестратора
        """
        self.logger = logging.getLogger("orchestrator")
        self.agents: Dict[str, BaseAgent] = {}
        self.lm_client: Optional[LMStudioClient] = None
        self.memory: Optional[AgentMemory] = None
        self.metrics_calculator: Optional[MetricsCalculator] = None
        
    async def initialize(self):
        """
        Инициализация компонентов оркестратора
        """
        self.logger.info("Инициализация оркестратора")
        
        try:
            # Инициализация клиента LM Studio
            self.lm_client = LMStudioClient(
                base_url=config.LMSTUDIO_URL,
                model_name=config.MODEL_NAME,
                timeout=config.REQUEST_TIMEOUT,
                max_retries=config.MAX_RETRIES
            )
            
            # Проверка соединения
            async with self.lm_client:
                if not await self.lm_client.test_connection():
                    raise Exception("Не удалось установить соединение с LM Studio")
            
            # Инициализация памяти
            self.memory = AgentMemory(
                memory_dir=config.MEMORY_DIR,
                cache_enabled=config.PROMPT_CACHE_ENABLED,
                cache_max_size=config.PROMPT_CACHE_MAX_SIZE
            )
            
            # Инициализация калькулятора метрик
            self.metrics_calculator = MetricsCalculator()
            
            # Инициализация агентов
            self._initialize_agents()
            
            self.logger.info("Оркестратор успешно инициализирован")
            
        except Exception as e:
            self.logger.error(f"Ошибка инициализации оркестратора: {e}")
            raise
    
    def _initialize_agents(self):
        """Инициализация всех агентов"""
        self.agents = {
            "correction_prompt_generator": CorrectionPromptGenerator(
                memory=self.memory,
                lm_client=self.lm_client
            ),
            "corrector_ensemble": CorrectorEnsemble(
                lm_client=self.lm_client,
                memory=self.memory,
                metrics_calculator=self.metrics_calculator
            ),
            "corrector_aggregator": CorrectorAggregator(
                lm_client=self.lm_client,
                metrics_calculator=self.metrics_calculator
            ),
            "correction_judge": CorrectionJudge(),
            "summarization_prompt_generator": SummarizationPromptGenerator(
                memory=self.memory,
                lm_client=self.lm_client
            ),
            "summarizer_ensemble": SummarizerEnsemble(
                lm_client=self.lm_client,
                memory=self.memory,
                metrics_calculator=self.metrics_calculator
            ),
            "summarizer_aggregator": SummarizerAggregator(
                lm_client=self.lm_client,
                metrics_calculator=self.metrics_calculator
            ),
            "summarization_judge": SummarizationJudge()
        }
        
        self.logger.info(f"Инициализировано {len(self.agents)} агентов")
    
    async def process_text(self, 
                          input_text: str,
                          reference_text: Optional[str] = None,
                          reference_summary: Optional[str] = None,
                          domain: str = "general",
                          enable_correction: bool = True,
                          enable_summarization: bool = True) -> Dict[str, Any]:
        """
        Основной метод обработки текста.
        
        Args:
            input_text: Входной текст
            reference_text: Эталонный текст для коррекции (опционально)
            reference_summary: Эталонная суммаризация (опционально)
            domain: Домен текста
            enable_correction: Включить коррекцию
            enable_summarization: Включить суммаризацию
            
        Returns:
            Результаты обработки
        """
        self.logger.info(f"Начало обработки текста (длина: {len(input_text)})")
        
        # Инициализация оркестратора если еще не сделано
        if not self.agents:
            await self.initialize()
        
        # Инициализация состояния
        state = {
            "input_text": input_text,
            "reference_text": reference_text,
            "reference_summary": reference_summary,
            "domain": domain,
            "correction_attempts": 0,
            "summary_attempts": 0,
            "start_time": datetime.now().isoformat()
        }
        
        try:
            # Этап коррекции
            if enable_correction:
                state = await self._run_correction_pipeline(state)
                
                # Проверка необходимости повтора коррекции
                while (state.get("correction_retry_needed", False) and 
                       state.get("correction_attempts", 0) < config.MAX_CORRECTION_ATTEMPTS):
                    self.logger.info(f"Повторная коррекция (попытка {state.get('correction_attempts', 0) + 1})")
                    state = await self._run_correction_pipeline(state)
            
            # Этап суммаризации
            if enable_summarization and state.get("corrected_text"):
                state = await self._run_summarization_pipeline(state)
                
                # Проверка необходимости повтора суммаризации
                while (state.get("summary_retry_needed", False) and 
                       state.get("summary_attempts", 0) < config.SUMMARIZATION_ATTEMPTS):
                    self.logger.info(f"Повторная суммаризация (попытка {state.get('summary_attempts', 0) + 1})")
                    state = await self._run_summarization_pipeline(state)
            
            # Финализация результатов
            state["end_time"] = datetime.now().isoformat()
            state["processing_time"] = self._calculate_processing_time(state)
            
            # Add separate times for correction and summarization
            if "correction_start_time" in state and "correction_end_time" in state:
                correction_time = self._calculate_time_difference(
                    state["correction_start_time"], 
                    state["correction_end_time"]
                )
                state["correction_time"] = correction_time
                
            if "summarization_start_time" in state and "summarization_end_time" in state:
                summary_time = self._calculate_time_difference(
                    state["summarization_start_time"], 
                    state["summarization_end_time"]
                )
                state["summary_time"] = summary_time
            
            # Save results (only text files, JSON disabled)
            # await self._save_results(state)  # JSON saving disabled
            
            # Save metrics to realtime store for web monitoring
            await self._save_metrics_to_realtime_store(state)
            
            self.logger.info("Обработка текста завершена")
            return state
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки текста: {e}")
            state["error"] = str(e)
            state["end_time"] = datetime.now().isoformat()
            return state
    
    async def _run_correction_pipeline(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Запуск конвейера коррекции.
        
        Args:
            state: Текущее состояние
            
        Returns:
            Обновленное состояние
        """
        # Add timing
        state["correction_start_time"] = datetime.now().isoformat()
        
        # Basic correction pipeline
        correction_agents = [
            "correction_prompt_generator",
            "corrector_ensemble"
        ]
        
        # Run basic pipeline first
        state = await self._run_agent_pipeline(correction_agents, state, "коррекции")
        
        # Check quality and decide on aggregation
        needs_aggregation = False
        if "metrics_correction" in state:
            quality = self._assess_correction_quality(state["metrics_correction"])
            self.logger.info(f"Correction quality: {quality}")
            if quality in ["ТРЕБУЕТСЯ УЛУЧШЕНИЕ", "УДОВЛЕТВОРИТЕЛЬНО"]:
                needs_aggregation = True
                self.logger.info("Low quality detected, enabling aggregation")
        
        # Add aggregator if needed
        if needs_aggregation:
            state = await self._run_agent_pipeline(["corrector_aggregator"], state, "агрегации коррекции")
        
        # Final quality assessment
        result_state = await self._run_agent_pipeline(["correction_judge"], state, "оценки коррекции")
        
        # Add end time
        result_state["correction_end_time"] = datetime.now().isoformat()
        
        return result_state
    
    def _assess_correction_quality(self, metrics: Dict[str, float]) -> str:
        """Assess correction quality based on metrics"""
        delta_wer = metrics.get("delta_wer", 0)
        delta_lev = metrics.get("delta_lev", 0)
        cor_score = metrics.get("cor_score", 0)
        
        # Use same logic as correction_judge
        thresholds = config.CORRECTION_THRESHOLDS
        
        if (delta_wer > thresholds["excellent"]["delta_wer"] and 
            delta_lev > thresholds["excellent"]["delta_lev"]):
            return "ОТЛИЧНО"
        elif (delta_wer > thresholds["good"]["delta_wer"] and 
              delta_lev > thresholds["good"]["delta_lev"]):
            return "ХОРОШО"
        elif delta_wer > thresholds["satisfactory"]["delta_wer"]:
            return "УДОВЛЕТВОРИТЕЛЬНО"
        else:
            return "ТРЕБУЕТСЯ УЛУЧШЕНИЕ"
    
    def _assess_summarization_quality(self, metrics: Dict[str, float]) -> str:
        """Assess summarization quality based on metrics"""
        sum_score = metrics.get("sum_score", 0)
        geval_score = metrics.get("geval", 0)
        llm_judge_score = metrics.get("llm_judge", 0)
        
        # Quality assessment based on SumScore and other metrics
        if sum_score >= 0.8 and geval_score >= 0.8 and llm_judge_score >= 7:
            return "ОТЛИЧНО"
        elif sum_score >= 0.6 and geval_score >= 0.6 and llm_judge_score >= 6:
            return "ХОРОШО"
        elif sum_score >= 0.4 and geval_score >= 0.4 and llm_judge_score >= 5:
            return "УДОВЛЕТВОРИТЕЛЬНО"
        else:
            return "ТРЕБУЕТСЯ УЛУЧШЕНИЕ"
    
    async def _run_summarization_pipeline(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Запуск конвейера суммаризации.
        
        Args:
            state: Текущее состояние
            
        Returns:
            Обновленное состояние
        """
        # Add timing
        state["summarization_start_time"] = datetime.now().isoformat()
        
        # Basic summarization pipeline
        summarization_agents = [
            "summarization_prompt_generator",
            "summarizer_ensemble"
        ]
        
        # Run basic pipeline first
        state = await self._run_agent_pipeline(summarization_agents, state, "суммаризации")
        
        # Check quality and decide on aggregation
        needs_aggregation = False
        if "metrics_summary" in state:
            quality = self._assess_summarization_quality(state["metrics_summary"])
            self.logger.info(f"Summarization quality: {quality}")
            if quality in ["ТРЕБУЕТСЯ УЛУЧШЕНИЕ", "УДОВЛЕТВОРИТЕЛЬНО"]:
                needs_aggregation = True
                self.logger.info("Low quality detected, enabling aggregation")
        
        # Add aggregator if needed
        if needs_aggregation:
            state = await self._run_agent_pipeline(["summarizer_aggregator"], state, "агрегации суммаризации")
        
        # Final quality assessment
        result_state = await self._run_agent_pipeline(["summarization_judge"], state, "оценки суммаризации")
        
        # Add end time
        result_state["summarization_end_time"] = datetime.now().isoformat()
        
        return result_state
    
    async def _run_agent_pipeline(self, 
                                 agents: List[str], 
                                 state: Dict[str, Any],
                                 pipeline_name: str) -> Dict[str, Any]:
        """
        Launching agent pipeline.
        
        Args:
            agents: List of agent names
            state: Current state
            pipeline_name: Pipeline name for logging
            
    Returns:
        Updated state
    """
        self.logger.info(f"Launching {pipeline_name} pipeline")
        
        for agent_name in agents:
            if agent_name not in self.agents:
                self.logger.error(f"Agent {agent_name} not found")
                continue
            
            agent = self.agents[agent_name]
            
            try:
                # Visual separator for agent work
                print(f"\n{'='*60}")
                print(f"  {agent_name.upper()}")
                print(f"{'='*60}")
                
                self.logger.info(f"Executing agent: {agent_name}")
                
                # Execute agent
                state = await agent.execute(state)
                
                # Print variants table after ensemble agents
                if agent_name == "corrector_ensemble":
                    from main import print_variant_table
                    if "ensemble_outputs" in state and "ensemble_prompts" in state and "ensemble_metrics" in state:
                        correction_variants = []
                        for i, (output, prompt_type, metrics) in enumerate(zip(
                            state["ensemble_outputs"], 
                            state["ensemble_prompts"], 
                            state["ensemble_metrics"]
                        )):
                            temp = state["ensemble_temperatures"][i] if "ensemble_temperatures" in state else 0.0
                            correction_variants.append({
                                "prompt_type": prompt_type,
                                "temperature": temp,
                                "metrics": metrics
                            })
                        print_variant_table(correction_variants, "correction")
                        
                elif agent_name == "summarizer_ensemble":
                    from main import print_variant_table
                    if "ensemble_summary_outputs" in state and "ensemble_summary_prompts" in state and "ensemble_summary_metrics" in state:
                        summary_variants = []
                        for i, (output, prompt_type, metrics) in enumerate(zip(
                            state["ensemble_summary_outputs"], 
                            state["ensemble_summary_prompts"], 
                            state["ensemble_summary_metrics"]
                        )):
                            temp = state["ensemble_summary_temperatures"][i] if "ensemble_summary_temperatures" in state else 0.0
                            summary_variants.append({
                                "prompt_type": prompt_type,
                                "temperature": temp,
                                "metrics": metrics
                            })
                        print_variant_table(summary_variants, "summarization")
                    else:
                        self.logger.warning("Missing required keys for summary table")
                
                # Save intermediate state if enabled
                if config.SAVE_INTERMEDIATE_STATES:
                    await self._save_intermediate_state(state, agent_name)
                
                self.logger.info(f"Agent {agent_name} executed successfully")
                
            except Exception as e:
                self.logger.error(f"Error in agent {agent_name}: {e}")
                
                if config.STOP_ON_ERROR:
                    raise Exception(f"Error in {pipeline_name} pipeline: {e}")
                else:
                    # Continue execution if error is not critical
                    continue
        
        self.logger.info(f"Pipeline {pipeline_name} completed")
        return state
    
    def _calculate_processing_time(self, state: Dict[str, Any]) -> float:
        """
        Расчет времени обработки.
        
        Args:
            state: Состояние с временными метками
            
        Returns:
            Время обработки в секундах
        """
        try:
            start_time = datetime.fromisoformat(state["start_time"])
            end_time = datetime.fromisoformat(state["end_time"])
            return (end_time - start_time).total_seconds()
        except:
            return 0.0
    
    async def _save_results(self, state: Dict[str, Any]):
        """
        Сохранение результатов обработки.
        
        Args:
            state: Финальное состояние
        """
        try:
            # Создание директорий
            config.CORRECTION_DIR.mkdir(parents=True, exist_ok=True)
            config.SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
            config.CORRECTION_METRICS_DIR.mkdir(parents=True, exist_ok=True)
            config.SUMMARY_METRICS_DIR.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Сохранение результатов коррекции
            if "corrected_text" in state:
                correction_file = config.CORRECTION_DIR / f"correction_{timestamp}.json"
                correction_data = {
                    "input_text": state["input_text"],
                    "corrected_text": state["corrected_text"],
                    "metrics_correction": state.get("metrics_correction", {}),
                    "correction_quality": state.get("correction_quality"),
                    "timestamp": timestamp
                }
                
                # Add reference_text only if it exists
                if state.get("reference_text"):
                    correction_data["reference_text"] = state["reference_text"]
                
                with open(correction_file, 'w', encoding='utf-8') as f:
                    json.dump(correction_data, f, ensure_ascii=False, indent=2)
            
            # Сохранение результатов суммаризации
            if "summary" in state:
                summary_file = config.SUMMARY_DIR / f"summary_{timestamp}.json"
                summary_data = {
                    "input_text": state["corrected_text"],
                    "summary": state["summary"],
                    "summary_metrics": state.get("summary_metrics", {}),
                    "summary_quality": state.get("summary_quality"),
                    "timestamp": timestamp
                }
                
                # Add reference_summary only if it exists
                if state.get("reference_summary"):
                    summary_data["reference_summary"] = state["reference_summary"]
                
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(summary_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Результаты сохранены с меткой времени: {timestamp}")
            
        except Exception as e:
            self.logger.error(f"Ошибка сохранения результатов: {e}")
    
    async def _save_intermediate_state(self, state: Dict[str, Any], agent_name: str):
        """
        Сохранение промежуточного состояния.
        
        Args:
            state: Текущее состояние
            agent_name: Имя агента
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            intermediate_file = config.LOGS_DIR / f"intermediate_{agent_name}_{timestamp}.json"
            
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Ошибка сохранения промежуточного состояния: {e}")
    
    async def _save_metrics_to_realtime_store(self, state: Dict[str, Any]):
        """
        Сохранение метрик в realtime store для веб-мониторинга.
        
        Args:
            state: Состояние с метриками
        """
        try:
            from src.utils.realtime_metrics import realtime_store
            
            # Prepare metrics data
            metrics_data = {}
            
            # Correction metrics
            if "metrics_correction" in state:
                correction_metrics = state["metrics_correction"]
                metrics_data.update({
                    'delta_WER': correction_metrics.get('delta_wer', 0),
                    'Lev_Rating': correction_metrics.get('lev_rating', 0),
                    'delta_Lev': correction_metrics.get('delta_lev', 0),
                    'Perplexity': correction_metrics.get('perplexity', 0),
                    'CorScore': correction_metrics.get('corScore', 0),
                    'best_prompt_cor': state.get('best_prompt_cor', 'basic')
                })
            
            # Summarization metrics
            if "summary_metrics" in state:
                summary_metrics = state["summary_metrics"]
                metrics_data.update({
                    'G_Eval': summary_metrics.get('geval', 0),
                    'METEOR': summary_metrics.get('meteor', 0),
                    'LLM_Judge': summary_metrics.get('llm_judge', 0),
                    'BertScore': summary_metrics.get('bert_score', 0),
                    'SumScore': summary_metrics.get('sum_score', 0),
                    'best_prompt_sum': state.get('best_prompt_sum', 'basic')
                })
            
            # Add processing times
            if "correction_time" in state:
                metrics_data['correction_time'] = state['correction_time']
            if "summary_time" in state:
                metrics_data['summary_time'] = state['summary_time']
            
            # Add quality ratings
            if "correction_quality" in state:
                metrics_data['correction_quality'] = state['correction_quality']
            if "summary_quality" in state:
                metrics_data['summary_quality'] = state['summary_quality']
            
            # Add timestamp and filename
            timestamp = datetime.now().isoformat()
            metrics_data['timestamp'] = timestamp
            metrics_data['filename'] = state.get('filename', f"file_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            filename = metrics_data['filename']
            
            # Save to realtime store
            realtime_store.add_metrics(filename, metrics_data)
            
        except Exception as e:
            self.logger.error(f"Ошибка сохранения метрик в realtime store: {e}")

    async def cleanup(self):
        """
        Очистка ресурсов
        """
        try:
            if self.lm_client:
                await self.lm_client.close()
            
            self.logger.info("Ресурсы оркестратора очищены")
            
        except Exception as e:
            self.logger.error(f"Ошибка очистки ресурсов: {e}")
    
    def _calculate_time_difference(self, start_time: str, end_time: str) -> str:
        """
        Расчет разницы времени между start и end
        
        Args:
            start_time: Время начала в ISO формате
            end_time: Время окончания в ISO формате
            
        Returns:
            Строка с разницей времени
        """
        try:
            from datetime import datetime
            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            diff = end - start
            
            # Форматируем разницу
            total_seconds = int(diff.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            
            if hours > 0:
                return f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                return f"{minutes}m {seconds}s"
            else:
                return f"{seconds}s"
        except Exception as e:
            self.logger.error(f"Ошибка расчета разницы времени: {e}")
            return "N/A"
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Получение информации об агентах
        
        Returns:
            Словарь с информацией об агентах
        """
        agent_info = {}
        for name, agent in self.agents.items():
            agent_info[name] = {
                "class": agent.__class__.__name__,
                "description": getattr(agent, "description", ""),
                "enabled": getattr(agent, "enabled", True)
            }
        return agent_info
