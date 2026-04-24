"""
Агенты мультиагентной системы CorSumAgentsAI
"""

from .base_agent import BaseAgent
from .correction_prompt_generator import CorrectionPromptGenerator
from .corrector_ensemble import CorrectorEnsemble
from .corrector_aggregator import CorrectorAggregator
from .correction_judge import CorrectionJudge
from .summarization_prompt_generator import SummarizationPromptGenerator
from .summarizer_ensemble import SummarizerEnsemble
from .summarizer_aggregator import SummarizerAggregator
from .summarization_judge import SummarizationJudge

__all__ = [
    "BaseAgent",
    "CorrectionPromptGenerator",
    "CorrectorEnsemble", 
    "CorrectorAggregator",
    "CorrectionJudge",
    "SummarizationPromptGenerator",
    "SummarizerEnsemble",
    "SummarizerAggregator",
    "SummarizationJudge"
]
