"""
Мультиагентная система CorSumAgentsAI для коррекции и суммаризации текста
"""

__version__ = "1.0.0"
__author__ = "CorSumAgentsAI Team"
__description__ = "Multi-agent system for text correction and summarization using local LLMs"

from .orchestrator import Orchestrator

__all__ = ["Orchestrator"]
