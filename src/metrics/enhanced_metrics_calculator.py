"""
Enhanced metrics calculator with BLEU and ROUGE support
"""

import re
import math
import numpy as np
from typing import Dict, Any, Optional, List
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import jiwer
from Levenshtein import distance as levenshtein_distance
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from collections import Counter

from .metrics_calculator import MetricsCalculator

class EnhancedMetricsCalculator(MetricsCalculator):
    """
    Enhanced metrics calculator with BLEU and ROUGE support
    """
    
    def __init__(self, 
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 perplexity_model_name: str = "gpt2"):
        """
        Initialize enhanced metrics calculator.
        
        Args:
            embedding_model_name: Name of embedding model
            perplexity_model_name: Name of perplexity model
        """
        super().__init__(embedding_model_name, perplexity_model_name)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def calculate_bleu(self, reference: str, hypothesis: str) -> float:
        """
        Calculate BLEU score.
        
        Args:
            reference: Reference text
            hypothesis: Hypothesis text
            
        Returns:
            BLEU score in range [0, 1]
        """
        try:
            # Tokenize texts
            reference_tokens = word_tokenize(reference.lower())
            hypothesis_tokens = word_tokenize(hypothesis.lower())
            
            # Calculate BLEU score
            bleu_score = sentence_bleu([reference_tokens], hypothesis_tokens)
            return bleu_score
            
        except Exception as e:
            self.logger.error(f"Error calculating BLEU: {e}")
            return 0.0
    
    def calculate_rouge_scores(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores.
        
        Args:
            reference: Reference text
            hypothesis: Hypothesis text
            
        Returns:
            Dictionary with ROUGE scores
        """
        try:
            scores = self.rouge_scorer.score(reference, hypothesis)
            
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougel': scores['rougeL'].fmeasure
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating ROUGE: {e}")
            return {
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougel': 0.0
            }
    
    async def calculate_enhanced_summary_metrics(self,
                                                original_text: str,
                                                summary_text: str,
                                                reference_summary: Optional[str] = None,
                                                lm_client=None) -> Dict[str, float]:
        """
        Calculate enhanced summary metrics including BLEU and ROUGE.
        
        Args:
            original_text: Original text
            summary_text: Generated summary
            reference_summary: Reference summary
            lm_client: LM Studio client for LLM metrics
            
        Returns:
            Dictionary with enhanced metrics
        """
        # Get base metrics
        metrics = await self.calculate_summary_metrics(
            original_text=original_text,
            summary_text=summary_text,
            reference_summary=reference_summary,
            lm_client=lm_client
        )
        
        # Add BLEU and ROUGE if reference is available
        if reference_summary:
            bleu_score = self.calculate_bleu(reference_summary, summary_text)
            rouge_scores = self.calculate_rouge_scores(reference_summary, summary_text)
            
            metrics.update({
                'bleu': bleu_score,
                'rouge1': rouge_scores['rouge1'],
                'rouge2': rouge_scores['rouge2'],
                'rougel': rouge_scores['rougel']
            })
        else:
            # Default values if no reference
            metrics.update({
                'bleu': 0.0,
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougel': 0.0
            })
        
        return metrics
    
    def calculate_enhanced_correction_metrics(self,
                                            original_text: str,
                                            corrected_text: str,
                                            reference_text: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate enhanced correction metrics.
        
        Args:
            original_text: Original text
            corrected_text: Corrected text
            reference_text: Reference text
            
        Returns:
            Dictionary with enhanced metrics
        """
        # Get base metrics
        metrics = self.calculate_correction_metrics(
            original_text=original_text,
            corrected_text=corrected_text,
            reference_text=reference_text
        )
        
        # Add additional metrics if reference is available
        if reference_text:
            # Calculate character-level metrics
            char_lev = self._calculate_char_level_levenshtein(reference_text, corrected_text)
            metrics['char_lev_rating'] = char_lev
            
            # Calculate word accuracy
            word_accuracy = self._calculate_word_accuracy(reference_text, corrected_text)
            metrics['word_accuracy'] = word_accuracy
        
        return metrics
    
    def _calculate_char_level_levenshtein(self, text1: str, text2: str) -> float:
        """
        Calculate character-level Levenshtein similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Character-level similarity in range [0, 1]
        """
        try:
            max_len = max(len(text1), len(text2))
            if max_len == 0:
                return 1.0
                
            distance = levenshtein_distance(text1, text2)
            similarity = 1 - (distance / max_len)
            return similarity
            
        except Exception as e:
            self.logger.error(f"Error calculating char-level Levenshtein: {e}")
            return 0.0
    
    def _calculate_word_accuracy(self, reference: str, hypothesis: str) -> float:
        """
        Calculate word-level accuracy.
        
        Args:
            reference: Reference text
            hypothesis: Hypothesis text
            
        Returns:
            Word accuracy in range [0, 1]
        """
        try:
            ref_words = word_tokenize(reference.lower())
            hyp_words = word_tokenize(hypothesis.lower())
            
            if not ref_words:
                return 0.0
                
            # Count exact word matches
            matches = sum(1 for word in hyp_words if word in ref_words)
            accuracy = matches / len(ref_words)
            
            return min(accuracy, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating word accuracy: {e}")
            return 0.0
