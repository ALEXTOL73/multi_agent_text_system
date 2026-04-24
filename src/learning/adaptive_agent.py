"""
Adaptive Self-Learning Agent System
Implements machine learning capabilities for agents to improve over time
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass


@dataclass
class LearningExperience:
    """Single learning experience with context and results"""
    input_text: str
    prompt_used: str
    output_result: str
    metrics: Dict[str, float]
    timestamp: str
    feedback_score: Optional[float] = None
    success_rating: Optional[str] = None


class AdaptiveAgent:
    """
    Self-learning agent that improves performance based on experience
    """
    
    def __init__(self, agent_name: str, learning_rate: float = 0.1):
        self.agent_name = agent_name
        self.learning_rate = learning_rate
        self.experience_history: List[LearningExperience] = []
        self.performance_history: List[Dict[str, float]] = []
        self.prompt_templates: Dict[str, str] = {}
        self.success_patterns: List[Dict[str, Any]] = []
        self.failure_patterns: List[Dict[str, Any]] = []
        
        # Learning parameters
        self.min_experiences_for_learning = 10
        self.performance_threshold = 0.7
        
        self.logger = logging.getLogger(f"adaptive_agent.{agent_name}")
    
    def add_experience(self, experience: LearningExperience) -> None:
        """Add new learning experience"""
        self.experience_history.append(experience)
        
        # Track performance over time
        performance_summary = {
            'timestamp': experience.timestamp,
            'cor_score': experience.metrics.get('cor_score', 0),
            'sum_score': experience.metrics.get('sum_score', 0),
            'wer': experience.metrics.get('wer', 0),
            'lev_rating': experience.metrics.get('lev_rating', 0)
        }
        self.performance_history.append(performance_summary)
        
        # Trigger learning if enough experiences
        if len(self.experience_history) >= self.min_experiences_for_learning:
            self._learn_from_experiences()
    
    def _learn_from_experiences(self) -> None:
        """Analyze experiences and improve strategies"""
        self.logger.info(f"Starting learning process with {len(self.experience_history)} experiences")
        
        # Separate successful and failed experiences
        successful = [exp for exp in self.experience_history if self._is_successful(exp)]
        failed = [exp for exp in self.experience_history if not self._is_successful(exp)]
        
        # Analyze patterns
        self._analyze_success_patterns(successful)
        self._analyze_failure_patterns(failed)
        
        # Generate improved prompts
        self._generate_adaptive_prompts()
        
        # Update strategies
        self._update_strategies()
        
        self.logger.info("Learning cycle completed")
    
    def _is_successful(self, experience: LearningExperience) -> bool:
        """Determine if experience was successful"""
        # Use multiple metrics for success determination
        cor_score = experience.metrics.get('cor_score', 0)
        sum_score = experience.metrics.get('sum_score', 0)
        wer = experience.metrics.get('wer', 1.0)
        
        # Success criteria
        success = (
            cor_score >= self.performance_threshold or
            sum_score >= self.performance_threshold or
            wer <= 0.2  # Low WER indicates good correction
        )
        
        return success
    
    def _analyze_success_patterns(self, successful_experiences: List[LearningExperience]) -> None:
        """Analyze patterns in successful experiences"""
        if not successful_experiences:
            return
        
        # Extract common patterns
        patterns = {
            'prompt_types': {},
            'text_characteristics': {},
            'metric_ranges': {}
        }
        
        for exp in successful_experiences:
            # Analyze prompt types
            prompt_type = self._classify_prompt(exp.prompt_used)
            patterns['prompt_types'][prompt_type] = patterns['prompt_types'].get(prompt_type, 0) + 1
            
            # Analyze text characteristics
            text_length = len(exp.input_text)
            length_category = self._categorize_length(text_length)
            patterns['text_characteristics'][length_category] = patterns['text_characteristics'].get(length_category, 0) + 1
            
            # Analyze metric ranges
            for metric_name, value in exp.metrics.items():
                if isinstance(value, (int, float)):
                    range_category = self._categorize_metric_value(metric_name, value)
                    if metric_name not in patterns['metric_ranges']:
                        patterns['metric_ranges'][metric_name] = {}
                    patterns['metric_ranges'][metric_name][range_category] = patterns['metric_ranges'][metric_name].get(range_category, 0) + 1
        
        self.success_patterns = [patterns]
        self.logger.info(f"Identified {len(successful_experiences)} successful patterns")
    
    def _analyze_failure_patterns(self, failed_experiences: List[LearningExperience]) -> None:
        """Analyze patterns in failed experiences"""
        if not failed_experiences:
            return
        
        # Similar analysis for failures
        patterns = {
            'prompt_types': {},
            'text_characteristics': {},
            'common_issues': {}
        }
        
        for exp in failed_experiences:
            prompt_type = self._classify_prompt(exp.prompt_used)
            patterns['prompt_types'][prompt_type] = patterns['prompt_types'].get(prompt_type, 0) + 1
            
            text_length = len(exp.input_text)
            length_category = self._categorize_length(text_length)
            patterns['text_characteristics'][length_category] = patterns['text_characteristics'].get(length_category, 0) + 1
            
            # Identify common issues
            issues = self._identify_issues(exp)
            for issue in issues:
                patterns['common_issues'][issue] = patterns['common_issues'].get(issue, 0) + 1
        
        self.failure_patterns = [patterns]
        self.logger.info(f"Identified {len(failed_experiences)} failure patterns")
    
    def _generate_adaptive_prompts(self) -> None:
        """Generate improved prompts based on learning"""
        if not self.success_patterns:
            return
        
        # Find most successful prompt patterns
        best_prompt_types = self._get_best_prompt_types()
        
        # Generate adaptive prompt templates
        for prompt_type in best_prompt_types:
            adaptive_prompt = self._create_adaptive_prompt(prompt_type)
            self.prompt_templates[f"adaptive_{prompt_type}"] = adaptive_prompt
        
        self.logger.info(f"Generated {len(self.prompt_templates)} adaptive prompts")
    
    def _create_adaptive_prompt(self, prompt_type: str) -> str:
        """Create adaptive prompt based on learned patterns"""
        base_templates = {
            'correction': "Based on successful patterns, {instructions}",
            'summarization': "Following proven approaches, {instructions}",
            'few_shot': "Using these examples: {examples}. Apply to: {text}",
            'cot': "Step-by-step reasoning: {steps}. Apply to: {text}"
        }
        
        # Enhance with learned insights
        if prompt_type in self.success_patterns[0]['prompt_types']:
            success_rate = self.success_patterns[0]['prompt_types'][prompt_type] / len(self.experience_history)
            
            if success_rate > 0.8:  # Highly successful
                return f"OPTIMIZED {prompt_type.upper()}: " + base_templates.get(prompt_type, "{instructions}")
            elif success_rate > 0.6:  # Moderately successful
                return f"IMPROVED {prompt_type.upper()}: " + base_templates.get(prompt_type, "{instructions}")
            else:  # Needs improvement
                return f"EXPERIMENTAL {prompt_type.upper()}: " + base_templates.get(prompt_type, "{instructions}")
        
        return base_templates.get(prompt_type, "{instructions}")
    
    def get_adaptive_prompt(self, task_type: str, context: Dict[str, Any]) -> str:
        """Get best adaptive prompt for current context"""
        # Analyze current context
        text_length = len(context.get('input_text', ''))
        length_category = self._categorize_length(text_length)
        
        # Find best matching adaptive prompt
        best_prompt = None
        best_score = 0
        
        for prompt_name, prompt_template in self.prompt_templates.items():
            score = self._evaluate_prompt_fit(prompt_template, context)
            if score > best_score:
                best_score = score
                best_prompt = prompt_template
        
        if best_prompt:
            return self._customize_prompt(best_prompt, context)
        
        # Fallback to default
        return self._get_default_prompt(task_type)
    
    def _evaluate_prompt_fit(self, prompt_template: str, context: Dict[str, Any]) -> float:
        """Evaluate how well a prompt fits the current context"""
        score = 0.0
        
        # Check if prompt has worked well for similar contexts
        text_length = len(context.get('input_text', ''))
        length_category = self._categorize_length(text_length)
        
        # Look at historical performance
        similar_experiences = [
            exp for exp in self.experience_history
            if self._categorize_length(len(exp.input_text)) == length_category
        ]
        
        if similar_experiences:
            avg_performance = np.mean([
                exp.metrics.get('cor_score', 0) + exp.metrics.get('sum_score', 0)
                for exp in similar_experiences
            ])
            score = avg_performance / 2.0  # Normalize to 0-1 range
        
        return score
    
    def _customize_prompt(self, prompt_template: str, context: Dict[str, Any]) -> str:
        """Customize prompt template for specific context"""
        # Add context-specific instructions
        text_length = len(context.get('input_text', ''))
        
        if text_length > 1000:
            customization = " Focus on maintaining coherence in long texts."
        elif text_length < 100:
            customization = " Ensure comprehensive coverage despite brevity."
        else:
            customization = " Balance detail and conciseness."
        
        return prompt_template + customization
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        if not self.experience_history:
            return {'status': 'No learning data available'}
        
        # Calculate statistics
        total_experiences = len(self.experience_history)
        successful_experiences = sum(1 for exp in self.experience_history if self._is_successful(exp))
        success_rate = successful_experiences / total_experiences
        
        # Performance trends
        if len(self.performance_history) > 1:
            recent_performance = np.mean([
                hist.get('cor_score', 0) + hist.get('sum_score', 0)
                for hist in self.performance_history[-5:]
            ])
            early_performance = np.mean([
                hist.get('cor_score', 0) + hist.get('sum_score', 0)
                for hist in self.performance_history[:5]
            ])
            improvement = (recent_performance - early_performance) / early_performance * 100
        else:
            improvement = 0
        
        return {
            'total_experiences': total_experiences,
            'success_rate': success_rate,
            'improvement_percentage': improvement,
            'adaptive_prompts_count': len(self.prompt_templates),
            'success_patterns_count': len(self.success_patterns),
            'failure_patterns_count': len(self.failure_patterns),
            'learning_cycles': len(self.performance_history) // self.min_experiences_for_learning
        }
    
    def save_learning_state(self, file_path: str) -> None:
        """Save learning state to file"""
        state = {
            'agent_name': self.agent_name,
            'experience_history': [
                {
                    'input_text': exp.input_text,
                    'prompt_used': exp.prompt_used,
                    'output_result': exp.output_result,
                    'metrics': exp.metrics,
                    'timestamp': exp.timestamp,
                    'feedback_score': exp.feedback_score,
                    'success_rating': exp.success_rating
                }
                for exp in self.experience_history
            ],
            'performance_history': self.performance_history,
            'prompt_templates': self.prompt_templates,
            'success_patterns': self.success_patterns,
            'failure_patterns': self.failure_patterns,
            'learning_rate': self.learning_rate
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Learning state saved to {file_path}")
    
    def load_learning_state(self, file_path: str) -> None:
        """Load learning state from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # Restore experiences
            self.experience_history = [
                LearningExperience(
                    input_text=exp['input_text'],
                    prompt_used=exp['prompt_used'],
                    output_result=exp['output_result'],
                    metrics=exp['metrics'],
                    timestamp=exp['timestamp'],
                    feedback_score=exp.get('feedback_score'),
                    success_rating=exp.get('success_rating')
                )
                for exp in state['experience_history']
            ]
            
            self.performance_history = state['performance_history']
            self.prompt_templates = state['prompt_templates']
            self.success_patterns = state['success_patterns']
            self.failure_patterns = state['failure_patterns']
            self.learning_rate = state['learning_rate']
            
            self.logger.info(f"Learning state loaded from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading learning state: {e}")
    
    # Helper methods
    def _classify_prompt(self, prompt: str) -> str:
        """Classify prompt type"""
        prompt_lower = prompt.lower()
        if 'few-shot' in prompt_lower or 'example' in prompt_lower:
            return 'few_shot'
        elif 'step-by-step' in prompt_lower or 'reasoning' in prompt_lower:
            return 'cot'
        elif 'summar' in prompt_lower:
            return 'summarization'
        else:
            return 'correction'
    
    def _categorize_length(self, length: int) -> str:
        """Categorize text length"""
        if length < 100:
            return 'short'
        elif length < 500:
            return 'medium'
        else:
            return 'long'
    
    def _categorize_metric_value(self, metric_name: str, value: float) -> str:
        """Categorize metric value"""
        if metric_name in ['wer', 'delta_wer', 'delta_lev']:
            return 'low' if value < 0.1 else 'medium' if value < 0.3 else 'high'
        else:  # Higher is better
            return 'high' if value > 0.8 else 'medium' if value > 0.6 else 'low'
    
    def _identify_issues(self, experience: LearningExperience) -> List[str]:
        """Identify common issues in failed experiences"""
        issues = []
        
        if experience.metrics.get('wer', 0) > 0.3:
            issues.append('high_wer')
        
        if experience.metrics.get('cor_score', 0) < 0.5:
            issues.append('low_cor_score')
        
        if experience.metrics.get('sum_score', 0) < 0.5:
            issues.append('low_sum_score')
        
        return issues
    
    def _get_best_prompt_types(self) -> List[str]:
        """Get most successful prompt types"""
        if not self.success_patterns:
            return ['correction', 'summarization']
        
        prompt_types = self.success_patterns[0]['prompt_types']
        sorted_types = sorted(prompt_types.items(), key=lambda x: x[1], reverse=True)
        return [ptype for ptype, count in sorted_types[:3]]
    
    def _update_strategies(self) -> None:
        """Update agent strategies based on learning"""
        # This would update the actual agent behavior
        # Implementation depends on specific agent architecture
        pass
    
    def _get_default_prompt(self, task_type: str) -> str:
        """Get default prompt for task type"""
        defaults = {
            'correction': 'Correct the text for grammar, spelling, and punctuation errors.',
            'summarization': 'Summarize the text in 5 sentences or less.',
            'few_shot': 'Use examples to guide your response.',
            'cot': 'Think step-by-step to solve the problem.'
        }
        return defaults.get(task_type, 'Complete the task to the best of your ability.')
