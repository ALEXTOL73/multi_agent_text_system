"""
Performance Analyzer for Self-Learning Agents
Analyzes agent performance and provides insights for improvement
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

from .adaptive_agent import LearningExperience


class PerformanceAnalyzer:
    """
    Analyzes agent performance and provides actionable insights
    """
    
    def __init__(self):
        self.logger = logging.getLogger("performance_analyzer")
    
    def analyze_agent_performance(self, experiences: List[LearningExperience]) -> Dict[str, Any]:
        """Comprehensive performance analysis for an agent"""
        if not experiences:
            return {'status': 'No data available'}
        
        analysis = {
            'overview': self._get_performance_overview(experiences),
            'trends': self._analyze_performance_trends(experiences),
            'patterns': self._identify_performance_patterns(experiences),
            'recommendations': self._generate_recommendations(experiences),
            'metric_analysis': self._analyze_metrics(experiences),
            'prompt_effectiveness': self._analyze_prompt_effectiveness(experiences)
        }
        
        return analysis
    
    def _get_performance_overview(self, experiences: List[LearningExperience]) -> Dict[str, Any]:
        """Get high-level performance overview"""
        total_experiences = len(experiences)
        
        # Calculate success rates
        successful_experiences = [exp for exp in experiences if self._is_successful(exp)]
        success_rate = len(successful_experiences) / total_experiences
        
        # Calculate average metrics
        avg_metrics = {}
        for metric_name in ['cor_score', 'sum_score', 'wer', 'lev_rating', 'geval', 'meteor']:
            values = [exp.metrics.get(metric_name) for exp in experiences if exp.metrics.get(metric_name) is not None]
            if values:
                avg_metrics[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # Time-based analysis
        timestamps = [datetime.fromisoformat(exp.timestamp) for exp in experiences]
        time_span = (max(timestamps) - min(timestamps)).days
        
        return {
            'total_experiences': total_experiences,
            'success_rate': success_rate,
            'successful_experiences': len(successful_experiences),
            'average_metrics': avg_metrics,
            'time_span_days': time_span,
            'experiences_per_day': total_experiences / max(time_span, 1)
        }
    
    def _analyze_performance_trends(self, experiences: List[LearningExperience]) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        # Sort experiences by timestamp
        sorted_experiences = sorted(experiences, key=lambda x: x.timestamp)
        
        # Calculate moving averages
        window_size = min(10, len(sorted_experiences) // 4)
        if window_size < 3:
            return {'status': 'Insufficient data for trend analysis'}
        
        trends = {}
        
        # Analyze each metric trend
        for metric_name in ['cor_score', 'sum_score', 'wer', 'lev_rating']:
            values = [exp.metrics.get(metric_name) for exp in sorted_experiences if exp.metrics.get(metric_name) is not None]
            
            if len(values) >= window_size:
                # Calculate moving averages
                moving_avg = self._calculate_moving_average(values, window_size)
                
                # Determine trend direction
                if len(moving_avg) >= 2:
                    trend_slope = self._calculate_trend_slope(moving_avg)
                    trend_direction = 'improving' if trend_slope > 0 else 'declining' if trend_slope < 0 else 'stable'
                    
                    trends[metric_name] = {
                        'direction': trend_direction,
                        'slope': trend_slope,
                        'current_value': moving_avg[-1],
                        'change_percentage': ((moving_avg[-1] - moving_avg[0]) / moving_avg[0]) * 100 if moving_avg[0] != 0 else 0
                    }
        
        return trends
    
    def _identify_performance_patterns(self, experiences: List[LearningExperience]) -> Dict[str, Any]:
        """Identify patterns in performance"""
        patterns = {
            'text_length_patterns': self._analyze_text_length_patterns(experiences),
            'prompt_type_patterns': self._analyze_prompt_type_patterns(experiences),
            'time_based_patterns': self._analyze_time_based_patterns(experiences),
            'metric_correlations': self._analyze_metric_correlations(experiences)
        }
        
        return patterns
    
    def _analyze_text_length_patterns(self, experiences: List[LearningExperience]) -> Dict[str, Any]:
        """Analyze performance patterns by text length"""
        length_categories = {'short': [], 'medium': [], 'long': []}
        
        for exp in experiences:
            text_length = len(exp.input_text)
            if text_length < 100:
                length_categories['short'].append(exp)
            elif text_length < 500:
                length_categories['medium'].append(exp)
            else:
                length_categories['long'].append(exp)
        
        patterns = {}
        for category, exps in length_categories.items():
            if exps:
                success_rate = sum(1 for exp in exps if self._is_successful(exp)) / len(exps)
                avg_cor_score = np.mean([exp.metrics.get('cor_score', 0) for exp in exps])
                
                patterns[category] = {
                    'count': len(exps),
                    'success_rate': success_rate,
                    'avg_cor_score': avg_cor_score
                }
        
        return patterns
    
    def _analyze_prompt_type_patterns(self, experiences: List[LearningExperience]) -> Dict[str, Any]:
        """Analyze performance patterns by prompt type"""
        prompt_types = {}
        
        for exp in experiences:
            prompt_type = self._classify_prompt(exp.prompt_used)
            if prompt_type not in prompt_types:
                prompt_types[prompt_type] = []
            prompt_types[prompt_type].append(exp)
        
        patterns = {}
        for prompt_type, exps in prompt_types.items():
            if exps:
                success_rate = sum(1 for exp in exps if self._is_successful(exp)) / len(exps)
                avg_cor_score = np.mean([exp.metrics.get('cor_score', 0) for exp in exps])
                
                patterns[prompt_type] = {
                    'count': len(exps),
                    'success_rate': success_rate,
                    'avg_cor_score': avg_cor_score,
                    'effectiveness_score': success_rate * avg_cor_score
                }
        
        return patterns
    
    def _analyze_time_based_patterns(self, experiences: List[LearningExperience]) -> Dict[str, Any]:
        """Analyze performance patterns by time"""
        # Group by hour of day
        hourly_performance = {}
        
        for exp in experiences:
            timestamp = datetime.fromisoformat(exp.timestamp)
            hour = timestamp.hour
            
            if hour not in hourly_performance:
                hourly_performance[hour] = []
            hourly_performance[hour].append(exp)
        
        patterns = {}
        for hour, exps in hourly_performance.items():
            if len(exps) >= 3:  # Only analyze if enough data
                success_rate = sum(1 for exp in exps if self._is_successful(exp)) / len(exps)
                avg_cor_score = np.mean([exp.metrics.get('cor_score', 0) for exp in exps])
                
                patterns[hour] = {
                    'count': len(exps),
                    'success_rate': success_rate,
                    'avg_cor_score': avg_cor_score
                }
        
        return patterns
    
    def _analyze_metric_correlations(self, experiences: List[LearningExperience]) -> Dict[str, Any]:
        """Analyze correlations between metrics"""
        # Extract metric values
        metrics_data = {}
        
        for exp in experiences:
            for metric_name, value in exp.metrics.items():
                if isinstance(value, (int, float)):
                    if metric_name not in metrics_data:
                        metrics_data[metric_name] = []
                    metrics_data[metric_name].append(value)
        
        # Calculate correlations
        correlations = {}
        metric_names = list(metrics_data.keys())
        
        for i, metric1 in enumerate(metric_names):
            for metric2 in metric_names[i+1:]:
                if len(metrics_data[metric1]) == len(metrics_data[metric2]):
                    correlation = np.corrcoef(metrics_data[metric1], metrics_data[metric2])[0, 1]
                    if not np.isnan(correlation):
                        correlations[f"{metric1}_vs_{metric2}"] = {
                            'correlation': correlation,
                            'strength': 'strong' if abs(correlation) > 0.7 else 'moderate' if abs(correlation) > 0.3 else 'weak'
                        }
        
        return correlations
    
    def _analyze_prompt_effectiveness(self, experiences: List[LearningExperience]) -> Dict[str, Any]:
        """Analyze effectiveness of different prompts"""
        prompt_performance = {}
        
        for exp in experiences:
            prompt_key = exp.prompt_used[:100]  # Use first 100 chars as key
            
            if prompt_key not in prompt_performance:
                prompt_performance[prompt_key] = []
            prompt_performance[prompt_key].append(exp)
        
        effectiveness = {}
        for prompt, exps in prompt_performance.items():
            if len(exps) >= 2:  # Only analyze if used multiple times
                success_rate = sum(1 for exp in exps if self._is_successful(exp)) / len(exps)
                avg_cor_score = np.mean([exp.metrics.get('cor_score', 0) for exp in exps])
                avg_sum_score = np.mean([exp.metrics.get('sum_score', 0) for exp in exps])
                
                effectiveness[prompt] = {
                    'usage_count': len(exps),
                    'success_rate': success_rate,
                    'avg_cor_score': avg_cor_score,
                    'avg_sum_score': avg_sum_score,
                    'overall_score': (success_rate + avg_cor_score + avg_sum_score) / 3
                }
        
        # Sort by overall score
        sorted_effectiveness = dict(
            sorted(effectiveness.items(), 
                  key=lambda x: x[1]['overall_score'], 
                  reverse=True)
        )
        
        return sorted_effectiveness
    
    def _generate_recommendations(self, experiences: List[LearningExperience]) -> List[str]:
        """Generate actionable recommendations based on performance analysis"""
        recommendations = []
        
        # Analyze overall performance
        success_rate = sum(1 for exp in experiences if self._is_successful(exp)) / len(experiences)
        
        if success_rate < 0.6:
            recommendations.append("Overall success rate is low. Consider reviewing prompt templates and strategies.")
        
        # Analyze prompt effectiveness
        prompt_analysis = self._analyze_prompt_effectiveness(experiences)
        if prompt_analysis:
            best_prompt = list(prompt_analysis.keys())[0]
            worst_prompt = list(prompt_analysis.keys())[-1]
            
            if len(prompt_analysis) > 1:
                best_score = prompt_analysis[best_prompt]['overall_score']
                worst_score = prompt_analysis[worst_prompt]['overall_score']
                
                if best_score - worst_score > 0.3:
                    recommendations.append(f"Significant performance difference between prompts. Consider using '{best_prompt[:50]}...' more frequently.")
        
        # Analyze text length patterns
        length_patterns = self._analyze_text_length_patterns(experiences)
        if length_patterns:
            # Find worst performing length category
            worst_category = min(length_patterns.items(), key=lambda x: x[1]['success_rate'])
            if worst_category[1]['success_rate'] < 0.5:
                recommendations.append(f"Performance is poor for {worst_category[0]} texts. Consider developing specialized prompts for this category.")
        
        # Analyze trends
        trends = self._analyze_performance_trends(experiences)
        declining_metrics = [metric for metric, data in trends.items() 
                           if data.get('direction') == 'declining' and data.get('slope', 0) < -0.01]
        
        if declining_metrics:
            recommendations.append(f"Declining performance detected in: {', '.join(declining_metrics)}. Review recent changes and consider strategy adjustments.")
        
        return recommendations
    
    def _calculate_moving_average(self, values: List[float], window_size: int) -> List[float]:
        """Calculate moving average"""
        moving_avg = []
        for i in range(len(values) - window_size + 1):
            window = values[i:i + window_size]
            moving_avg.append(np.mean(window))
        return moving_avg
    
    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate linear trend slope"""
        if len(values) < 2:
            return 0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate linear regression slope
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def _is_successful(self, experience: LearningExperience) -> bool:
        """Determine if experience was successful"""
        cor_score = experience.metrics.get('cor_score', 0)
        sum_score = experience.metrics.get('sum_score', 0)
        wer = experience.metrics.get('wer', 1.0)
        
        return (cor_score >= 0.7 or sum_score >= 0.7 or wer <= 0.2)
    
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
    
    def _analyze_metrics(self, experiences: List[LearningExperience]) -> Dict[str, Any]:
        """Detailed analysis of individual metrics"""
        metric_analysis = {}
        
        # Get all metric names
        all_metrics = set()
        for exp in experiences:
            all_metrics.update(exp.metrics.keys())
        
        for metric_name in all_metrics:
            values = [exp.metrics.get(metric_name) for exp in experiences if exp.metrics.get(metric_name) is not None]
            
            if values:
                metric_analysis[metric_name] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75)
                }
                
                # Determine if metric is healthy
                if metric_name in ['wer', 'delta_wer', 'delta_lev']:
                    # Lower is better
                    healthy_threshold = 0.2
                    metric_analysis[metric_name]['healthy'] = np.mean(values) <= healthy_threshold
                else:
                    # Higher is better
                    healthy_threshold = 0.7
                    metric_analysis[metric_name]['healthy'] = np.mean(values) >= healthy_threshold
        
        return metric_analysis
    
    def generate_performance_report(self, agent_name: str, experiences: List[LearningExperience], output_file: Optional[str] = None) -> str:
        """Generate comprehensive performance report"""
        analysis = self.analyze_agent_performance(experiences)
        
        report = f"""
# Performance Analysis Report for {agent_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
- Total Experiences: {analysis['overview']['total_experiences']}
- Success Rate: {analysis['overview']['success_rate']:.2%}
- Successful Experiences: {analysis['overview']['successful_experiences']}
- Time Span: {analysis['overview']['time_span_days']} days
- Experiences per Day: {analysis['overview']['experiences_per_day']:.1f}

## Performance Trends
"""
        
        for metric, trend_data in analysis['trends'].items():
            if isinstance(trend_data, dict) and 'direction' in trend_data:
                report += f"- {metric}: {trend_data['direction']} ({trend_data['change_percentage']:+.1f}%)\n"
        
        report += "\n## Recommendations\n"
        for i, rec in enumerate(analysis['recommendations'], 1):
            report += f"{i}. {rec}\n"
        
        report += "\n## Prompt Effectiveness\n"
        for prompt, data in list(analysis['prompt_effectiveness'].items())[:5]:
            report += f"- '{prompt[:50]}...': {data['overall_score']:.3f} ({data['usage_count']} uses)\n"
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(f"Performance report saved to {output_file}")
        
        return report
