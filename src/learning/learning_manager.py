"""
Learning Manager for Self-Improving Agents
Orchestrates the learning process across all agents
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from .adaptive_agent import AdaptiveAgent, LearningExperience


@dataclass
class LearningConfig:
    """Configuration for learning system"""
    learning_enabled: bool = True
    min_experiences_threshold: int = 10
    performance_improvement_threshold: float = 0.05
    learning_interval_hours: int = 6
    save_learning_state: bool = True
    auto_adapt_prompts: bool = True
    feedback_collection_enabled: bool = True


class LearningManager:
    """
    Manages learning across multiple adaptive agents
    """
    
    def __init__(self, config: Optional[LearningConfig] = None):
        self.config = config or LearningConfig()
        self.adaptive_agents: Dict[str, AdaptiveAgent] = {}
        self.global_learning_data: List[LearningExperience] = []
        self.learning_history: List[Dict[str, Any]] = []
        self.last_learning_cycle = None
        
        # Paths for learning data
        self.learning_dir = Path("data/learning")
        self.learning_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("learning_manager")
        
        # Initialize agents
        self._initialize_agents()
    
    def _initialize_agents(self) -> None:
        """Initialize adaptive agents for each agent type"""
        agent_types = [
            'corrector_ensemble',
            'summarizer_ensemble', 
            'corrector_aggregator',
            'summarizer_aggregator',
            'correction_judge',
            'summarization_judge'
        ]
        
        for agent_type in agent_types:
            self.adaptive_agents[agent_type] = AdaptiveAgent(
                agent_name=agent_type,
                learning_rate=self.config.learning_enabled
            )
            
            # Try to load existing learning state
            state_file = self.learning_dir / f"{agent_type}_learning.json"
            if state_file.exists():
                self.adaptive_agents[agent_type].load_learning_state(str(state_file))
    
    async def process_agent_result(self, agent_name: str, result_data: Dict[str, Any]) -> None:
        """Process agent result and add to learning experience"""
        if not self.config.learning_enabled:
            return
        
        try:
            # Extract learning experience from result
            experience = self._extract_learning_experience(agent_name, result_data)
            
            if experience:
                # Add to specific agent
                if agent_name in self.adaptive_agents:
                    self.adaptive_agents[agent_name].add_experience(experience)
                
                # Add to global learning data
                self.global_learning_data.append(experience)
                
                self.logger.info(f"Added learning experience for {agent_name}")
                
                # Trigger learning cycle if needed
                await self._check_learning_cycle()
                
        except Exception as e:
            self.logger.error(f"Error processing learning experience for {agent_name}: {e}")
    
    def _extract_learning_experience(self, agent_name: str, result_data: Dict[str, Any]) -> Optional[LearningExperience]:
        """Extract learning experience from agent result data"""
        try:
            # Get input text
            input_text = result_data.get('input_text', '')
            if not input_text:
                return None
            
            # Get prompt used
            prompt_used = result_data.get('prompt_used', result_data.get('best_prompt', ''))
            
            # Get output result
            output_result = result_data.get('output_result', result_data.get('corrected_text', result_data.get('summary', '')))
            
            # Get metrics
            metrics = {}
            if 'metrics' in result_data:
                metrics = result_data['metrics']
            elif 'metrics_correction' in result_data:
                metrics.update(result_data['metrics_correction'])
            elif 'summary_metrics' in result_data:
                metrics.update(result_data['summary_metrics'])
            
            # Ensure we have some metrics
            if not metrics:
                return None
            
            # Create learning experience
            experience = LearningExperience(
                input_text=input_text,
                prompt_used=prompt_used,
                output_result=output_result,
                metrics=metrics,
                timestamp=datetime.now().isoformat(),
                feedback_score=result_data.get('feedback_score'),
                success_rating=result_data.get('success_rating')
            )
            
            return experience
            
        except Exception as e:
            self.logger.error(f"Error extracting learning experience: {e}")
            return None
    
    async def _check_learning_cycle(self) -> None:
        """Check if it's time for a learning cycle"""
        if not self.config.learning_enabled:
            return
        
        # Check if enough time has passed since last learning cycle
        if self.last_learning_cycle:
            time_since_last = datetime.now() - self.last_learning_cycle
            if time_since_last < timedelta(hours=self.config.learning_interval_hours):
                return
        
        # Check if any agent has enough experiences
        for agent_name, agent in self.adaptive_agents.items():
            if len(agent.experience_history) >= self.config.min_experiences_threshold:
                await self._run_learning_cycle(agent_name)
                break  # Only run one cycle at a time
    
    async def _run_learning_cycle(self, agent_name: str) -> None:
        """Run learning cycle for specific agent"""
        try:
            self.logger.info(f"Starting learning cycle for {agent_name}")
            
            agent = self.adaptive_agents[agent_name]
            
            # Agent learns from its experiences
            if hasattr(agent, '_learn_from_experiences'):
                agent._learn_from_experiences()
            
            # Update global learning history
            learning_summary = {
                'agent_name': agent_name,
                'timestamp': datetime.now().isoformat(),
                'experiences_count': len(agent.experience_history),
                'success_rate': len([exp for exp in agent.experience_history if agent._is_successful(exp)]) / len(agent.experience_history),
                'adaptive_prompts_count': len(agent.prompt_templates),
                'statistics': agent.get_learning_statistics()
            }
            
            self.learning_history.append(learning_summary)
            
            # Save learning state
            if self.config.save_learning_state:
                await self._save_learning_state(agent_name)
            
            self.last_learning_cycle = datetime.now()
            
            self.logger.info(f"Learning cycle completed for {agent_name}")
            
        except Exception as e:
            self.logger.error(f"Error in learning cycle for {agent_name}: {e}")
    
    async def _save_learning_state(self, agent_name: str) -> None:
        """Save learning state for agent"""
        try:
            agent = self.adaptive_agents[agent_name]
            state_file = self.learning_dir / f"{agent_name}_learning.json"
            agent.save_learning_state(str(state_file))
            
        except Exception as e:
            self.logger.error(f"Error saving learning state for {agent_name}: {e}")
    
    def get_adaptive_prompt(self, agent_name: str, task_type: str, context: Dict[str, Any]) -> Optional[str]:
        """Get adaptive prompt for agent"""
        if not self.config.auto_adapt_prompts:
            return None
        
        if agent_name in self.adaptive_agents:
            return self.adaptive_agents[agent_name].get_adaptive_prompt(task_type, context)
        
        return None
    
    def collect_feedback(self, agent_name: str, experience_id: str, feedback_score: float, feedback_text: str = "") -> None:
        """Collect human feedback for learning"""
        if not self.config.feedback_collection_enabled:
            return
        
        try:
            # Find the experience and update feedback
            agent = self.adaptive_agents.get(agent_name)
            if agent:
                for experience in agent.experience_history:
                    if experience.timestamp == experience_id:
                        experience.feedback_score = feedback_score
                        break
                
                self.logger.info(f"Collected feedback for {agent_name}: {feedback_score}")
                
        except Exception as e:
            self.logger.error(f"Error collecting feedback: {e}")
    
    def get_learning_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive learning dashboard data"""
        dashboard_data = {
            'learning_enabled': self.config.learning_enabled,
            'total_experiences': len(self.global_learning_data),
            'learning_cycles': len(self.learning_history),
            'last_learning_cycle': self.last_learning_cycle.isoformat() if self.last_learning_cycle else None,
            'agents': {}
        }
        
        # Add agent-specific data
        for agent_name, agent in self.adaptive_agents.items():
            dashboard_data['agents'][agent_name] = agent.get_learning_statistics()
        
        # Add global performance trends
        if len(self.learning_history) > 1:
            recent_cycles = self.learning_history[-5:]
            early_cycles = self.learning_history[:5]
            
            recent_avg_success = sum(cycle['success_rate'] for cycle in recent_cycles) / len(recent_cycles)
            early_avg_success = sum(cycle['success_rate'] for cycle in early_cycles) / len(early_cycles)
            
            dashboard_data['global_improvement'] = (recent_avg_success - early_avg_success) / early_avg_success * 100
        else:
            dashboard_data['global_improvement'] = 0
        
        return dashboard_data
    
    async def optimize_system_performance(self) -> Dict[str, Any]:
        """Optimize overall system performance based on learning"""
        optimization_results = {
            'prompt_optimizations': [],
            'parameter_adjustments': [],
            'strategy_changes': []
        }
        
        if not self.config.learning_enabled:
            return optimization_results
        
        try:
            # Analyze all agents' performance
            for agent_name, agent in self.adaptive_agents.items():
                if len(agent.experience_history) < self.config.min_experiences_threshold:
                    continue
                
                # Identify optimization opportunities
                optimizations = await self._identify_optimizations(agent_name, agent)
                optimization_results['prompt_optimizations'].extend(optimizations['prompts'])
                optimization_results['parameter_adjustments'].extend(optimizations['parameters'])
                optimization_results['strategy_changes'].extend(optimizations['strategies'])
            
            self.logger.info(f"System optimization completed: {len(optimization_results['prompt_optimizations'])} optimizations identified")
            
        except Exception as e:
            self.logger.error(f"Error in system optimization: {e}")
        
        return optimization_results
    
    async def _identify_optimizations(self, agent_name: str, agent: AdaptiveAgent) -> Dict[str, List[str]]:
        """Identify optimization opportunities for agent"""
        optimizations = {
            'prompts': [],
            'parameters': [],
            'strategies': []
        }
        
        try:
            # Analyze recent performance trends
            recent_experiences = agent.experience_history[-20:]
            if len(recent_experiences) < 10:
                return optimizations
            
            # Calculate recent success rate
            recent_success_rate = sum(1 for exp in recent_experiences if agent._is_successful(exp)) / len(recent_experiences)
            
            # Identify underperforming areas
            if recent_success_rate < 0.6:
                optimizations['prompts'].append(f"Generate new adaptive prompts for {agent_name} - low success rate")
            
            # Check for specific metric issues
            avg_wer = sum(exp.metrics.get('wer', 0) for exp in recent_experiences) / len(recent_experiences)
            if avg_wer > 0.3:
                optimizations['parameters'].append(f"Adjust correction parameters for {agent_name} - high WER")
            
            avg_cor_score = sum(exp.metrics.get('cor_score', 0) for exp in recent_experiences) / len(recent_experiences)
            if avg_cor_score < 0.5:
                optimizations['strategies'].append(f"Revise correction strategy for {agent_name} - low CorScore")
            
        except Exception as e:
            self.logger.error(f"Error identifying optimizations for {agent_name}: {e}")
        
        return optimizations
    
    def export_learning_data(self, file_path: str) -> None:
        """Export all learning data for analysis"""
        try:
            export_data = {
                'config': {
                    'learning_enabled': self.config.learning_enabled,
                    'min_experiences_threshold': self.config.min_experiences_threshold,
                    'learning_interval_hours': self.config.learning_interval_hours
                },
                'global_learning_data': [
                    {
                        'agent_name': agent_name,
                        'input_text': exp.input_text[:100] + '...',  # Truncate for export
                        'prompt_used': exp.prompt_used[:100] + '...',
                        'metrics': exp.metrics,
                        'timestamp': exp.timestamp,
                        'feedback_score': exp.feedback_score,
                        'success_rating': exp.success_rating
                    }
                    for exp in self.global_learning_data
                ],
                'learning_history': self.learning_history,
                'agent_statistics': {
                    agent_name: agent.get_learning_statistics()
                    for agent_name, agent in self.adaptive_agents.items()
                }
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Learning data exported to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting learning data: {e}")
    
    def reset_learning(self, agent_name: Optional[str] = None) -> None:
        """Reset learning data for specific agent or all agents"""
        if agent_name and agent_name in self.adaptive_agents:
            # Reset specific agent
            self.adaptive_agents[agent_name].experience_history.clear()
            self.adaptive_agents[agent_name].performance_history.clear()
            self.adaptive_agents[agent_name].prompt_templates.clear()
            self.adaptive_agents[agent_name].success_patterns.clear()
            self.adaptive_agents[agent_name].failure_patterns.clear()
            
            # Remove learning state file
            state_file = self.learning_dir / f"{agent_name}_learning.json"
            if state_file.exists():
                state_file.unlink()
            
            self.logger.info(f"Reset learning for {agent_name}")
            
        else:
            # Reset all agents
            for name, agent in self.adaptive_agents.items():
                agent.experience_history.clear()
                agent.performance_history.clear()
                agent.prompt_templates.clear()
                agent.success_patterns.clear()
                agent.failure_patterns.clear()
                
                # Remove learning state file
                state_file = self.learning_dir / f"{name}_learning.json"
                if state_file.exists():
                    state_file.unlink()
            
            # Reset global data
            self.global_learning_data.clear()
            self.learning_history.clear()
            self.last_learning_cycle = None
            
            self.logger.info("Reset learning for all agents")
    
    def update_config(self, new_config: LearningConfig) -> None:
        """Update learning configuration"""
        self.config = new_config
        self.logger.info("Learning configuration updated")
