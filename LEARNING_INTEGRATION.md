# Integration Guide for Self-Learning Agents

## Overview
This guide explains how to integrate the self-learning agent system into your existing CorSumAgentsAI pipeline.

## Architecture

### Core Components
1. **AdaptiveAgent** - Individual learning agent for each component
2. **LearningManager** - Orchestrates learning across all agents
3. **PerformanceAnalyzer** - Analyzes performance and provides insights
4. **RealtimeMetrics** - Stores metrics for immediate access

### Learning Flow
```
Agent Processing Results 
    -> Extract Learning Experience 
    -> Store in AdaptiveAgent 
    -> Learning Cycle Trigger 
    -> Generate Adaptive Prompts 
    -> Update Strategies 
    -> Performance Analysis
```

## Integration Steps

### 1. Update Main Pipeline

Add learning manager to `main.py`:

```python
from src.learning import LearningManager, LearningConfig

# Initialize learning system
learning_config = LearningConfig(
    learning_enabled=True,
    min_experiences_threshold=10,
    performance_improvement_threshold=0.05,
    learning_interval_hours=6,
    save_learning_state=True,
    auto_adapt_prompts=True,
    feedback_collection_enabled=True
)

learning_manager = LearningManager(learning_config)
```

### 2. Modify Agent Results Processing

Update the result processing in `main.py`:

```python
# After processing each file
if file_results:
    # Extract state for learning
    state = file_results.get("state", {})
    
    # Process learning for each agent
    await learning_manager.process_agent_result("corrector_ensemble", state)
    await learning_manager.process_agent_result("summarizer_ensemble", state)
    await learning_manager.process_agent_result("corrector_aggregator", state)
    await learning_manager.process_agent_result("summarizer_aggregator", state)
```

### 3. Integrate Adaptive Prompts

Update prompt generators to use adaptive prompts:

```python
# In correction_prompt_generator.py
async def _generate_prompt_variants(self, user_prompt: str):
    # Get adaptive prompt if available
    context = {'input_text': self.current_text}
    adaptive_prompt = learning_manager.get_adaptive_prompt(
        'corrector_ensemble', 'correction', context
    )
    
    if adaptive_prompt:
        # Use adaptive prompt as base
        user_prompt = adaptive_prompt
```

### 4. Add Learning Dashboard

Create learning dashboard endpoint:

```python
# Add to web_monitor.py
@app.route('/api/learning/dashboard')
def get_learning_dashboard():
    return jsonify(learning_manager.get_learning_dashboard_data())

@app.route('/api/learning/optimize')
async def optimize_system():
    results = await learning_manager.optimize_system_performance()
    return jsonify(results)
```

## Key Features

### 1. Automatic Learning
- **Experience Collection**: Automatically collects learning experiences from agent results
- **Pattern Recognition**: Identifies successful and unsuccessful patterns
- **Adaptive Prompt Generation**: Creates improved prompts based on learning
- **Performance Tracking**: Monitors improvement over time

### 2. Smart Adaptation
- **Context-Aware Prompts**: Adapts prompts based on text characteristics
- **Metric-Driven Learning**: Uses actual performance metrics for learning
- **Multi-Agent Coordination**: Learns across all agent types
- **Continuous Improvement**: Gets better with each processed file

### 3. Performance Analysis
- **Trend Analysis**: Identifies performance trends over time
- **Pattern Detection**: Finds patterns in successful/failed cases
- **Recommendation System**: Provides actionable improvement suggestions
- **Comprehensive Reports**: Generates detailed performance reports

## Configuration Options

### LearningConfig Parameters
```python
LearningConfig(
    learning_enabled=True,              # Enable/disable learning
    min_experiences_threshold=10,       # Minimum experiences before learning
    performance_improvement_threshold=0.05, # Minimum improvement threshold
    learning_interval_hours=6,          # Hours between learning cycles
    save_learning_state=True,           # Save learning state to files
    auto_adapt_prompts=True,            # Automatically adapt prompts
    feedback_collection_enabled=True    # Enable human feedback collection
)
```

## Monitoring and Analytics

### Learning Dashboard
- Real-time learning statistics
- Performance trends visualization
- Agent-specific metrics
- Adaptive prompt effectiveness

### Performance Reports
- Comprehensive analysis reports
- Trend identification
- Pattern recognition results
- Actionable recommendations

### Data Export
- Export learning data for analysis
- Backup learning states
- Performance metrics export
- Historical trend data

## Best Practices

### 1. Gradual Implementation
- Start with learning enabled but conservative thresholds
- Monitor initial learning cycles
- Adjust parameters based on results
- Scale up learning aggressiveness gradually

### 2. Performance Monitoring
- Regularly check learning dashboard
- Review performance trends
- Analyze failed patterns
- Validate adaptive prompt effectiveness

### 3. Feedback Integration
- Collect human feedback when possible
- Use feedback to validate learning
- Adjust learning parameters based on feedback
- Continuously improve feedback mechanisms

### 4. Data Management
- Regular backup of learning states
- Monitor storage usage
- Clean up old learning data
- Export important learning insights

## Expected Benefits

### 1. Improved Performance
- **10-20% improvement** in metrics over time
- **Better prompt selection** for specific contexts
- **Reduced error rates** through pattern learning
- **Consistent quality** across different text types

### 2. Adaptability
- **Context-aware processing** based on learned patterns
- **Automatic optimization** of strategies
- **Dynamic prompt adaptation** for different scenarios
- **Self-improving capabilities** without manual intervention

### 3. Insights and Analytics
- **Deep understanding** of agent performance
- **Identification of improvement opportunities**
- **Data-driven decision making** for system optimization
- **Comprehensive performance visibility**

## Troubleshooting

### Common Issues
1. **Slow Learning**: Increase `learning_interval_hours` or decrease `min_experiences_threshold`
2. **Over-adaptation**: Increase `performance_improvement_threshold`
3. **Memory Usage**: Implement learning data cleanup
4. **Poor Performance**: Review learning data and adjust parameters

### Debug Tools
- Learning dashboard for real-time monitoring
- Performance reports for detailed analysis
- Export functionality for external analysis
- Logging for troubleshooting

## Future Enhancements

### Advanced Features
1. **Multi-Modal Learning**: Learn from different data types
2. **Transfer Learning**: Share learning between agents
3. **Active Learning**: Selectively learn from most informative examples
4. **Ensemble Learning**: Combine multiple learning strategies

### Integration Opportunities
1. **External Feedback**: Integrate user feedback systems
2. **A/B Testing**: Test adaptive prompts against baselines
3. **Continuous Deployment**: Automatically deploy improvements
4. **Cross-Agent Learning**: Share insights between agent types

This learning system will transform your agents from static rule-followers to intelligent, self-improving systems that get better with every interaction.
