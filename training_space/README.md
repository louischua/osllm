---
title: OpenLLM Live Training Space
emoji: 🚀
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: gpl-3.0
---

# 🚀 OpenLLM Live Training Space

## 📚 What is This Space?

Welcome to the **OpenLLM Live Training Space**! This is an interactive web application where you can train new language models from existing checkpoints with customizable parameters. Think of it as a "training playground" where you can experiment with different training configurations in real-time.

### 🎯 What Makes This Special?

Unlike most AI demos that only allow you to use pre-trained models, **this space lets you actually train new models** with your own settings:

- **Interactive Training**: Configure and start training sessions in real-time
- **Parameter Experimentation**: Try different learning rates, batch sizes, and optimization settings
- **Live Monitoring**: Watch training progress and metrics as they happen
- **Educational**: Learn how different parameters affect model training
- **No Setup Required**: Train models without installing anything locally

## 🧠 Understanding Model Training

### What is Model Training?

Model training is like teaching a student by showing them millions of examples. The model learns patterns from the data and gradually improves its ability to predict what comes next.

**Example Training Process:**
1. **Input**: "The weather today is..."
2. **Model Prediction**: "sunny" (might be wrong initially)
3. **Correction**: "Actually, it's rainy"
4. **Learning**: Model adjusts its "thinking" to do better next time
5. **Repeat**: Millions of times until the model gets good at predictions

### How Does Training Work?

#### The Training Loop
1. **Forward Pass**: Model makes a prediction
2. **Loss Calculation**: Measure how wrong the prediction was
3. **Backward Pass**: Calculate how to adjust the model
4. **Parameter Update**: Update model weights to improve
5. **Repeat**: Continue until the model performs well

#### Key Parameters
- **Learning Rate**: How big steps to take when learning (too big = overshooting, too small = slow learning)
- **Batch Size**: How many examples to process at once (affects memory usage and training speed)
- **Training Steps**: How long to train (more steps = potentially better performance)
- **Optimizer**: Algorithm for updating model weights (AdamW, Adam, SGD)

## 🎮 How to Use This Space

### Step-by-Step Guide

#### 1. Configure Training Parameters
- **Learning Rate**: Start with 3e-4 (0.0003) for most cases
- **Batch Size**: Choose based on your memory constraints (8-16 is usually good)
- **Training Steps**: 
  - 1000 steps = Quick experiment (10-30 minutes)
  - 5000 steps = Medium training (1-3 hours)
  - 10000 steps = Extended training (3-8 hours)

#### 2. Start Training
- Click the "🚀 Start Training" button
- Watch the status updates in real-time
- Monitor loss values and training progress

#### 3. Monitor Progress
- **Loss**: Should decrease over time (lower is better)
- **Learning Rate**: May change based on scheduler
- **Steps**: Current progress through training

#### 4. Download Results
- Once training completes, download your trained model
- Use it for text generation or further fine-tuning

### Training Scenarios

#### Quick Experiments (1000 steps)
- **Best for**: Testing different learning rates and configurations
- **Duration**: 10-30 minutes
- **Use case**: Hyperparameter exploration and rapid prototyping

#### Medium Training (5000 steps)
- **Best for**: Significant model improvement and fine-tuning
- **Duration**: 1-3 hours
- **Use case**: Model optimization and performance enhancement

#### Extended Training (10000 steps)
- **Best for**: Maximum performance improvement
- **Duration**: 3-8 hours
- **Use case**: Production model development and research

## 📊 Understanding the Parameters

### Learning Parameters
- **Learning Rate**: Controls how fast the model learns
  - Too high: Model might overshoot and never converge
  - Too low: Training takes forever
  - Sweet spot: Usually between 1e-4 and 1e-3

- **Batch Size**: Number of examples processed together
  - Larger: More stable gradients, but uses more memory
  - Smaller: Less memory, but potentially less stable training

### Optimization Settings
- **Gradient Accumulation**: Simulates larger batch sizes with less memory
- **Optimizer**: Algorithm for updating weights
  - AdamW: Usually the best choice for transformers
  - Adam: Good general-purpose optimizer
  - SGD: Simple but may need more tuning

- **Scheduler**: How learning rate changes over time
  - Cosine: Smooth decrease, often works well
  - Linear: Straight-line decrease
  - Constant: No change (rarely used)

### Advanced Options
- **Weight Decay**: Prevents overfitting by penalizing large weights
- **Gradient Clipping**: Prevents exploding gradients
- **Warmup Steps**: Gradually increase learning rate at the start

## 🎓 Educational Value

### What You'll Learn

#### 1. Training Dynamics
- How loss decreases over time
- The relationship between learning rate and convergence
- When to stop training (avoiding overfitting)

#### 2. Hyperparameter Tuning
- How different parameters affect training
- The trade-offs between speed and quality
- Best practices for different scenarios

#### 3. Model Development
- The complete training workflow
- How to evaluate model performance
- When to use different training strategies

#### 4. Practical Skills
- Reading training logs and metrics
- Understanding model convergence
- Debugging training issues

### Learning Path

#### Beginner Level
1. Start with default parameters
2. Try different training step counts
3. Observe how loss changes over time

#### Intermediate Level
1. Experiment with different learning rates
2. Try different optimizers and schedulers
3. Understand the relationship between parameters

#### Advanced Level
1. Fine-tune all parameters for optimal performance
2. Understand the underlying training algorithms
3. Apply these concepts to your own projects

## 🔬 Research Applications

### What Can You Do With This?

#### 1. Hyperparameter Research
- Study how different parameters affect training
- Find optimal configurations for specific tasks
- Understand parameter interactions

#### 2. Training Methodologies
- Compare different optimization strategies
- Study learning rate schedules
- Research training stability techniques

#### 3. Model Development
- Prototype new training approaches
- Test different architectures
- Develop custom training pipelines

#### 4. Educational Research
- Study how people learn about ML
- Develop better teaching methods
- Create interactive learning experiences

## 🛠️ Technical Details

### Base Model
This space uses the **lemms/openllm-small-extended-9k** model as the starting point, which is our best-performing model with:
- **Architecture**: GPT-style transformer
- **Parameters**: ~35.8M
- **Training**: 9,000 steps on SQUAD dataset
- **Performance**: ~5.2 loss, ~177 perplexity

### Training Infrastructure
- **Framework**: PyTorch with custom training loop
- **Optimization**: AdamW optimizer with cosine scheduling
- **Memory Management**: Gradient checkpointing and accumulation
- **Monitoring**: Real-time loss and metric tracking

### Limitations
- **Demo Mode**: This is a demonstration of training capabilities
- **Resource Constraints**: Limited GPU time per session
- **Model Size**: Currently supports small models only
- **Dataset**: Uses pre-processed SQUAD dataset

## 🔗 Related Resources

### OpenLLM Project
- **[Model Demo Space](https://huggingface.co/spaces/lemms/llm)** - Test trained models
- **[GitHub Repository](https://github.com/louischua/osllm)** - Source code and documentation
- **[Training Documentation](../docs/TRAINING_IMPROVEMENTS.md)** - Detailed training guide

### Learning Resources
- **PyTorch Tutorials**: Official PyTorch documentation
- **Transformer Papers**: "Attention Is All You Need" and follow-ups
- **Training Guides**: Hugging Face training tutorials

### Community
- **GitHub Discussions**: Ask questions and share results
- **Discord/Slack**: Join our community chat
- **Twitter**: Follow for updates and announcements

## 📞 Support and Contact

### Getting Help
- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general help
- **Email**: louischua@gmail.com for private matters

### Contact Information
- **Author**: Louis Chua Bean Chong
- **GitHub**: https://github.com/louischua/openllm
- **Model Demo**: https://huggingface.co/spaces/lemms/llm

## 📄 License

This space is part of the OpenLLM project and is available under the GPLv3 license for open source use, with commercial licensing options available.

---

## 🎉 Start Training!

Ready to train your own language model? Configure your parameters and click "Start Training" to begin your AI learning journey!

**Remember**: This is a demonstration space. For production training, please refer to the full OpenLLM documentation and run training locally or on your own infrastructure.

---

*This space is maintained by Louis Chua Bean Chong and the open-source community. Your feedback and contributions are welcome!*
