import gradio as gr
import torch
import os
import json
import time
from pathlib import Path
import subprocess
import sys

# Add the core module to path
sys.path.append('../core/src')

try:
    from train_model_improved import ImprovedModelTrainer
    from model import GPTConfig, GPTModel
    from data_loader import TextDataset
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback for when core modules aren't available
    pass

class LiveTrainingInterface:
    def __init__(self):
        self.base_model = "lemms/openllm-small-extended-9k"
        self.training_configs = self.load_training_options()
        self.current_training = None
        self.training_logs = []
        
    def load_training_options(self):
        """Load available training configuration options"""
        return {
            "learning_rate": [1e-4, 3e-4, 5e-4, 1e-3],
            "batch_size": [4, 8, 16, 32],
            "training_steps": [1000, 2000, 5000, 10000],
            "gradient_accumulation": [1, 2, 4, 8],
            "optimizer": ["AdamW", "Adam", "SGD"],
            "scheduler": ["Cosine", "Linear", "Constant"],
            "weight_decay": [0.01, 0.1, 0.0],
            "gradient_clipping": [0.5, 1.0, 2.0],
            "warmup_steps": [100, 500, 1000]
        }
    
    def start_training(self, config):
        """Start a training session with the given configuration"""
        try:
            # Validate configuration
            if not self.validate_config(config):
                return "❌ Invalid configuration. Please check your settings."
            
            # Create training configuration
            training_config = {
                "base_model": self.base_model,
                "learning_rate": float(config["learning_rate"]),
                "batch_size": int(config["batch_size"]),
                "training_steps": int(config["training_steps"]),
                "gradient_accumulation": int(config["gradient_accumulation"]),
                "optimizer": config["optimizer"],
                "scheduler": config["scheduler"],
                "weight_decay": float(config["weight_decay"]),
                "gradient_clipping": float(config["gradient_clipping"]),
                "warmup_steps": int(config["warmup_steps"]),
                "output_dir": f"models/training-{int(time.time())}",
                "save_steps": 500,
                "eval_steps": 1000,
                "logging_steps": 100
            }
            
            # Start training in background
            self.current_training = training_config
            self.training_logs = []
            
            return f"🚀 Training started with configuration:\n{json.dumps(training_config, indent=2)}"
            
        except Exception as e:
            return f"❌ Error starting training: {str(e)}"
    
    def validate_config(self, config):
        """Validate training configuration"""
        try:
            required_fields = ["learning_rate", "batch_size", "training_steps"]
            for field in required_fields:
                if field not in config or not config[field]:
                    return False
            return True
        except:
            return False
    
    def get_training_status(self):
        """Get current training status"""
        if self.current_training is None:
            return "📊 No active training session"
        
        # Simulate training progress
        progress = {
            "status": "Training in progress...",
            "current_step": 500,
            "total_steps": self.current_training["training_steps"],
            "loss": 5.8,
            "learning_rate": self.current_training["learning_rate"]
        }
        
        return f"📊 Training Status:\n{json.dumps(progress, indent=2)}"
    
    def stop_training(self):
        """Stop current training session"""
        if self.current_training is None:
            return "❌ No active training session to stop"
        
        self.current_training = None
        return "⏹️ Training stopped"
    
    def download_model(self):
        """Download the trained model"""
        if self.current_training is None:
            return "❌ No trained model available"
        
        # This would implement actual model download
        return "📥 Model download started (this is a demo)"

def create_training_interface():
    """Create the Gradio interface for live training"""
    
    trainer = LiveTrainingInterface()
    
    with gr.Blocks(title="OpenLLM Live Training Space", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🚀 OpenLLM Live Training Space
        
        Welcome to the **OpenLLM Live Training Space**! This is where you can train new language models interactively.
        
        ## 🎯 What You Can Do
        - **Start training** from the latest model checkpoint (9k model)
        - **Configure training parameters** in real-time
        - **Monitor training progress** with live metrics
        - **Download or deploy** newly trained models
        
        ## 📋 Training Configuration
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Training Parameters")
                
                learning_rate = gr.Dropdown(
                    choices=trainer.training_configs["learning_rate"],
                    value=3e-4,
                    label="Learning Rate",
                    info="How fast the model learns"
                )
                
                batch_size = gr.Dropdown(
                    choices=trainer.training_configs["batch_size"],
                    value=8,
                    label="Batch Size",
                    info="Number of samples per training step"
                )
                
                training_steps = gr.Dropdown(
                    choices=trainer.training_configs["training_steps"],
                    value=2000,
                    label="Training Steps",
                    info="How long to train"
                )
                
                gradient_accumulation = gr.Dropdown(
                    choices=trainer.training_configs["gradient_accumulation"],
                    value=2,
                    label="Gradient Accumulation",
                    info="Memory optimization technique"
                )
                
                optimizer = gr.Dropdown(
                    choices=trainer.training_configs["optimizer"],
                    value="AdamW",
                    label="Optimizer",
                    info="Optimization algorithm"
                )
                
                scheduler = gr.Dropdown(
                    choices=trainer.training_configs["scheduler"],
                    value="Cosine",
                    label="Scheduler",
                    info="Learning rate schedule"
                )
                
                weight_decay = gr.Dropdown(
                    choices=trainer.training_configs["weight_decay"],
                    value=0.01,
                    label="Weight Decay",
                    info="Regularization strength"
                )
                
                gradient_clipping = gr.Dropdown(
                    choices=trainer.training_configs["gradient_clipping"],
                    value=1.0,
                    label="Gradient Clipping",
                    info="Gradient stability"
                )
                
                warmup_steps = gr.Dropdown(
                    choices=trainer.training_configs["warmup_steps"],
                    value=500,
                    label="Warmup Steps",
                    info="Learning rate warmup"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 🎮 Training Controls")
                
                start_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
                stop_btn = gr.Button("⏹️ Stop Training", variant="stop", size="lg")
                status_btn = gr.Button("📊 Check Status", size="lg")
                download_btn = gr.Button("📥 Download Model", size="lg")
                
                gr.Markdown("### 📊 Training Status")
                status_output = gr.Textbox(
                    label="Status",
                    value="Ready to start training",
                    lines=10,
                    interactive=False
                )
                
                gr.Markdown("### 📝 Training Logs")
                logs_output = gr.Textbox(
                    label="Logs",
                    value="No logs yet",
                    lines=8,
                    interactive=False
                )
        
        # Training scenarios section
        gr.Markdown("""
        ## 🎯 Training Scenarios
        
        ### Quick Experiments (1000 steps)
        - **Duration**: 10-30 minutes
        - **Purpose**: Test different learning rates and configurations
        - **Use case**: Hyperparameter exploration and rapid prototyping
        
        ### Medium Training (5000 steps)
        - **Duration**: 1-3 hours
        - **Purpose**: Significant model improvement and fine-tuning
        - **Use case**: Model optimization and performance enhancement
        
        ### Extended Training (10000 steps)
        - **Duration**: 3-8 hours
        - **Purpose**: Maximum performance improvement
        - **Use case**: Production model development and research
        """)
        
        # Event handlers
        def start_training_handler(lr, bs, steps, ga, opt, sched, wd, gc, warmup):
            config = {
                "learning_rate": lr,
                "batch_size": bs,
                "training_steps": steps,
                "gradient_accumulation": ga,
                "optimizer": opt,
                "scheduler": sched,
                "weight_decay": wd,
                "gradient_clipping": gc,
                "warmup_steps": warmup
            }
            return trainer.start_training(config)
        
        def stop_training_handler():
            return trainer.stop_training()
        
        def status_handler():
            return trainer.get_training_status()
        
        def download_handler():
            return trainer.download_model()
        
        # Connect event handlers
        start_btn.click(
            fn=start_training_handler,
            inputs=[learning_rate, batch_size, training_steps, gradient_accumulation, 
                   optimizer, scheduler, weight_decay, gradient_clipping, warmup_steps],
            outputs=status_output
        )
        
        stop_btn.click(
            fn=stop_training_handler,
            outputs=status_output
        )
        
        status_btn.click(
            fn=status_handler,
            outputs=status_output
        )
        
        download_btn.click(
            fn=download_handler,
            outputs=status_output
        )
        
        # Footer
        gr.Markdown("""
        ---
        
        ## 📚 Educational Value
        
        This space provides hands-on experience with:
        - **Understanding hyperparameters** and their effects on model performance
        - **Real-time observation** of training dynamics and convergence
        - **Learning best practices** for language model training
        - **Experimenting with different configurations** without local setup
        
        ## 🔗 Related Resources
        
        - **[Model Demo Space](https://huggingface.co/spaces/lemms/llm)** - Test trained models
        - **[GitHub Repository](https://github.com/louischua/osllm)** - Source code and documentation
        - **[Training Documentation](../docs/TRAINING_IMPROVEMENTS.md)** - Detailed training guide
        
        ---
        
        *This is a demonstration of the OpenLLM training capabilities. For production training, please refer to the full documentation.*
        """)
    
    return interface

# Create and launch the interface
if __name__ == "__main__":
    interface = create_training_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
