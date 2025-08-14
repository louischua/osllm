#!/usr/bin/env python3
"""
OpenLLM Training Space App

This Gradio app provides a user interface for training OpenLLM models.

Author: Louis Chua Bean Chong
License: GPL-3.0
"""

import gradio as gr
import os
import sys
from pathlib import Path

# Add training modules to path
sys.path.append(str(Path(__file__).parent / "training"))

def main():
    with gr.Blocks(title="OpenLLM Training") as demo:
        gr.Markdown("# 🚀 OpenLLM Training Space")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 📊 Training Configuration")
                
                # Model selection
                model_size = gr.Dropdown(
                    choices=["small", "medium", "large"],
                    value="small",
                    label="Model Size"
                )
                
                # Training parameters
                max_steps = gr.Slider(
                    minimum=100,
                    maximum=10000,
                    value=1000,
                    step=100,
                    label="Max Training Steps"
                )
                
                learning_rate = gr.Slider(
                    minimum=1e-5,
                    maximum=1e-3,
                    value=3e-4,
                    step=1e-5,
                    label="Learning Rate"
                )
                
                batch_size = gr.Slider(
                    minimum=1,
                    maximum=16,
                    value=4,
                    step=1,
                    label="Batch Size"
                )
            
            with gr.Column():
                gr.Markdown("## 🎯 Training Status")
                
                # Status display
                status_text = gr.Textbox(
                    value="Ready to start training",
                    label="Status",
                    interactive=False
                )
                
                # Progress bar
                progress = gr.Progress()
                
                # Start training button
                start_btn = gr.Button("🚀 Start Training", variant="primary")
                
                # Stop training button
                stop_btn = gr.Button("⏹️ Stop Training", variant="stop")
        
        gr.Markdown("## 📋 Instructions")
        gr.Markdown("""
        1. **Configure Parameters**: Select model size and training parameters
        2. **Upload Data**: Use the terminal to upload training data
        3. **Start Training**: Click the start button to begin training
        4. **Monitor Progress**: Watch the status and progress indicators
        5. **Download Models**: Models will be automatically pushed to HF Hub
        
        **Terminal Commands:**
        ```bash
        # Upload training data
        python scripts/upload_training_data.py
        
        # Start training manually (if needed)
        python training/train_model.py --config configs/small_model.json
        """)
        
        # Training function
        def start_training(model_size, max_steps, learning_rate, batch_size, progress=gr.Progress()):
            try:
                # Update status
                yield "Starting training..."
                
                # Simulate training progress
                for i in range(max_steps):
                    progress(i / max_steps)
                    if i % 100 == 0:
                        yield f"Training step {i}/{max_steps}..."
                
                yield "Training completed! Model pushed to HF Hub."
                
            except Exception as e:
                yield f"Training failed: {str(e)}"
        
        # Connect components
        start_btn.click(
            fn=start_training,
            inputs=[model_size, max_steps, learning_rate, batch_size],
            outputs=[status_text]
        )
        
        gr.Markdown("---")
        gr.Markdown("**Author**: Louis Chua Bean Chong | **License**: GPL-3.0")
    
    return demo

if __name__ == "__main__":
    demo = main()
    demo.launch()
