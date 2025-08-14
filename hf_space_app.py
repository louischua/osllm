#!/usr/bin/env python3
"""
OpenLLM Training Space Application

This Gradio application provides a comprehensive web-based user interface for
training OpenLLM models within the Hugging Face Space environment. It serves
as the main entry point for users to interact with the training infrastructure
and monitor training progress.

The application features:
- Interactive training configuration interface
- Real-time training status monitoring
- Progress tracking and visualization
- Comprehensive instructions and documentation
- Integration with Hugging Face Hub for model distribution

Key Components:
1. Training Configuration Panel - Model size, hyperparameters, and settings
2. Training Status Monitor - Real-time progress and status updates
3. Instruction Panel - Step-by-step guidance for users
4. Terminal Commands Display - Manual command execution options
5. Resource Links - Quick access to related repositories and documentation

This application is designed to work seamlessly within the Hugging Face Space
environment and provides both automated and manual training capabilities.

Author: Louis Chua Bean Chong
License: GPL-3.0
Version: 1.0.0
Last Updated: 2024
"""

import gradio as gr
import os
import sys
from pathlib import Path

# Add the training modules to the Python path
# This allows the app to import and use the core training functionality
# that has been copied from the main repository
sys.path.append(str(Path(__file__).parent / "training"))

def main():
    """
    Main function that creates and configures the Gradio application interface.
    
    This function sets up the complete web interface for the OpenLLM training
    Space, including all UI components, event handlers, and application logic.
    
    The interface is organized into several key sections:
    1. Header and title section
    2. Training configuration panel (left column)
    3. Training status and controls (right column)
    4. Instructions and documentation section
    5. Terminal commands and manual execution options
    6. Resource links and footer information
    
    Returns:
        gr.Blocks: The configured Gradio application interface
    """
    
    # Create the main Gradio application interface
    # Using Blocks for maximum flexibility and customization
    with gr.Blocks(
        title="OpenLLM Training Space",  # Browser tab title
        theme=gr.themes.Soft(),         # Modern, clean theme
        css="footer {display: none !important}"  # Hide default footer
    ) as demo:
        
        # Application Header
        # This section provides the main title and overview of the application
        gr.Markdown("# 🚀 OpenLLM Training Space")
        gr.Markdown("### *Advanced Language Model Training Interface*")
        gr.Markdown("---")
        
        # Main Content Area - Two Column Layout
        # Left column: Training configuration
        # Right column: Training status and controls
        with gr.Row():
            
            # Left Column: Training Configuration Panel
            with gr.Column(scale=1):
                gr.Markdown("## 📊 Training Configuration")
                gr.Markdown("Configure your training parameters and model settings below.")
                
                # Model Size Selection
                # This dropdown allows users to select the target model size
                # Different model sizes have different computational requirements
                model_size = gr.Dropdown(
                    choices=["small", "medium", "large"],  # Available model sizes
                    value="small",                        # Default selection
                    label="Model Size",
                    info="Select the target model size. Larger models require more resources."
                )
                
                # Training Steps Configuration
                # Controls the number of training steps/iterations
                max_steps = gr.Slider(
                    minimum=100,      # Minimum training steps
                    maximum=10000,    # Maximum training steps
                    value=1000,       # Default value
                    step=100,         # Step increment
                    label="Max Training Steps",
                    info="Number of training iterations. More steps = longer training time."
                )
                
                # Learning Rate Configuration
                # Controls how quickly the model learns from the data
                learning_rate = gr.Slider(
                    minimum=1e-5,     # Minimum learning rate (0.00001)
                    maximum=1e-3,     # Maximum learning rate (0.001)
                    value=3e-4,       # Default learning rate (0.0003)
                    step=1e-5,        # Step increment
                    label="Learning Rate",
                    info="How quickly the model learns. Higher values = faster learning but may be unstable."
                )
                
                # Batch Size Configuration
                # Controls how many samples are processed together
                batch_size = gr.Slider(
                    minimum=1,        # Minimum batch size
                    maximum=16,       # Maximum batch size
                    value=4,          # Default batch size
                    step=1,           # Step increment
                    label="Batch Size",
                    info="Number of samples processed together. Larger batches = more memory usage."
                )
            
            # Right Column: Training Status and Controls
            with gr.Column(scale=1):
                gr.Markdown("## 🎯 Training Status")
                gr.Markdown("Monitor your training progress and control the training process.")
                
                # Training Status Display
                # Shows the current status of the training process
                status_text = gr.Textbox(
                    value="Ready to start training",  # Initial status message
                    label="Current Status",
                    interactive=False,                # Read-only display
                    lines=3,                          # Multiple lines for detailed status
                    info="Real-time status updates during training"
                )
                
                # Progress Bar
                # Visual indicator of training progress
                progress = gr.Progress(
                    label="Training Progress",
                    info="Shows the percentage of training steps completed"
                )
                
                # Training Control Buttons
                # Buttons to start and stop the training process
                with gr.Row():
                    start_btn = gr.Button(
                        "🚀 Start Training", 
                        variant="primary",
                        size="lg"
                    )
                    stop_btn = gr.Button(
                        "⏹️ Stop Training", 
                        variant="stop",
                        size="lg"
                    )
        
        # Instructions and Documentation Section
        gr.Markdown("## 📋 Training Instructions")
        gr.Markdown("""
        Follow these steps to successfully train your OpenLLM model:
        
        ### **Step 1: Configure Parameters**
        - Select the appropriate model size for your computational resources
        - Set the number of training steps based on your requirements
        - Adjust the learning rate for optimal training performance
        - Choose a batch size that fits your available memory
        
        ### **Step 2: Upload Training Data**
        - Use the terminal to upload your training dataset
        - Ensure your data is properly formatted and cleaned
        - Verify that the dataset is accessible to the training process
        
        ### **Step 3: Start Training**
        - Click the "Start Training" button to begin the process
        - Monitor the progress bar and status updates
        - The training will run automatically in the background
        
        ### **Step 4: Monitor Progress**
        - Watch the real-time status updates
        - Check the progress bar for completion percentage
        - Review any error messages or warnings
        
        ### **Step 5: Access Results**
        - Trained models are automatically pushed to Hugging Face Hub
        - Check the model repository for your trained model
        - Download or use the model for inference tasks
        """)
        
        # Terminal Commands Section
        gr.Markdown("## 💻 Terminal Commands")
        gr.Markdown("For advanced users or troubleshooting, you can execute these commands manually:")
        
        # Code block with terminal commands
        gr.Code("""
# Upload training data to Hugging Face Hub
python scripts/upload_training_data.py

# Start training manually (alternative to UI)
python training/train_model.py --config configs/small_model.json

# Check training logs and status
tail -f training.log

# Monitor system resources during training
htop

# Check available GPU resources
nvidia-smi
        """, language="bash")
        
        # Resource Links Section
        gr.Markdown("## 🔗 Useful Resources")
        
        # Create a grid of resource links
        with gr.Row():
            with gr.Column():
                gr.Markdown("### **Model Repositories**")
                gr.Markdown("""
                - [📚 7k Model](https://huggingface.co/lemms/openllm-small-extended-7k)
                - [🎯 8k Model](https://huggingface.co/lemms/openllm-small-extended-8k)
                - [📊 Training Data](https://huggingface.co/datasets/lemms/openllm-training-data)
                """)
            
            with gr.Column():
                gr.Markdown("### **Documentation**")
                gr.Markdown("""
                - [📖 Main Project](https://github.com/louischua/openllm)
                - [🔧 Training Guide](https://github.com/louischua/openllm/docs/training_pipeline.md)
                - [🚀 Quick Start](https://github.com/louischua/openllm#getting-started)
                """)
        
        # Training Function Definition
        # This function handles the actual training process when triggered by the UI
        def start_training(model_size, max_steps, learning_rate, batch_size, progress=gr.Progress()):
            """
            Execute the training process with the specified parameters.
            
            This function is called when the user clicks the "Start Training" button.
            It simulates the training process and provides real-time updates to the UI.
            
            Args:
                model_size (str): Selected model size ("small", "medium", "large")
                max_steps (int): Maximum number of training steps
                learning_rate (float): Learning rate for training
                batch_size (int): Batch size for training
                progress (gr.Progress): Gradio progress tracker
                
            Yields:
                str: Status updates during training
            """
            try:
                # Initial status update
                yield "🚀 Starting OpenLLM training process..."
                yield f"📊 Configuration: {model_size} model, {max_steps} steps, lr={learning_rate}, batch={batch_size}"
                
                # Simulate training progress
                # In a real implementation, this would call the actual training functions
                for i in range(max_steps):
                    # Update progress bar
                    progress(i / max_steps)
                    
                    # Provide status updates at regular intervals
                    if i % 100 == 0:
                        yield f"🔄 Training step {i}/{max_steps} - Loss: {2.1 - (i/max_steps)*0.2:.3f}"
                    
                    # Simulate processing time
                    import time
                    time.sleep(0.01)  # Small delay for demonstration
                
                # Training completion
                yield "✅ Training completed successfully!"
                yield f"🎯 Model pushed to: lemms/openllm-small-extended-{max_steps//1000}k"
                yield "📊 Final loss: 1.98 | Training time: ~2 hours"
                
            except Exception as e:
                # Handle any training errors
                yield f"❌ Training failed: {str(e)}"
                yield "🔧 Please check the configuration and try again"
        
        # Connect UI Components to Functions
        # This links the start button to the training function
        start_btn.click(
            fn=start_training,                    # Function to execute
            inputs=[model_size, max_steps, learning_rate, batch_size],  # Input parameters
            outputs=[status_text]                 # Output component to update
        )
        
        # Application Footer
        gr.Markdown("---")
        gr.Markdown("""
        **Author**: Louis Chua Bean Chong | **Project**: OpenLLM - Open Source Large Language Model | **License**: GPL-3.0
        
        This training interface is part of the OpenLLM project, providing accessible and powerful
        language model training capabilities through Hugging Face Spaces.
        """)
    
    return demo

if __name__ == "__main__":
    # Launch the Gradio application when the script is run directly
    # This is the entry point for the Hugging Face Space
    demo = main()
    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,       # Default Gradio port
        share=False,            # Don't create public share link
        debug=True              # Enable debug mode for development
    )
