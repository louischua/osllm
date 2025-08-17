#!/usr/bin/env python3
"""
OpenLLM Training Space - Main Application

This is the main entry point for the Hugging Face Space.
It provides a web interface for running OpenLLM training with authentication.

Author: Louis Chua Bean Chong
License: GPLv3
"""

import os
import sys
from pathlib import Path

import gradio as gr

# Import our authentication and training modules
try:
    from openllm_training_with_auth import OpenLLMTrainingManager
    from space_auth_test import test_space_authentication

    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    print(f"❌ Required modules not available: {e}")


def create_space_interface():
    """Create the Gradio interface for the Space."""

    def run_authentication_test():
        """Run the authentication test and return results."""
        try:
            if not MODULES_AVAILABLE:
                return "❌ Required modules not available. Please check deployment."

            # Capture output from authentication test
            import contextlib
            import io

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                success = test_space_authentication()

            result = output.getvalue()

            if success:
                return f"✅ Authentication Test Results:\n\n{result}"
            else:
                return f"❌ Authentication Test Failed:\n\n{result}"

        except Exception as e:
            return f"❌ Error running authentication test: {e}"

    def run_training(model_size, training_steps):
        """Run the OpenLLM training with authentication."""
        try:
            if not MODULES_AVAILABLE:
                return "❌ Required modules not available. Please check deployment."

            # Security mitigation: Input validation and sanitization
            if not isinstance(model_size, str) or model_size not in ["small", "medium", "large"]:
                return "❌ Invalid model size. Must be 'small', 'medium', or 'large'."

            if (
                not isinstance(training_steps, (int, float))
                or training_steps < 1000
                or training_steps > 50000
            ):
                return "❌ Invalid training steps. Must be between 1000 and 50000."

            # Sanitize inputs
            model_size = str(model_size).strip().lower()
            training_steps = int(float(training_steps))

            # Capture output from training
            import contextlib
            import io

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                training_manager = OpenLLMTrainingManager()
                repo_id = training_manager.run_training(model_size=model_size, steps=training_steps)

            result = output.getvalue()

            return f"✅ Training Results:\n\n{result}\n\n🎉 Model available at: https://huggingface.co/{repo_id}"

        except Exception as e:
            return f"❌ Error running training: {e}"

    def check_space_environment():
        """Check the Space environment and configuration."""
        try:
            # Check if we're in a Space
            space_vars = ["SPACE_ID", "SPACE_HOST", "SPACE_REPO_ID"]
            is_space = any(os.getenv(var) for var in space_vars)

            # Check HF_TOKEN
            hf_token = os.getenv("HF_TOKEN")

            result = "🔍 Space Environment Check:\n\n"

            if is_space:
                result += "✅ Running in Hugging Face Space environment\n"
                for var in space_vars:
                    value = os.getenv(var)
                    if value:
                        result += f"   - {var}: {value}\n"
            else:
                result += "ℹ️ Running in local environment\n"

            # Test Space's built-in authentication
            try:
                from huggingface_hub import whoami

                user_info = whoami()
                result += f"✅ Space built-in authentication working\n"
                result += f"   - User: {user_info['name']}\n"
                result += f"   - Full name: {user_info['fullname']}\n"
                result += f"   - Authentication: Space built-in token\n"
            except Exception as auth_error:
                result += f"❌ Space built-in authentication failed: {str(auth_error)[:50]}...\n"

                if hf_token:
                    result += f"✅ HF access token found: {hf_token[:8]}...{hf_token[-4:]}\n"
                    result += "   - Source: HF access token in Space settings\n"
                else:
                    result += "❌ HF access token not found\n"
                    result += "   - Please set HF_TOKEN in Space settings with HF access token\n"
                    result += "   - Or ensure Space has proper authentication permissions\n"

            result += f"\n📁 Available modules: {'✅' if MODULES_AVAILABLE else '❌'}"

            return result

        except Exception as e:
            return f"❌ Error checking environment: {e}"

    # Create the Gradio interface with security mitigations
    with gr.Blocks(
        title="OpenLLM Training Space",
        theme=gr.themes.Soft(),
        # Security mitigations
        analytics_enabled=False,  # Disable analytics
    ) as interface:
        gr.Markdown(
            """
        # 🚀 OpenLLM Training Space
        
        Welcome to the OpenLLM Training Space! This Space provides a complete environment for training OpenLLM models with automatic Hugging Face authentication and model upload.
        
                 ## 🔐 Authentication
         
         This Space uses HF access token for secure authentication. The HF_TOKEN is automatically available from your Space settings.
        
        ## 📋 Available Actions
        
        1. **Environment Check**: Verify Space configuration and authentication
        2. **Authentication Test**: Test Hugging Face authentication
        3. **Run Training**: Start OpenLLM training with automatic upload
        """
        )

        with gr.Tab("🔍 Environment Check"):
            gr.Markdown("Check the Space environment and configuration.")
            env_check_btn = gr.Button("Check Environment", variant="primary")
            env_output = gr.Textbox(label="Environment Status", lines=10, interactive=False)
            env_check_btn.click(check_space_environment, outputs=env_output)

        with gr.Tab("🔐 Authentication Test"):
            gr.Markdown("Test Hugging Face authentication using HF access token.")
            auth_test_btn = gr.Button("Run Authentication Test", variant="primary")
            auth_output = gr.Textbox(label="Authentication Results", lines=15, interactive=False)
            auth_test_btn.click(run_authentication_test, outputs=auth_output)

        with gr.Tab("🚀 Run Training"):
            gr.Markdown(
                """
            Start OpenLLM training with automatic model upload.
            
            **Training Parameters:**
            - **Model Size**: Choose the model size (small, medium, large)
            - **Training Steps**: Number of training steps (default: 8000)
            
            **Expected Results:**
            - Training will complete successfully
            - Model will be uploaded to Hugging Face Hub
            - Repository will be created with proper model files
            """
            )

            with gr.Row():
                model_size = gr.Dropdown(
                    choices=["small", "medium", "large"],
                    value="small",
                    label="Model Size",
                    info="Choose the model size for training",
                )
                training_steps = gr.Number(
                    value=8000,
                    label="Training Steps",
                    info="Number of training steps",
                    minimum=1000,
                    maximum=50000,
                )

            train_btn = gr.Button("Start Training", variant="primary", size="lg")
            train_output = gr.Textbox(label="Training Results", lines=20, interactive=False)

            train_btn.click(run_training, inputs=[model_size, training_steps], outputs=train_output)

        with gr.Tab("📚 Documentation"):
            gr.Markdown(
                """
            ## 📖 Available Documentation
            
            - **HUGGINGFACE_SPACE_SETUP_GUIDE.md**: Complete setup guide
            - **SPACE_AUTHENTICATION_SUMMARY.md**: Authentication summary
            - **SPACE_READY_SUMMARY.md**: Deployment summary
            
            ## 🔧 Available Scripts
            
            - **space_auth_test.py**: Authentication verification
            - **openllm_training_with_auth.py**: Complete training script
            - **integrate_auth_into_training.py**: Integration guide
            - **setup_hf_space_auth.py**: Space authentication setup
            - **verify_space_auth.py**: Space verification script
            
            ## 🎯 Quick Start
            
            1. Check the environment to verify configuration
            2. Run authentication test to ensure GitHub secrets are working
            3. Start training with your desired parameters
            4. Monitor the training progress and model upload
            
            ## 🔒 Security
            
            - HF_TOKEN is securely stored in GitHub repository secrets
            - No hardcoded tokens in any scripts
            - Automatic cleanup of test repositories
            - Proper error handling and logging
            """
            )

    return interface


if __name__ == "__main__":
    # Create and launch the interface
    interface = create_space_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        # Security mitigations for Gradio vulnerabilities
        allowed_paths=[],  # Restrict file access
        auth=None,  # Disable authentication to prevent code injection
        quiet=True,  # Reduce logging
    )
