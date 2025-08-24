#!/usr/bin/env python3
"""
Direct HF Space Fix Script

This script directly uploads the fixed app.py file to the Hugging Face Space
to resolve the Gradio compatibility issue without relying on GitHub Actions.

Author: Louis Chua Bean Chong
License: GPL-3.0
"""

import os
from pathlib import Path
from huggingface_hub import HfApi


def fix_hf_space():
    """Upload the fixed app.py to the HF Space."""

    print("🔧 Fixing HF Space with corrected app.py...")

    # Configuration
    space_repo = "lemms/openllm"
    source_file = "hf_space_app_fixed.py"
    target_file = "app.py"

    # Check if source file exists
    if not os.path.exists(source_file):
        print(f"❌ Source file not found: {source_file}")
        return False

    # Initialize HF API
    api = HfApi()

    try:
        # Upload the fixed app file
        print(f"📤 Uploading {source_file} as {target_file} to {space_repo}...")

        api.upload_file(
            path_or_fileobj=source_file,
            path_in_repo=target_file,
            repo_id=space_repo,
            repo_type="space",
            commit_message="fix: Resolve Gradio 4.44.1 compatibility issue - Remove unsupported 'info' parameter from gr.JSON()",
        )

        print(f"✅ Successfully uploaded fixed app.py to {space_repo}")
        print(f"🔗 Space URL: https://huggingface.co/spaces/{space_repo}")
        print("🔄 The Space should restart automatically with the fixed code")

        return True

    except Exception as e:
        print(f"❌ Error uploading to HF Space: {e}")
        print("   This might be due to authentication issues or permissions")
        return False


if __name__ == "__main__":
    fix_hf_space()
