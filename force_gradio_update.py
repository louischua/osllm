#!/usr/bin/env python3
"""
Force Gradio Update Script

This script forces an update to the latest Gradio version to resolve caching issues.

Author: Louis Chua Bean Chong
License: GPLv3
"""

import subprocess
import sys
import os

def force_gradio_update():
    """Force update Gradio to the latest version."""
    print("🔄 Force updating Gradio to version 4.44.1...")
    
    try:
        # Uninstall current Gradio
        print("📦 Uninstalling current Gradio...")
        subprocess.run([
            sys.executable, "-m", "pip", "uninstall", "gradio", "-y"
        ], check=True)
        
        # Install specific version
        print("📦 Installing Gradio 4.44.1...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "gradio==4.44.1", "--no-cache-dir"
        ], check=True)
        
        # Verify installation
        print("🔍 Verifying Gradio installation...")
        result = subprocess.run([
            sys.executable, "-c", "import gradio; print(f'Gradio version: {gradio.__version__}')"
        ], capture_output=True, text=True, check=True)
        
        print(f"✅ {result.stdout.strip()}")
        print("🎉 Gradio update completed successfully!")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error updating Gradio: {e}")
        return False

if __name__ == "__main__":
    success = force_gradio_update()
    sys.exit(0 if success else 1)
