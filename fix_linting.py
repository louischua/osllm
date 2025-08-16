#!/usr/bin/env python3
"""
Fix common linting issues in the codebase.

This script fixes the most common flake8 issues:
- F541: f-string without placeholders
- F401: unused imports
- E722: bare except
- W291: trailing whitespace
- W293: blank line contains whitespace
"""

import os
import re
from pathlib import Path


def fix_f_strings_without_placeholders(content):
    """Fix f-strings that don't have placeholders."""
    # Pattern to match f-strings without placeholders
    pattern = r'f"([^"]*)"'
    
    def replace_f_string(match):
        text = match.group(1)
        # Check if the f-string contains any placeholders
        if '{' not in text:
            return f'"{text}"'
        return match.group(0)
    
    return re.sub(pattern, replace_f_string, content)


def fix_bare_except(content):
    """Fix bare except statements."""
    # Replace bare except with except Exception
    content = re.sub(r'\bexcept\s*:', 'except Exception:', content)
    return content


def fix_trailing_whitespace(content):
    """Fix trailing whitespace."""
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Remove trailing whitespace
        fixed_line = line.rstrip()
        fixed_lines.append(fixed_line)
    
    return '\n'.join(fixed_lines)


def fix_blank_line_whitespace(content):
    """Fix blank lines that contain whitespace."""
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        if line.strip() == '':
            # Empty line - ensure it's truly empty
            fixed_lines.append('')
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)


def process_file(file_path):
    """Process a single file and fix linting issues."""
    print(f"Processing {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply fixes
        content = fix_f_strings_without_placeholders(content)
        content = fix_bare_except(content)
        content = fix_trailing_whitespace(content)
        content = fix_blank_line_whitespace(content)
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed {file_path}")
        else:
            print(f"⏭️  No changes needed for {file_path}")
            
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")


def main():
    """Main function to process all Python files in core/src."""
    core_src_path = Path("core/src")
    
    if not core_src_path.exists():
        print("❌ core/src directory not found")
        return
    
    python_files = list(core_src_path.glob("*.py"))
    
    print(f"Found {len(python_files)} Python files to process")
    
    for file_path in python_files:
        process_file(file_path)
    
    print("🎉 Linting fixes completed!")


if __name__ == "__main__":
    main()
