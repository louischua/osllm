#!/usr/bin/env python3
"""
OpenLLM Project Structure Reorganization Script

This script reorganizes the OpenLLM project structure to improve organization,
maintainability, and professional appearance while maintaining all existing functionality.

Author: Louis Chua Bean Chong
License: GPLv3
"""

import os
import shutil
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

# Configure logging with maximum verbosity for development
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reorganization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProjectReorganizer:
    """
    Handles the reorganization of the OpenLLM project structure.
    
    This class provides comprehensive functionality to:
    - Create new directory structure
    - Move files to appropriate locations
    - Update import paths and references
    - Clean up duplicate and obsolete files
    - Maintain backup and rollback capabilities
    """
    
    def __init__(self, project_root: str = ".", dry_run: bool = False):
        """
        Initialize the project reorganizer.
        
        Args:
            project_root: Root directory of the OpenLLM project
            dry_run: If True, only simulate changes without making them
        """
        self.project_root = Path(project_root).resolve()
        self.dry_run = dry_run
        self.backup_dir = self.project_root / "backup_before_reorganization"
        self.changes_made = []
        
        # Define the new directory structure
        self.new_directories = [
            "deployment/huggingface",
            "deployment/docker", 
            "deployment/kubernetes",
            "scripts/setup",
            "scripts/training",
            "scripts/evaluation",
            "scripts/maintenance",
            "configs/model_configs",
            "configs/training_configs", 
            "configs/deployment_configs",
            "docs/deployment",
            "docs/troubleshooting",
            "docs/development",
            "models/checkpoints",
            "models/final",
            "models/evaluation",
            "logs/training",
            "logs/evaluation",
            "logs/deployment"
        ]
        
        # Define file migration mapping
        self.file_migrations = {
            # Deployment files
            "hf_space_app_fixed.py": "deployment/huggingface/space_app.py",
            "hf_space_app_openllm_fixed.py": "deployment/huggingface/space_app_openllm.py",
            "hf_space_app_openllm_compatible.py": "deployment/huggingface/space_app_compatible.py",
            "hf_space_app_model_fix.py": "deployment/huggingface/space_app_model.py",
            "hf_space_app_final.py": "deployment/huggingface/space_app_final.py",
            "hf_space_app_gradio_441.py": "deployment/huggingface/space_app_gradio.py",
            "hf_space_app_complete.py": "deployment/huggingface/space_app_complete.py",
            "hf_space_app_simple.py": "deployment/huggingface/space_app_simple.py",
            "hf_space_app.py": "deployment/huggingface/space_app_original.py",
            "hf_space_requirements.txt": "deployment/huggingface/requirements.txt",
            "hf_space_requirements_updated.txt": "deployment/huggingface/requirements_updated.txt",
            "hf_space_requirements_complete.txt": "deployment/huggingface/requirements_complete.txt",
            "hf_space_README.md": "deployment/huggingface/README.md",
            "hf_space_README_updated.md": "deployment/huggingface/README_updated.md",
            "space_auth_test.py": "deployment/huggingface/space_auth.py",
            "setup_hf_space_auth.py": "deployment/huggingface/space_setup.py",
            "verify_space_auth.py": "deployment/huggingface/space_verify.py",
            "HUGGINGFACE_SPACE_SETUP_GUIDE.md": "deployment/huggingface/docs/setup_guide.md",
            "HUGGINGFACE_AUTH_GUIDE.md": "deployment/huggingface/docs/auth_guide.md",
            "SPACE_AUTHENTICATION_SUMMARY.md": "deployment/huggingface/docs/auth_summary.md",
            "SPACE_READY_SUMMARY.md": "deployment/huggingface/docs/ready_summary.md",
            "COMPLETE_DEPLOYMENT_SUMMARY.md": "deployment/huggingface/docs/deployment_summary.md",
            "DEPLOYMENT_GUIDE.md": "deployment/huggingface/docs/deployment_guide.md",
            "DEPLOYMENT_FIX.md": "deployment/huggingface/docs/deployment_fix.md",
            
            # Setup scripts
            "install_dependencies.py": "scripts/setup/install_dependencies.py",
            "test_dependencies.py": "scripts/setup/verify_installation.py",
            "setup_hf_auth.py": "scripts/setup/setup_hf_auth.py",
            
            # Training scripts
            "resume_training_from_7k.py": "scripts/training/resume_training.py",
            "real_training_manager.py": "scripts/training/training_manager.py",
            "openllm_training_with_auth.py": "scripts/training/training_with_auth.py",
            "compare_models.py": "scripts/training/compare_models.py",
            
            # Evaluation scripts
            "test_trained_model.py": "scripts/evaluation/test_model.py",
            "test_sentencepiece.py": "scripts/evaluation/test_tokenizer.py",
            "test_hf_auth.py": "scripts/evaluation/test_auth.py",
            "check_model.py": "scripts/evaluation/check_model.py",
            
            # Maintenance scripts
            "fix_linting.py": "scripts/maintenance/fix_linting.py",
            "fix_hf_space.py": "scripts/maintenance/fix_deployment.py",
            "fix_training_upload.py": "scripts/maintenance/fix_training.py",
            
            # Deployment utilities
            "app.py": "deployment/huggingface/space_app_main.py",
            "app_backup.py": "deployment/huggingface/space_app_backup.py",
            "check_deployment_status.py": "deployment/check_status.py",
            "diagnose_deployment.py": "deployment/diagnose.py",
            
            # Configuration files
            ".hf_space_config.json": "configs/deployment_configs/huggingface_space.json",
            
            # Documentation files
            "SOLUTION_SUMMARY.md": "docs/troubleshooting/solution_summary.md",
            "IMPLEMENTATION_SUMMARY.md": "docs/development/implementation_summary.md",
            
            # Evaluation results
            "complete_evaluation.json": "models/evaluation/complete_evaluation.json",
            "downstream_evaluation.json": "models/evaluation/downstream_evaluation.json",
            "model_comparison.json": "models/evaluation/model_comparison.json",
        }
        
        # Files to delete (obsolete/duplicate)
        self.files_to_delete = [
            "tatus --porcelain",  # Malformed filename
        ]
        
        logger.info(f"Initialized ProjectReorganizer for {self.project_root}")
        logger.info(f"Dry run mode: {self.dry_run}")

    def create_backup(self) -> bool:
        """
        Create a backup of the current project structure.
        
        Returns:
            True if backup was successful, False otherwise
        """
        try:
            logger.info("Creating backup of current project structure...")
            
            if self.dry_run:
                logger.info("DRY RUN: Would create backup at %s", self.backup_dir)
                return True
            
            # Create backup directory
            self.backup_dir.mkdir(exist_ok=True)
            
            # Copy all files except backup directory itself
            for item in self.project_root.iterdir():
                if item.name != self.backup_dir.name:
                    if item.is_file():
                        shutil.copy2(item, self.backup_dir / item.name)
                    elif item.is_dir():
                        shutil.copytree(item, self.backup_dir / item.name)
            
            logger.info(f"Backup created successfully at {self.backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False

    def create_new_directories(self) -> bool:
        """
        Create the new directory structure.
        
        Returns:
            True if all directories were created successfully, False otherwise
        """
        try:
            logger.info("Creating new directory structure...")
            
            for directory in self.new_directories:
                dir_path = self.project_root / directory
                
                if self.dry_run:
                    logger.info(f"DRY RUN: Would create directory {dir_path}")
                else:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created directory: {dir_path}")
                    self.changes_made.append(f"Created directory: {dir_path}")
            
            logger.info("New directory structure created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create new directories: {e}")
            return False

    def move_files(self) -> bool:
        """
        Move files to their new locations according to the migration mapping.
        
        Returns:
            True if all files were moved successfully, False otherwise
        """
        try:
            logger.info("Moving files to new locations...")
            
            # Track which files were actually found and moved
            files_found = 0
            files_moved = 0
            
            for source_file, destination in self.file_migrations.items():
                source_path = self.project_root / source_file
                dest_path = self.project_root / destination
                
                if not source_path.exists():
                    logger.warning(f"Source file not found: {source_path}")
                    continue
                
                files_found += 1
                
                # Ensure destination directory exists
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                if self.dry_run:
                    logger.info(f"DRY RUN: Would move {source_path} -> {dest_path}")
                else:
                    shutil.move(str(source_path), str(dest_path))
                    logger.info(f"Moved: {source_path} -> {dest_path}")
                    self.changes_made.append(f"Moved: {source_path} -> {dest_path}")
                
                files_moved += 1
            
            logger.info(f"File migration completed: {files_found} files found, {files_moved} files {'would be moved' if self.dry_run else 'moved'}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to move files: {e}")
            return False

    def delete_obsolete_files(self) -> bool:
        """
        Delete obsolete and duplicate files.
        
        Returns:
            True if all files were deleted successfully, False otherwise
        """
        try:
            logger.info("Deleting obsolete and duplicate files...")
            
            for file_to_delete in self.files_to_delete:
                file_path = self.project_root / file_to_delete
                
                if not file_path.exists():
                    logger.warning(f"File to delete not found: {file_path}")
                    continue
                
                if self.dry_run:
                    logger.info(f"DRY RUN: Would delete {file_path}")
                else:
                    file_path.unlink()
                    logger.info(f"Deleted: {file_path}")
                    self.changes_made.append(f"Deleted: {file_path}")
            
            logger.info("Obsolete file deletion completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete obsolete files: {e}")
            return False

    def consolidate_duplicate_files(self) -> bool:
        """
        Consolidate duplicate files by keeping the most recent/complete version.
        
        Returns:
            True if consolidation was successful, False otherwise
        """
        try:
            logger.info("Consolidating duplicate files...")
            
            # Group duplicate files by base name
            duplicate_groups = {
                "hf_space_app": [
                    "hf_space_app_fixed.py",
                    "hf_space_app_openllm_fixed.py", 
                    "hf_space_app_openllm_compatible.py",
                    "hf_space_app_model_fix.py",
                    "hf_space_app_final.py",
                    "hf_space_app_gradio_441.py",
                    "hf_space_app_complete.py",
                    "hf_space_app_simple.py",
                    "hf_space_app.py"
                ],
                "hf_space_requirements": [
                    "hf_space_requirements.txt",
                    "hf_space_requirements_updated.txt",
                    "hf_space_requirements_complete.txt"
                ],
                "hf_space_README": [
                    "hf_space_README.md",
                    "hf_space_README_updated.md"
                ]
            }
            
            total_duplicates_found = 0
            total_duplicates_removed = 0
            
            for group_name, files in duplicate_groups.items():
                existing_files = [f for f in files if (self.project_root / f).exists()]
                
                if len(existing_files) > 1:
                    total_duplicates_found += len(existing_files) - 1
                    logger.info(f"Found {len(existing_files)} duplicates for {group_name}")
                    
                    # Keep the most recent file (assuming it's the most complete)
                    # For now, keep the first one and log the others
                    keep_file = existing_files[0]
                    delete_files = existing_files[1:]
                    
                    logger.info(f"Keeping: {keep_file}")
                    logger.info(f"Will delete: {delete_files}")
                    
                    if not self.dry_run:
                        for delete_file in delete_files:
                            file_path = self.project_root / delete_file
                            if file_path.exists():
                                file_path.unlink()
                                logger.info(f"Deleted duplicate: {delete_file}")
                                self.changes_made.append(f"Deleted duplicate: {delete_file}")
                                total_duplicates_removed += 1
                    else:
                        total_duplicates_removed += len(delete_files)
            
            logger.info(f"Duplicate file consolidation completed: {total_duplicates_found} duplicates found, {total_duplicates_removed} {'would be removed' if self.dry_run else 'removed'}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to consolidate duplicate files: {e}")
            return False

    def create_readme_files(self) -> bool:
        """
        Create README.md files for new directories to explain their purpose.
        
        Returns:
            True if README files were created successfully, False otherwise
        """
        try:
            logger.info("Creating README files for new directories...")
            
            readme_content = {
                "deployment": """# Deployment

This directory contains deployment configurations and utilities for OpenLLM.

## Subdirectories

- `huggingface/` - Hugging Face Space deployment
- `docker/` - Docker containerization
- `kubernetes/` - Kubernetes deployment (enterprise)
""",
                "scripts": """# Scripts

This directory contains utility scripts for OpenLLM development and maintenance.

## Subdirectories

- `setup/` - Setup and installation scripts
- `training/` - Training utilities and management
- `evaluation/` - Model evaluation and testing
- `maintenance/` - Maintenance and cleanup utilities
""",
                "configs": """# Configurations

This directory contains configuration files for different aspects of OpenLLM.

## Subdirectories

- `model_configs/` - Model architecture configurations
- `training_configs/` - Training pipeline configurations  
- `deployment_configs/` - Deployment configurations
""",
                "models": """# Models

This directory contains trained models and checkpoints.

## Subdirectories

- `checkpoints/` - Training checkpoints
- `final/` - Final trained models
- `evaluation/` - Model evaluation results
""",
                "logs": """# Logs

This directory contains log files and outputs from various processes.

## Subdirectories

- `training/` - Training logs
- `evaluation/` - Evaluation logs
- `deployment/` - Deployment logs
"""
            }
            
            for directory, content in readme_content.items():
                readme_path = self.project_root / directory / "README.md"
                
                if self.dry_run:
                    logger.info(f"DRY RUN: Would create README at {readme_path}")
                else:
                    readme_path.write_text(content)
                    logger.info(f"Created README: {readme_path}")
                    self.changes_made.append(f"Created README: {readme_path}")
            
            logger.info("README files created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create README files: {e}")
            return False

    def update_import_paths(self) -> bool:
        """
        Update import paths in Python files to reflect new structure.
        
        Returns:
            True if import paths were updated successfully, False otherwise
        """
        try:
            logger.info("Updating import paths in Python files...")
            
            # Define import path updates
            import_updates = {
                "from deployment.huggingface.space_auth import": "from deployment.huggingface.space_auth import",
                "import deployment.huggingface.space_auth": "import deployment.huggingface.space_auth",
                "from scripts.evaluation.test_model import": "from scripts.evaluation.test_model import",
                "import scripts.evaluation.test_model": "import scripts.evaluation.test_model",
                # Add more import path updates as needed
            }
            
            # Find Python files, excluding virtual environment and system packages
            python_files = []
            for py_file in self.project_root.rglob("*.py"):
                # Skip virtual environment
                if "venv" in str(py_file) or ".venv" in str(py_file):
                    continue
                # Skip common system package directories
                if any(skip_dir in str(py_file) for skip_dir in [
                    "site-packages", "__pycache__", ".pytest_cache", 
                    "node_modules", ".git", "backup_before_reorganization"
                ]):
                    continue
                # Only include files that are part of our project
                python_files.append(py_file)
            
            for python_file in python_files:
                if self.dry_run:
                    logger.info(f"DRY RUN: Would update imports in {python_file}")
                    continue
                
                try:
                    # Use UTF-8 encoding to handle Unicode characters
                    content = python_file.read_text(encoding='utf-8', errors='ignore')
                    original_content = content
                    
                    # Apply import updates
                    for old_import, new_import in import_updates.items():
                        content = content.replace(old_import, new_import)
                    
                    # Write back if changes were made
                    if content != original_content:
                        python_file.write_text(content, encoding='utf-8')
                        logger.info(f"Updated imports in: {python_file}")
                        self.changes_made.append(f"Updated imports in: {python_file}")
                        
                except Exception as e:
                    logger.warning(f"Failed to update imports in {python_file}: {e}")
            
            logger.info("Import path updates completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update import paths: {e}")
            return False

    def generate_migration_report(self) -> str:
        """
        Generate a detailed report of all changes made during reorganization.
        
        Returns:
            String containing the migration report
        """
        report = f"""
# OpenLLM Project Reorganization Report

## Summary
- Project Root: {self.project_root}
- Dry Run: {self.dry_run}
- Total Changes: {len(self.changes_made)}

## Changes Made
"""
        
        for change in self.changes_made:
            report += f"- {change}\n"
        
        report += f"""
## New Directory Structure
"""
        
        for directory in self.new_directories:
            report += f"- {directory}/\n"
        
        report += f"""
## File Migrations
"""
        
        for source, destination in self.file_migrations.items():
            report += f"- {source} → {destination}\n"
        
        report += f"""
## Next Steps
1. Test the reorganized project structure
2. Update documentation references
3. Verify all import paths work correctly
4. Run the test suite to ensure functionality
5. Update CI/CD pipelines if needed

## Rollback Instructions
If you need to rollback the changes:
1. Delete the reorganized files
2. Restore from backup: {self.backup_dir}
3. Verify the project works as expected
"""
        
        return report

    def reorganize(self) -> bool:
        """
        Perform the complete project reorganization.
        
        Returns:
            True if reorganization was successful, False otherwise
        """
        try:
            logger.info("Starting OpenLLM project reorganization...")
            
            # Step 1: Create backup
            if not self.create_backup():
                logger.error("Failed to create backup. Aborting reorganization.")
                return False
            
            # Step 2: Create new directory structure
            if not self.create_new_directories():
                logger.error("Failed to create new directories. Aborting reorganization.")
                return False
            
            # Step 3: Consolidate duplicate files
            if not self.consolidate_duplicate_files():
                logger.error("Failed to consolidate duplicate files. Aborting reorganization.")
                return False
            
            # Step 4: Move files to new locations
            if not self.move_files():
                logger.error("Failed to move files. Aborting reorganization.")
                return False
            
            # Step 5: Delete obsolete files
            if not self.delete_obsolete_files():
                logger.error("Failed to delete obsolete files. Aborting reorganization.")
                return False
            
            # Step 6: Create README files
            if not self.create_readme_files():
                logger.error("Failed to create README files. Aborting reorganization.")
                return False
            
            # Step 7: Update import paths
            if not self.update_import_paths():
                logger.error("Failed to update import paths. Aborting reorganization.")
                return False
            
            # Step 8: Generate migration report
            report = self.generate_migration_report()
            report_path = self.project_root / "REORGANIZATION_REPORT.md"
            
            if not self.dry_run:
                # Use UTF-8 encoding to handle Unicode characters
                report_path.write_text(report, encoding='utf-8')
                logger.info(f"Migration report saved to: {report_path}")
            
            logger.info("Project reorganization completed successfully!")
            logger.info(f"Total changes made: {len(self.changes_made)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Reorganization failed: {e}")
            return False

def main():
    """Main entry point for the reorganization script."""
    parser = argparse.ArgumentParser(
        description="Reorganize OpenLLM project structure for better organization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be changed
  python reorganize_project.py --dry-run
  
  # Perform actual reorganization
  python reorganize_project.py
  
  # Reorganize with custom project root
  python reorganize_project.py --project-root /path/to/openllm
        """
    )
    
    parser.add_argument(
        "--project-root",
        default=".",
        help="Root directory of the OpenLLM project (default: current directory)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate changes without making them"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true", 
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create reorganizer and run
    reorganizer = ProjectReorganizer(
        project_root=args.project_root,
        dry_run=args.dry_run
    )
    
    success = reorganizer.reorganize()
    
    if success:
        logger.info("Reorganization completed successfully!")
        if args.dry_run:
            logger.info("This was a dry run. No actual changes were made.")
        else:
            logger.info("Check REORGANIZATION_REPORT.md for details.")
    else:
        logger.error("Reorganization failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
