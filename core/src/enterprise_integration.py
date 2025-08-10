#!/usr/bin/env python3
# Copyright (C) 2024 Louis Chua Bean Chong
#
# This file is part of OpenLLM.
#
# OpenLLM is dual-licensed:
# 1. For open source use: GNU General Public License v3.0
# 2. For commercial use: Commercial License (contact for details)
#
# See LICENSE and docs/LICENSES.md for full license information.

"""
Enterprise Integration Layer for OpenLLM

This module provides an optional plugin mechanism to load enterprise-only
modules without coupling the open source core to proprietary code. It follows
the project rule: core functionality must work without proprietary
dependencies; enterprise features are optional extensions.

How it works:
- Attempts to locate a Python module that exposes enterprise commands
- Supports two discovery methods:
  1) Python package on sys.path: `openllm_enterprise`
  2) Filesystem path via env var `OPENLLM_ENTERPRISE_PATH` that contains
     a module with `register_cli(subparsers)` function
- If found, calls `register_cli(subparsers)` to register additional CLI commands

Security and Licensing:
- No proprietary code is included in the open repository
- This module only performs optional dynamic imports if the user provides
  an enterprise package or path
- All core code remains GPLv3 compliant

Usage (enterprise side expected contract):
    # In the enterprise package/module
    def register_cli(subparsers):
        parser = subparsers.add_parser(
            "enterprise-train",
            help="Enterprise: RLHF training",
            description="Run RLHF training using enterprise-only components."
        )
        parser.add_argument("--config", required=True)
        parser.set_defaults(func=enterprise_train_entry)

    def enterprise_train_entry(args):
        ...

Author: Louis Chua Bean Chong
License: GPLv3 (core); enterprise modules remain out-of-tree
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any


def _try_import_by_name(module_name: str):
    """Attempt to import a module by name. Returns module or None on failure."""
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _try_import_from_path(module_path: str):
    """
    Attempt to import a module from a filesystem path.

    The path may point either to a package directory (containing __init__.py)
    or to a .py file. This function prepends the parent directory to sys.path
    and imports the module by stem name.
    """
    try:
        path = Path(module_path)
        if not path.exists():
            return None

        if path.is_file():
            parent = str(path.parent)
            mod_name = path.stem
        else:
            parent = str(path.parent)
            mod_name = path.name

        if parent not in sys.path:
            sys.path.insert(0, parent)
        return importlib.import_module(mod_name)
    except Exception:
        return None


def load_enterprise_cli(subparsers: Any) -> bool:
    """
    Try to load enterprise-only CLI commands.

    Discovery order:
    1) Python package/module named `openllm_enterprise`
    2) Env var `OPENLLM_ENTERPRISE_PATH` pointing to a package dir or .py file

    If a discovered module exposes `register_cli(subparsers)`, it will be called
    to register enterprise commands. Returns True if any enterprise module was
    loaded successfully; otherwise False.
    """
    # 1) Try well-known package name
    enterprise_mod = _try_import_by_name("openllm_enterprise")
    if enterprise_mod and hasattr(enterprise_mod, "register_cli"):
        try:
            enterprise_mod.register_cli(subparsers)
            print("🔌 Loaded enterprise commands from openllm_enterprise package")
            return True
        except Exception:
            # Fail gracefully; core must continue to work
            pass

    # 2) Try explicit path via environment variable
    enterprise_path = os.environ.get("OPENLLM_ENTERPRISE_PATH")
    if enterprise_path:
        enterprise_mod = _try_import_from_path(enterprise_path)
        if enterprise_mod and hasattr(enterprise_mod, "register_cli"):
            try:
                enterprise_mod.register_cli(subparsers)
                print(
                    "🔌 Loaded enterprise commands from OPENLLM_ENTERPRISE_PATH="
                    f"{enterprise_path}"
                )
                return True
            except Exception:
                # Fail gracefully
                pass

    # Not found (by design this is optional)
    return False


__all__ = ["load_enterprise_cli"]

