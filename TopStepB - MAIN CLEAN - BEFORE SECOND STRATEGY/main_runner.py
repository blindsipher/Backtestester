#!/usr/bin/env python3
"""
Main Runner - Trading Strategy Pipeline Orchestrator
===================================================

Ultra-thin entry point that orchestrates the complete pipeline.
Pure facilitation - collects config and coordinates existing modules.

Database management is handled by the optimization module through its
existing StorageConfig system in optimization/config/optuna_config.py.
"""

import logging
import sys

from app.core.config_collector import collect_cli_config, collect_interactive_config, is_cli_mode
from app.pipeline import orchestrate_pipeline

logger = logging.getLogger(__name__)

def main():
    """Main entry point - detect mode and orchestrate pipeline"""

    try:
        # Collect configuration based on mode
        if is_cli_mode():
            state = collect_cli_config()
            if state is None:
                return 1  # CLI parsing failed or --help
        else:
            state = collect_interactive_config()

        # Orchestrate the pipeline - optimization module handles PostgreSQL internally
        result = orchestrate_pipeline(state)

        # Return appropriate exit code
        return 0 if result['success'] else 1
    except (ImportError, ModuleNotFoundError, KeyError, ValueError) as e:
        logger.error("Main runner failed: %s", e, exc_info=True)
        return 1
    except Exception as e:
        logger.critical("Unexpected error in main runner: %s", e, exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
