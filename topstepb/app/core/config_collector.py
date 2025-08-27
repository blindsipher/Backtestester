"""
Configuration Collection
========================

Handles all user input collection for pipeline configuration.
Clean separation between CLI and interactive modes.
"""

import argparse
import sys
import os
from typing import Optional, List

from .state import PipelineState


def collect_cli_config() -> Optional[PipelineState]:
    """
    Collect configuration from command line arguments
    
    Returns:
        PipelineState if all required args provided, None otherwise
    """
    parser = argparse.ArgumentParser(description='Trading Strategy Pipeline')
    
    # Required arguments
    parser.add_argument('--strategy', required=True, help='Strategy name')
    parser.add_argument('--symbol', required=True, help='Trading symbol (ES, NQ, etc.)')
    parser.add_argument('--timeframe', required=True, help='Timeframe (5m, 1h, etc.)')
    parser.add_argument('--account-type', required=True, help='Account type (topstep_50k, etc.)')
    parser.add_argument('--slippage', type=float, required=True, help='Slippage in ticks')
    parser.add_argument('--commission', type=float, required=True, help='Commission per trade')
    parser.add_argument('--contracts-per-trade', type=int, required=True, help='Number of contracts to trade per signal')
    parser.add_argument('--split-type', required=True, choices=['chronological', 'walk_forward'], help='Split method')
    
    # Optional arguments
    parser.add_argument('--data-file', help='Path to data file (optional)')
    parser.add_argument('--synthetic-bars', type=int, default=5000, help='Number of synthetic bars to generate (default: 5000)')
    
    # Optimization arguments
    parser.add_argument('--optimization-enabled', action='store_true', default=True, help='Enable parameter optimization (default: True)')
    parser.add_argument('--no-optimization', action='store_true', help='Disable parameter optimization')
    parser.add_argument('--max-trials', type=int, default=100, help='Maximum optimization trials (default: 100)')
    # Get default for help text
    default_workers = _get_default_workers()
    parser.add_argument('--max-workers', type=int, default=default_workers, help=f'Maximum parallel workers (default: {default_workers})')
    parser.add_argument('--memory-per-worker-mb', type=int, default=1500, help='Memory limit per worker in MB (default: 1500)')
    parser.add_argument('--timeout-per-trial', type=int, default=60, help='Maximum seconds per trial (default: 60)')
    parser.add_argument('--results-top-n', type=int, default=10, help='Number of top results to return (default: 10)')
    
    # Parse arguments
    try:
        args = parser.parse_args()
        
        
        # Handle optimization enabled/disabled logic
        optimization_enabled = args.optimization_enabled and not args.no_optimization
        
        # Validate max_workers
        validated_workers = _validate_workers(args.max_workers)
        
        # Normalize account type format: convert hyphens to underscores
        # CLI accepts both "topstep-50k" and "topstep_50k" but system expects "topstep_50k"
        normalized_account_type = args.account_type.replace('-', '_')
        
        # Normalize split type format: convert hyphens to underscores  
        # CLI accepts both "walk-forward" and "walk_forward" but system expects "walk_forward"
        normalized_split_type = args.split_type.replace('-', '_')
        
        return PipelineState(
            strategy_name=args.strategy,
            symbol=args.symbol,
            timeframe=args.timeframe,
            account_type=normalized_account_type,
            slippage_ticks=args.slippage,
            commission_per_trade=args.commission,
            contracts_per_trade=args.contracts_per_trade,
            split_type=normalized_split_type,
            data_file_path=args.data_file,
            synthetic_bars=args.synthetic_bars,
            optimization_enabled=optimization_enabled,
            max_trials=args.max_trials,
            max_workers=validated_workers,
            memory_per_worker_mb=args.memory_per_worker_mb,
            timeout_per_trial=args.timeout_per_trial,
            results_top_n=args.results_top_n
        )
        
    except SystemExit:
        # argparse calls sys.exit on --help or invalid args
        return None


def collect_interactive_config() -> PipelineState:
    """
    Collect configuration through interactive prompts
    
    Returns:
        PipelineState with user-provided configuration
    """
    print("\nInteractive Pipeline Configuration")
    print("=" * 40)
    
    # Strategy selection using discovery
    print("\n1. Strategy Selection:")
    strategy_name = _select_strategy_interactive()
    
    # Market configuration using existing config module
    print("\n2. Market Configuration:")
    symbol = _select_symbol_interactive()
    timeframe = _select_timeframe_interactive()
    account_type = _select_account_type_interactive()
    
    # Execution parameters
    print("\n3. Execution Parameters:")
    slippage_input = input("Enter slippage in ticks (default: 0.5): ").strip()
    slippage_ticks = float(slippage_input) if slippage_input else 0.5
    
    commission_input = input("Enter commission per trade (default: 2.50): ").strip()
    commission_per_trade = float(commission_input) if commission_input else 2.50
    
    contracts_input = input("Enter number of contracts per trade (default: 1): ").strip()
    contracts_per_trade = int(contracts_input) if contracts_input else 1
    
    # Data file (optional)
    print("\n4. Data Configuration:")
    data_file_path = None
    synthetic_bars = 5000
    
    use_file = input("Use data file? (y/N): ").strip().lower()
    if use_file in ['y', 'yes']:
        # Try to use tkinter file dialog
        try:
            import tkinter as tk
            from tkinter import filedialog
            print("Opening file dialog...")
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)  # Bring to front on Windows
            data_file_path = filedialog.askopenfilename(
                title="Select Data File",
                filetypes=[
                    ("Trading Data", ("*.csv", "*.parquet", "*.CSV", "*.PARQUET")),
                    ("CSV files", ("*.csv", "*.CSV")),
                    ("Parquet files", ("*.parquet", "*.PARQUET")),
                    ("All files", "*.*")
                ]
            )
            root.destroy()
            if not data_file_path:
                print("No file selected, will use synthetic data")
            else:
                print(f"Selected file: {data_file_path}")
        except (ImportError, Exception) as e:
            print(f"File dialog failed ({e}), enter file path manually:")
            data_file_path = input("Enter data file path (or press Enter for synthetic): ").strip()
            if not data_file_path:
                data_file_path = None
    
    # Ask for synthetic data amount if not using file
    if not data_file_path:
        bars_input = input("Enter number of synthetic bars to generate (default: 5000): ").strip()
        synthetic_bars = int(bars_input) if bars_input else 5000
    
    # Split configuration
    print("\n5. Data Split Configuration:")
    print("Split types: (1) chronological, (2) walk_forward")
    split_choice = input("Select split type (1 or 2): ").strip()
    split_type = "chronological" if split_choice == "1" else "walk_forward"
    
    # Optimization configuration
    print("\n6. Optimization Configuration:")
    opt_enabled = input("Enable parameter optimization? (Y/n): ").strip().lower()
    optimization_enabled = opt_enabled not in ['n', 'no']
    
    max_trials = 100
    # Use existing optimization resource management instead of utils resource_manager
    import os
    max_workers = min(4, os.cpu_count() or 1)  # Conservative default that works with existing optimization
    memory_per_worker_mb = 1500  # Default memory per worker
    timeout_per_trial = 60
    results_top_n = 10
    if optimization_enabled:
        trials_input = input("Maximum optimization trials (default: 100): ").strip()
        max_trials = int(trials_input) if trials_input else 100
        
        cpu_count = os.cpu_count() or 1
        default_workers = _get_default_workers()
        workers_input = input(f"Maximum parallel workers (default: {default_workers}, system CPU cores: {cpu_count}): ").strip()
        raw_workers = int(workers_input) if workers_input else default_workers
        max_workers = _validate_workers(raw_workers)
        
        memory_input = input("Memory per worker in MB (default: 400): ").strip()
        memory_per_worker_mb = int(memory_input) if memory_input else 400
        
        timeout_input = input("Maximum seconds per trial (default: 60): ").strip()
        timeout_per_trial = int(timeout_input) if timeout_input else 60
        
        results_input = input("Number of top results to return (default: 10): ").strip()
        results_top_n = int(results_input) if results_input else 10
    
    
    return PipelineState(
        strategy_name=strategy_name,
        symbol=symbol,
        timeframe=timeframe,
        account_type=account_type,
        slippage_ticks=slippage_ticks,
        commission_per_trade=commission_per_trade,
        contracts_per_trade=contracts_per_trade,
        split_type=split_type,
        data_file_path=data_file_path,
        synthetic_bars=synthetic_bars,
        optimization_enabled=optimization_enabled,
        max_trials=max_trials,
        max_workers=max_workers,
        memory_per_worker_mb=memory_per_worker_mb,
        timeout_per_trial=timeout_per_trial,
        results_top_n=results_top_n
    )


def _select_from_list_interactive(prompt: str, options: List[str]) -> str:
    """
    Generic helper to present numbered options and return selection
    
    Args:
        prompt: The prompt to show user
        options: List of options to choose from
        
    Returns:
        Selected option string
    """
    for i, option in enumerate(options, 1):
        print(f"  [{i}] {option}")
    
    while True:
        choice = input(f"\n{prompt} (1-{len(options)}): ").strip()
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(options):
                selected_option = options[choice_num - 1]
                print(f"Selected: {selected_option}")
                return selected_option
            else:
                print(f"Please enter a number between 1 and {len(options)}")
        except ValueError:
            print("Please enter a valid number")


def _select_strategy_interactive() -> str:
    """
    Interactive strategy selection using discovery
    
    Returns:
        Selected strategy name
    """
    try:
        # Import here to avoid circular imports
        from strategies import discover_strategies
        
        available_strategies = discover_strategies()
        strategy_names = list(available_strategies.keys())
        
        print("Available strategies:")
        return _select_from_list_interactive("Select strategy", strategy_names)
                
    except Exception as e:
        print(f"Error discovering strategies: {e}")
        print("Falling back to manual entry...")
        return input("Enter strategy name manually: ").strip()


def _select_symbol_interactive() -> str:
    """
    Interactive symbol selection using config module
    
    Returns:
        Selected symbol
    """
    try:
        # Import here to avoid circular imports
        from config.system_config import TopStepMarkets
        
        markets = TopStepMarkets()
        all_markets = markets.get_all_markets()
        available_symbols = list(all_markets.keys())
        
        print("Available symbols:")
        return _select_from_list_interactive("Select symbol", available_symbols)
        
    except Exception as e:
        print(f"Error loading symbols: {e}")
        return input("Enter symbol manually (ES, NQ, etc.): ").strip().upper()


def _select_timeframe_interactive() -> str:
    """
    Interactive timeframe selection using config module
    
    Returns:
        Selected timeframe
    """
    try:
        # Import here to avoid circular imports
        from config.system_config import SupportedTimeframes
        
        available_timeframes = SupportedTimeframes.SUPPORTED
        
        print("Available timeframes:")
        return _select_from_list_interactive("Select timeframe", available_timeframes)
        
    except Exception as e:
        print(f"Error loading timeframes: {e}")
        return input("Enter timeframe manually (5m, 1h, etc.): ").strip()


def _select_account_type_interactive() -> str:
    """
    Interactive account type selection
    
    Returns:
        Selected account type
    """
    # For now, use the known account types - could be made dynamic later
    available_accounts = ["topstep_50k", "topstep_100k", "topstep_150k"]
    
    print("Available account types:")
    return _select_from_list_interactive("Select account type", available_accounts)


def _get_default_workers() -> int:
    """
    Get intelligent default for max_workers based on environment variables and system specs.
    
    Returns:
        Validated default worker count
    """
    # First check environment variable (same as optuna config)
    if 'OPTUNA_MAX_WORKERS' in os.environ:
        try:
            env_workers = int(os.environ['OPTUNA_MAX_WORKERS'])
            return _validate_workers(env_workers)
        except (ValueError, TypeError):
            pass  # Fall through to computed default
    
    # Use optimized default based on CPU count (3x bandwidth improvement)
    cpu_count = os.cpu_count() or 1
    # Optimized: use 75% of available cores, minimum 1, maximum CPU_COUNT (removed -1 limitation)
    default = max(1, min(cpu_count, int(cpu_count * 0.75)))
    return default if default > 0 else 2  # Final fallback


def _validate_workers(workers: int) -> int:
    """
    Validate and potentially adjust worker count to reasonable bounds.
    
    Args:
        workers: Requested worker count
        
    Returns:
        Validated worker count within reasonable bounds
        
    Raises:
        ValueError: If workers is invalid and cannot be corrected
    """
    if workers < 1:
        raise ValueError(f"Worker count must be positive, got {workers}")
    
    cpu_count = os.cpu_count() or 1
    
    # Warn about potentially problematic configurations
    if workers > cpu_count * 2:
        print(f"Warning: {workers} workers exceeds recommended maximum ({cpu_count * 2}) for this system")
        print(f"   Consider using {min(workers, cpu_count * 2)} workers for better performance")
    
    # Hard limit: don't allow more than CPU_COUNT-1 workers to prevent system lockup
    max_workers = max(1, cpu_count - 1)
    if workers > max_workers:
        print(f"Warning: Limiting workers from {workers} to {max_workers} (CPU_COUNT-1) for system stability")
        workers = max_workers
    
    return workers


def is_cli_mode() -> bool:
    """
    Check if running in CLI mode (has command line arguments)
    
    Returns:
        True if CLI arguments provided, False for interactive mode
    """
    return len(sys.argv) > 1