#!/usr/bin/env python3
"""
queryome_cli.py
---------------

Command line interface for the Queryome medical research system.

Usage:
    python queryome_cli.py
    
Interactive mode:
    python queryome_cli.py --interactive
    
Direct query:
    python queryome_cli.py --query "What are the latest treatments for Type 2 diabetes?"
"""

import os
import sys
import argparse
from datetime import datetime
from agent import PIAgent, FileLogger, initialize_search_engine


def main():
    parser = argparse.ArgumentParser(
        description="Queryome - Multi-agent medical literature research system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  queryome_cli.py
  queryome_cli.py --interactive
  queryome_cli.py --query "What are the latest treatments for Type 2 diabetes?"
  queryome_cli.py --query "Efficacy of metformin in elderly patients" --log-dir ./custom_logs
        """
    )
    
    parser.add_argument(
        "--query", "-q",
        help="Research query to execute"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start interactive mode for multiple queries"
    )
    
    parser.add_argument(
        "--log-dir",
        help="Custom directory for logs (default: logs/timestamp)"
    )
    
    parser.add_argument(
        "--no-search-engine",
        action="store_true",
        help="Skip search engine initialization (for testing)"
    )
    
    args = parser.parse_args()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    # Initialize search engine unless skipped
    if not args.no_search_engine:
        if not initialize_search_engine(device="cuda:0"):
            print("ERROR: Failed to initialize search engine")
            sys.exit(1)
    else:
        print("[INIT] Skipping search engine initialization")
    
    # Set up logging
    if args.log_dir:
        log_dir = args.log_dir
    else:
        log_dir = os.path.join("logs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    
    logger = FileLogger(log_dir)
    logger.log(f"Queryome CLI started - Log directory: {os.path.abspath(log_dir)}")
    
    # Create PI Agent
    pi_agent = PIAgent(logger=logger)
    
    try:
        if args.query:
            # Direct query mode
            print(f"[QUERYOME] Processing query: {args.query}")
            logger.log(f"[CLI] Direct query mode: {args.query}")
            result = pi_agent.research(args.query)
            
            print("\n" + "="*80)
            print("RESEARCH RESULTS")
            print("="*80)
            print(result)
            
        elif args.interactive:
            # Interactive mode
            print("="*80)
            print("QUERYOME - Interactive Medical Research System")
            print("="*80)
            print("Enter your research questions. Type 'quit' or 'exit' to stop.")
            print("Type 'help' for usage information.")
            print()
            
            session_count = 0
            while True:
                try:
                    query = input("Research query> ").strip()
                    
                    if not query:
                        continue
                    
                    if query.lower() in ['quit', 'exit', 'q']:
                        print("Goodbye!")
                        break
                    
                    if query.lower() == 'help':
                        print("""
Available commands:
  help          - Show this help message
  quit, exit, q - Exit the program
  
Simply type your research question and press Enter to start research.

Examples:
  What are the side effects of metformin?
  Latest research on COVID-19 vaccines
  Efficacy of statins in cardiovascular disease prevention
                        """)
                        continue
                    
                    session_count += 1
                    print(f"\n[SESSION {session_count}] Processing: {query}")
                    logger.log(f"[CLI] Interactive session {session_count}: {query}")
                    
                    result = pi_agent.research(query)
                    
                    print("\n" + "="*60)
                    print(f"RESEARCH RESULTS - SESSION {session_count}")
                    print("="*60)
                    print(result)
                    print("="*60 + "\n")
                    
                except KeyboardInterrupt:
                    print("\n\nResearch interrupted. Type 'quit' to exit or continue with a new query.")
                    continue
                except EOFError:
                    print("\nGoodbye!")
                    break
        
        else:
            # Default: single query input
            query = input("Enter your research question: ").strip()
            if not query:
                print("No query provided. Exiting.")
                sys.exit(0)
            
            print(f"[QUERYOME] Processing query: {query}")
            logger.log(f"[CLI] Single query mode: {query}")
            result = pi_agent.research(query)
            
            print("\n" + "="*80)
            print("RESEARCH RESULTS") 
            print("="*80)
            print(result)
    
    except KeyboardInterrupt:
        logger.log("[CLI] Interrupted by user")
        print("\nResearch interrupted by user")
    except Exception as e:
        logger.log(f"[CLI] Error: {e}")
        print(f"ERROR: {e}")
    finally:
        logger.log("[CLI] Session ended")
        print(f"\nLogs saved to: {os.path.abspath(log_dir)}")


if __name__ == "__main__":
    main()