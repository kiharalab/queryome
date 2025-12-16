#!/usr/bin/env python3
"""
agent package
-------------

Multi-agent research system for medical literature search and analysis.

Main components:
- PIAgent: Primary Intelligence agent that plans and executes research
- SubAgentTeam: Consists of planner and critic agents working together  
- SynthesizerAgent: Creates final reports with proper citations
- FileLogger: Utility for logging research progress
"""

from .pi_agent import PIAgent
from .subagent_team import SubAgentTeam
from .synthesizer import SynthesizerAgent
from .utils import FileLogger, initialize_search_engine

__all__ = [
    'PIAgent',
    'SubAgentTeam', 
    'SynthesizerAgent',
    'FileLogger',
    'initialize_search_engine'
]