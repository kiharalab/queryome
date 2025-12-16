#!/usr/bin/env python3
"""
pi_agent.py
-----------

Primary Intelligence agent that plans and executes research.
"""

import json
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from .utils import FileLogger, hybrid_search, bm25_author_keywords_search, bm25_mesh_terms_search
from .subagent_team import SubAgentTeam
from .synthesizer import SynthesizerAgent


class PIAgent:
    """PI agent that plans and executes research."""
    
    def __init__(self, logger: FileLogger, model: str = "o3"):
        self.client = OpenAI()
        self.logger = logger
        self.model = model
        self.model_defaults = {
            "model": self.model,
            "reasoning": {"effort": "high", "summary": "detailed"},
        }
        
        self.pi_tools = [
            {"type": "function", "name": "run_subagent_team", "description": "Run multiple subagent teams in parallel to research different aspects of the question.", "parameters": {"type": "object", "properties": {"subquestions": {"type": "array", "items": {"type": "string"}, "description": "List of sub-questions to explore in parallel"}}, "required": ["subquestions"]}},
            {"type": "function", "name": "hybrid_search", "description": "Perform hybrid search combining FAISS vector search with BM25 title+abstract search.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query"}, "k": {"type": "integer", "description": "Number of results to return", "default": 10}, "alpha": {"type": "number", "description": "Weight for semantic search (0.0 = only BM25, 1.0 = only vector)", "default": 0.5}}, "required": ["query"]}},
            {"type": "function", "name": "bm25_author_keywords_search", "description": "Search specifically in author-provided keywords.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query"}, "k": {"type": "integer", "description": "Number of results", "default": 10}}, "required": ["query"]}},
            {"type": "function", "name": "bm25_mesh_terms_search", "description": "Search specifically in MeSH terms.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query"}, "k": {"type": "integer", "description": "Number of results", "default": 10}}, "required": ["query"]}},
            {"type": "function", "name": "synthesize_final_report", "description": "Call the synthesizer agent to create a final research report.", "parameters": {"type": "object", "properties": {"research_guide": {"type": "string", "description": "Guide for the synthesizer on what to write based on user's question"}}, "required": ["research_guide"]}},
        ]
        
        self.tool_dispatch = {
            "run_subagent_team": self._run_subagent_team,
            "hybrid_search": hybrid_search,
            "bm25_author_keywords_search": bm25_author_keywords_search,
            "bm25_mesh_terms_search": bm25_mesh_terms_search,
            "synthesize_final_report": self._synthesize_final_report
        }
        
        self.all_relevant_information = []
        self.user_query = ""
    
    def research(self, query: str) -> str:
        """Main research method that coordinates the entire research process."""
        self.logger.log(f"[PI] Starting research for query: '{query}'")
        self.user_query = query
        self.all_relevant_information = []
        
        pi_system_prompt = """
<PIAgentInstructions>
  <Identity>You are the Principal Investigator (PI) agent in Queryome, responsible for coordinating evidence-based biomedical literature research.</Identity>
  <ResearchWorkflow>
    <Step index="1">Analyze the user's query to judge complexity, scope, and required depth.</Step>
    <Step index="2">If the question is simple, perform targeted searches yourself and prepare a direct response.</Step>
    <Step index="3">If the question is complex or multi-faceted, break it into structured subquestions and use <ToolRef>run_subagent_team</ToolRef> to gather evidence in parallel.</Step>
    <Step index="4">Invoke the search tools whenever additional coverage or confirmation is needed; do not rely on memory.</Step>
    <Step index="5">After collecting sufficient evidence, craft a comprehensive research guide and call <ToolRef>synthesize_final_report</ToolRef>.</Step>
  </ResearchWorkflow>
  <Toolbox>
    <Tool name="hybrid_search">Combine FAISS vector search with BM25 title+abstract search; tune k and alpha (0.0 = only BM25, 1.0 = only vector) to balance semantic versus keyword emphasis.</Tool>
    <Tool name="bm25_author_keywords_search">Search author-provided keywords for terminology-focused hits.</Tool>
    <Tool name="bm25_mesh_terms_search">Search MeSH terms for ontology-aligned evidence.</Tool>
    <Tool name="run_subagent_team">Dispatch multiple subagents, each handling a subquestion; ideal for broad or layered queries.</Tool>
    <Tool name="synthesize_final_report">Send your finalized research guide to the synthesizer agent to generate the user-facing report.</Tool>
  </Toolbox>
  <SynthesizeFinalReportGuidelines>
    <CallTiming>Invoke only after assembling every article, insight, and citation the final report should reference.</CallTiming>
    <ResearchGuideRequirements>
      <Thesis length="1-2 sentences">Articulate the core message and framing the synthesizer must deliver.</Thesis>
      <Structure>Provide a section-by-section outline aligned with the user's requested format, detailing the arguments, comparisons, and evidence for each section.</Structure>
      <PriorityFindings>List 3-6 non-negotiable insights or takeaways and explain how they advance the user's goals.</PriorityFindings>
      <WritingDirectives>Specify tone (academic), target length overall or per section, formatting requests (e.g., boxed summary, bullet lists), and reminders about clarity, compartment-aware framing, or sensor-versus-effector themes.</WritingDirectives>
      <CitationPlan>Enumerate every reference to cite (PMID plus short label), highlight citations mandated by the user, and map high-priority articles to the sections they support. If run_subagent_team was not used, include the articles you located personally.</CitationPlan>
      <Caveats>Call out evidence gaps, conflicts, limitations, or quality issues the synthesizer must acknowledge.</Caveats>
    </ResearchGuideRequirements>
    <CitationReminder>Explicitly remind the synthesizer to use only the supplied references and to format citations as [Author, Year, PMID]. For multiple citations: [Author, Year, PMID; Author2, Year2, PMID2].</CitationReminder>
  </SynthesizeFinalReportGuidelines>
  <GeneralRules>
    <Rule>If you do not call synthesize_final_report, your direct answer must still include in-text citations formatted as [Author, Year, PMID].</Rule>
    <Rule>Think step-by-step, remain evidence-based, and never hallucinate.</Rule>
  </GeneralRules>
  <ExecutionReminder>Begin by analyzing the query, choosing a research strategy, and selecting the tools you need.</ExecutionReminder>
</PIAgentInstructions>
"""

#         pi_system_prompt = """
# You are the PI (Principal Investigator) agent in Queryome. 
# Your sole purpose is to answer biomedical multiple-choice questions accurately using evidence from the literature.

# Your process:
# 1. Read the biomedical multiple-choice question carefully.
# 2. Break it into subquestions, one for each answer option. Each subquestion should ask whether that specific option is supported or refuted by evidence.
# 3. Use run_subagent_team to assign these subquestions in parallel so subagents gather evidence for each option.
# 4. Collect the evidence and call synthesize_final_report. The research guide must:
#    - Summarize the evidence for and against each option.
#    - Clearly explain which option is best supported by the evidence.
# 5. After synthesis, output ONLY the correct option letter (A, B, C, or D). Do not include reasoning, explanations, or extra text in the final output.

# Available tools:
# - hybrid_search: Use for broad and comprehensive searches. Adjust alpha (0.0 = only BM25, 1.0 = only vector) depending on whether semantic similarity or keyword precision is needed.
# - bm25_author_keywords_search: Use for keyword-driven evidence.
# - bm25_mesh_terms_search: Use for medical terminology-driven evidence.
# - run_subagent_team: Always use this for multiple-choice questions (one subquestion per option).
# - synthesize_final_report: Always use this before producing the final option.

# Strict rules:
# - Every option must be evaluated with evidence.
# - The final output must be ONLY a single letter (A, B, C, or D).
# - Never hallucinate. Be fully evidence-based.
# """
        pi_user_prompt = f"User's Research Query: {query}"

        self.logger.log_text("pi_initial_prompt.txt", f"SYSTEM: {pi_system_prompt}\nUSER: {pi_user_prompt}")
        
        try:
            self.logger.log(f"[PI] Calling {self.model} for initial analysis...")
            pi_model_config = dict(self.model_defaults)
            pi_model_config["tools"] = self.pi_tools
            
            response = self.client.responses.create(
                input=[
                    {"role": "system", "content": pi_system_prompt},
                    {"role": "user", "content": pi_user_prompt}
                ],
                **pi_model_config
            )
            
            iteration = 0
            while True:
                iteration += 1
                self.logger.log(f"[PI] Reasoning iteration {iteration}")
                function_responses = self._invoke_functions_from_response(response, iteration)
                
                if not function_responses:
                    self.logger.log(f"[PI] No more function calls - research complete")
                    final_answer = response.output_text
                    self.logger.log_text("pi_final_answer_raw.txt", final_answer)

                    # Post-process citations if we have collected articles
                    if self.all_relevant_information:
                        from .citation_processor import process_citations
                        final_answer = process_citations(final_answer, self.all_relevant_information)
                        self.logger.log(f"[PI] Citation processing complete")

                    self.logger.log_text("pi_final_answer.txt", final_answer)
                    return final_answer
                else:
                    self.logger.log(f"[PI] Continuing reasoning with {len(function_responses)} function responses...")
                    response = self.client.responses.create(
                        input=function_responses, 
                        previous_response_id=response.id, 
                        **pi_model_config
                    )
        except Exception as e:
            self.logger.log(f"ERROR in PI Agent research: {str(e)}")
            return f"ERROR in PI Agent research: {str(e)}"
    
    def _run_subagent_team(self, subquestions: List[str]) -> str:
        """Run subagent teams in parallel for multiple subquestions."""
        self.logger.log(f"[PI] Running subagent teams for {len(subquestions)} subquestions")
        
        def research_single_subquestion(args):
            i, subquestion = args
            sub_logger = self.logger.get_sub_logger(f"subq_{i}_{subquestion}")
            sub_logger.log(f"[PI] Starting subagent team for: '{subquestion}'")
            team = SubAgentTeam(logger=sub_logger)
            result = team.research_subquestion(subquestion)
            sub_logger.log(f"[PI] Subagent team completed for: '{subquestion}' - found {len(result['relevant_entries'])} articles")
            return result
        
        max_workers = min(len(subquestions), 4)
        self.logger.log(f"[PI] Using {max_workers} parallel workers")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(research_single_subquestion, enumerate(subquestions)))
        
        total_new_articles = 0
        for result in results:
            total_new_articles += len(result["relevant_entries"])
            self.all_relevant_information.extend(result["relevant_entries"])
        
        self.logger.log(f"[PI] Collected {total_new_articles} total articles from all subagent teams")
        
        executive_summaries = [
            {
                "subquestion": r["subquestion"],
                "summary": r["executive_summary"],
                "entries_count": len(r["relevant_entries"])
            } for r in results
        ]
        summary_output = {
            "subquestions_researched": len(subquestions),
            "total_articles_found": len(self.all_relevant_information),
            "executive_summaries": executive_summaries
        }
        
        return json.dumps(summary_output, indent=2)
    
    def _synthesize_final_report(self, research_guide: str) -> str:
        """Call synthesizer agent to create final report."""
        self.logger.log(f"[PI] Calling synthesizer with {len(self.all_relevant_information)} articles")
        synthesizer_logger = self.logger.get_sub_logger("synthesizer")
        synthesizer = SynthesizerAgent(logger=synthesizer_logger)
        
        final_report = synthesizer.synthesize_report(
            user_query=self.user_query,
            research_guide=research_guide,
            relevant_information=self.all_relevant_information
        )
        self.logger.log(f"[PI] Final report generated by synthesizer")
        self.logger.log("="*80 + "\n" + final_report + "\n" + "="*80)
        return final_report
    
    def _invoke_functions_from_response(self, response, iteration: int) -> List[Dict]:
        """Execute function calls from PI response and log them."""
        function_outputs = []
        if hasattr(response, 'output'):
            function_calls_found = 0
            for i, response_item in enumerate(response.output):
                if response_item.type == 'function_call':
                    function_calls_found += 1
                    name, call_id = response_item.name, response_item.call_id
                    self.logger.log(f"[PI] Executing function call {function_calls_found}: {name}")
                    try:
                        arguments = json.loads(response_item.arguments)
                        self.logger.log(f"[PI] Function arguments: {arguments}")
                        self.logger.log_json(f"pi_iteration_{iteration}_tool_call_{i+1}_{name}.json", {"name": name, "arguments": arguments})
                        
                        target_tool = self.tool_dispatch.get(name)
                        if target_tool:
                            result = target_tool(**arguments)
                            self.logger.log(f"[PI] Function {name} executed successfully")
                            try:
                                self.logger.log_json(f"pi_iteration_{iteration}_tool_result_{i+1}_{name}.json", json.loads(result))
                            except (json.JSONDecodeError, TypeError):
                                self.logger.log_text(f"pi_iteration_{iteration}_tool_result_{i+1}_{name}.txt", str(result))
                        else:
                            result = f"ERROR: No tool registered for function call: {name}"
                            self.logger.log(f"[PI] ERROR: Function {name} not found")
                    except Exception as e:
                        result = f"ERROR: {e}"
                        self.logger.log(f"[PI] ERROR executing function {name}: {e}")
                    
                    function_outputs.append({"type": "function_call_output", "call_id": call_id, "output": str(result)})
                elif response_item.type == 'reasoning' and response_item.summary:
                    self.logger.log(f"[PI reasoning] {response_item.summary[0].text}")
            
            if function_calls_found == 0:
                self.logger.log(f"[PI] No function calls found in response")
        return function_outputs
