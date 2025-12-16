#!/usr/bin/env python3
"""
subagent_team.py
----------------

SubAgent team consisting of a planner agent and a critic agent working together.
"""

import json
from typing import List, Dict, Any
from openai import OpenAI
from .utils import FileLogger, hybrid_search, bm25_author_keywords_search, bm25_mesh_terms_search


class SubAgentTeam:
    """A team consisting of a planner agent and a critic agent working together."""
    
    def __init__(self, logger: FileLogger, model: str = "gpt-4o"):
        self.client = OpenAI()
        self.logger = logger
        self.model = model
        self.model_defaults = {"model": self.model}
        
        self.planner_tools = [
            {"type": "function", "function": {"name": "hybrid_search", "description": "Perform hybrid search combining FAISS vector search with BM25 title+abstract search.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query"}, "k": {"type": "integer", "description": "Number of results to return", "default": 10}, "alpha": {"type": "number", "description": "Weight for semantic search (0.0 = only BM25, 1.0 = only vector)", "default": 0.5}}, "required": ["query"]}}},
            {"type": "function", "function": {"name": "bm25_author_keywords_search", "description": "Search specifically in author-provided keywords.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query"}, "k": {"type": "integer", "description": "Number of results", "default": 10}}, "required": ["query"]}}},
            {"type": "function", "function": {"name": "bm25_mesh_terms_search", "description": "Search specifically in MeSH terms.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query"}, "k": {"type": "integer", "description": "Number of results", "default": 10}}, "required": ["query"]}}},
        ]
        
        self.tool_dispatch = {
            "hybrid_search": hybrid_search,
            "bm25_author_keywords_search": bm25_author_keywords_search,
            "bm25_mesh_terms_search": bm25_mesh_terms_search
        }
    
    def research_subquestion(self, subquestion: str) -> Dict[str, Any]:
        """Research a specific subquestion using planner and critic agents."""
        self.logger.log(f"[SUBAGENT] Starting research for subquestion: '{subquestion}'")
        
        planner_system_prompt = """
You are a research planner agent. Your task is to devise a search plan to find relevant articles for a given research question.
You have access to three search tools:
1. hybrid_search - Best for comprehensive searches combining semantic and keyword matching. Use the alpha parameter to control the balance: alpha=0.0 means only BM25 (keyword) search, alpha=1.0 means only vector (semantic) search. For exact keywords (e.g., protein names), use lower alpha. For semantic similarity, use higher alpha.
2. bm25_author_keywords_search - Good for finding papers by specific research focus or methodology via author keywords.
3. bm25_mesh_terms_search - Ideal for searching with medical terminology and standardized concepts using MeSH terms.

Based on the user's question, decide which search tool(s) to use. Formulate effective search queries.
You can perform multiple searches with different strategies to ensure comprehensive coverage.
"""
        planner_user_prompt = f"Please devise a search plan for the following question:\n\n{subquestion}"

        all_results = []
        max_iterations = 5
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            self.logger.log(f"[SUBAGENT] Starting iteration {iteration}/{max_iterations}")
            self.logger.log_text(f"iteration_{iteration}_planner_prompt.txt", f"SYSTEM: {planner_system_prompt}\nUSER: {planner_user_prompt}")
            try:
                self.logger.log(f"[SUBAGENT] Planner phase - calling {self.model}...")
                planner_model_config = dict(self.model_defaults)
                planner_model_config["tools"] = self.planner_tools
                
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": planner_system_prompt},
                        {"role": "user", "content": planner_user_prompt}
                    ],
                    **planner_model_config
                )
                
                self.logger.log(f"[SUBAGENT] Planner responded, executing function calls...")
                function_responses = self._invoke_functions_from_response(response, iteration)
                self.logger.log(f"[SUBAGENT] Executed {len(function_responses)} function calls")
                
                if function_responses:
                    new_results = []
                    for func_response in function_responses:
                        try:
                            result_data = json.loads(func_response["output"])
                            if "results" in result_data:
                                new_results.extend(result_data["results"])
                        except Exception as e:
                            self.logger.log(f"[SUBAGENT] Error parsing function response: {e}")
                    
                    if not new_results:
                        self.logger.log("[SUBAGENT] No new results found. Ending loop.")
                        break

                    self.logger.log(f"[SUBAGENT] Found {len(new_results)} new results. Critic phase - calling {self.model}...")

                    critic_system_prompt = '''
You are a research critic agent. Your task is to evaluate the relevance of the following articles to the research question.
For each article, provide a relevance score from 1 to 10 (1 = not relevant, 10 = highly relevant) and a brief justification.
Also, provide an overall assessment: should we continue searching or stop?

Respond in JSON format with two keys: "scores" and "decision".
- "scores": a list of objects, each with "pmid", "relevance_score", and "justification".
- "decision": either "CONTINUE" or "STOP". If continuing, add a "suggestion" key with ideas for the next search.
'''
                    critic_user_prompt = f"""
Research Question: "{subquestion}"

Articles to evaluate:
{json.dumps(new_results, indent=2)}
"""
                    self.logger.log_text(f"iteration_{iteration}_critic_prompt.txt", f"SYSTEM: {critic_system_prompt}\nUSER: {critic_user_prompt}")
                    
                    critic_response = self.client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": critic_system_prompt},
                            {"role": "user", "content": critic_user_prompt}
                        ],
                        response_format={"type": "json_object"},
                        **self.model_defaults
                    )
                    critic_decision_str = critic_response.choices[0].message.content
                    self.logger.log_text(f"iteration_{iteration}_critic_decision.txt", critic_decision_str)
                    
                    try:
                        critic_output = json.loads(critic_decision_str)
                        self.logger.log_json(f"iteration_{iteration}_critic_output.json", critic_output)

                        relevance_threshold = 6
                        scored_results = critic_output.get("scores", [])
                        
                        new_results_map = {str(r.get('pmid')): r for r in new_results}

                        for scored_result in scored_results:
                            pmid = str(scored_result.get('pmid'))
                            if pmid in new_results_map:
                                new_results_map[pmid]['relevance_score'] = scored_result.get('relevance_score')
                                new_results_map[pmid]['relevance_justification'] = scored_result.get('justification')

                        relevant_results = [r for r in new_results if r.get('relevance_score', 0) >= relevance_threshold]
                        all_results.extend(relevant_results)
                        
                        self.logger.log(f"[SUBAGENT] Added {len(relevant_results)} relevant results (score >= {relevance_threshold}). Total relevant results: {len(all_results)}")

                        critic_decision = critic_output.get("decision", "STOP").upper()
                        self.logger.log(f"[SUBAGENT] Critic decision: {critic_decision}")

                        if critic_decision == "STOP":
                            self.logger.log(f"[SUBAGENT] Critic says STOP - ending research loop")
                            break
                        else:
                            self.logger.log(f"[SUBAGENT] Critic says CONTINUE - preparing next iteration")
                            suggestion = critic_output.get("suggestion", "No suggestion provided.")
                            planner_user_prompt = f'''
Question: {subquestion}

We have already found {len(all_results)} relevant articles.
The critic has suggested the following for the next steps: {suggestion}

Please devise a new search plan based on this feedback.
'''
                    except json.JSONDecodeError as e:
                        self.logger.log(f"[SUBAGENT] Error parsing critic JSON response: {e}. Ending loop.")
                        self.logger.log(f"Raw critic response: {critic_decision_str}")
                        break

                else:
                    self.logger.log("[SUBAGENT] Planner did not request any tool calls. Ending loop.")
                    break
            except Exception as e:
                self.logger.log(f"[SUBAGENT] Error in iteration {iteration}: {e}")
                break
        
        self.logger.log(f"[SUBAGENT] Research loop completed. Total results: {len(all_results)}")
        self.logger.log_json("all_collected_articles.json", all_results)
        
        self.logger.log(f"[SUBAGENT] Generating executive summary with {self.model}...")
        summary_system_prompt = '''
You are a research analyst. Your task is to create an executive summary of research findings for a given question.
Based on the provided research findings, please provide:
1. A concise executive summary (2-3 paragraphs)
2. Key findings with citations (using PMID)
3. Main conclusions relevant to the research question
'''
        summary_user_prompt = f"""
Question: {subquestion}

Total relevant articles found: {len(all_results)}

Research Findings:
{json.dumps(all_results[:20], indent=2)}  # Top 20 relevant articles are provided for context
"""
        self.logger.log_text("executive_summary_prompt.txt", f"SYSTEM: {summary_system_prompt}\nUSER: {summary_user_prompt}")
        
        try:
            summary_response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": summary_system_prompt},
                    {"role": "user", "content": summary_user_prompt}
                ],
                **self.model_defaults
            )
            executive_summary = summary_response.choices[0].message.content
            self.logger.log(f"[SUBAGENT] Executive summary generated, length: {len(executive_summary)} chars")
            self.logger.log_text("executive_summary.txt", executive_summary)
        except Exception as e:
            self.logger.log(f"[SUBAGENT] Error generating summary: {e}")
            executive_summary = f"Error generating summary: {str(e)}"
        
        self.logger.log(f"[SUBAGENT] Subquestion research complete for: '{subquestion}'")
        return {
            "subquestion": subquestion,
            "executive_summary": executive_summary,
            "relevant_entries": all_results
        }
    
    def _invoke_functions_from_response(self, response, iteration: int) -> List[Dict]:
        """Execute function calls from response and log them."""
        function_outputs = []
        if hasattr(response, 'choices') and response.choices and hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
            self.logger.log(f"[SUBAGENT] Found {len(response.choices[0].message.tool_calls)} tool calls to execute")
            for i, tool_call in enumerate(response.choices[0].message.tool_calls):
                name, call_id = tool_call.function.name, tool_call.id
                self.logger.log(f"[SUBAGENT] Executing tool call {i+1}: {name}")
                try:
                    arguments = json.loads(tool_call.function.arguments)
                    self.logger.log(f"[SUBAGENT] Tool arguments: {arguments}")
                    self.logger.log_json(f"iteration_{iteration}_tool_call_{i+1}_{name}.json", {"name": name, "arguments": arguments})
                    
                    target_tool = self.tool_dispatch.get(name)
                    if target_tool:
                        result = target_tool(**arguments)
                        self.logger.log(f"[SUBAGENT] Tool {name} executed successfully")
                        try:
                            self.logger.log_json(f"iteration_{iteration}_tool_result_{i+1}_{name}.json", json.loads(result))
                        except json.JSONDecodeError:
                            self.logger.log_text(f"iteration_{iteration}_tool_result_{i+1}_{name}.txt", result)
                    else:
                        result = f"ERROR: No tool registered for function call: {name}"
                        self.logger.log(f"[SUBAGENT] ERROR: Tool {name} not found")
                except Exception as e:
                    result = f"ERROR: {e}"
                    self.logger.log(f"[SUBAGENT] ERROR executing tool {name}: {e}")
                
                function_outputs.append({"type": "function_call_output", "call_id": call_id, "output": str(result)})
        else:
            self.logger.log(f"[SUBAGENT] No tool calls found in response")
        return function_outputs