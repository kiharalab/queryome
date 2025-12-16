# #!/usr/bin/env python3
# """
# synthesizer.py
# --------------

# Synthesizer agent responsible for creating final research reports with proper citations.
# """

# import json
# from typing import List, Dict, Any
# from openai import OpenAI
# from .utils import FileLogger


# class SynthesizerAgent:
#     """Agent responsible for creating final research reports with proper citations."""
    
#     def __init__(self, logger: FileLogger, model: str = "gpt-4.1"):
#         self.client = OpenAI()
#         self.logger = logger
#         self.model = model
#         self.model_defaults = {"model": self.model}
    
#     def synthesize_report(self, user_query: str, research_guide: str, relevant_information: List[Dict[str, Any]]) -> str:
#         """Synthesize a final research report based on collected information."""
#         self.logger.log(f"[SYNTHESIZER] Starting synthesis for query: '{user_query}'")
#         self.logger.log(f"[SYNTHESIZER] Research guide length: {len(research_guide)} chars")
#         self.logger.log(f"[SYNTHESIZER] Relevant information count: {len(relevant_information)} articles")
        
#         print("="*80)
#         print("RELEVANT INFORMATION COLLECTED:")
#         print(json.dumps(relevant_information, indent=2))
#         print("="*80)

#         synthesis_prompt = f"""
#             You are a research synthesizer agent. Your task is to create a comprehensive research report based on the collected information.

#             Original User Query: {user_query}

#             Research Guide from PI Agent: {research_guide}

#             Available Research Information:
#             {json.dumps(relevant_information, indent=2)}

#             Instructions:
#             1. Create a well-structured research report that directly addresses the user's query
#             2. Use proper academic citations in the format (Author et al., Year, PMID: XXXXXX)
#             3. Organize information logically with clear sections
#             4. Synthesize findings rather than just listing them
#             5. Highlight key insights and implications
#             6. Include limitations and areas for future research if applicable
#             7. Ensure all claims are properly cited
#             8. NEVER cite articles that are not in the relevant_information list
#             9. Never fabricate citations or results

#             MOST IMPORTANT: ENSURE THAT CITATIONS ARE ACCURATE AND REFER TO THE PROVIDED RELEVANT INFORMATION ONLY.

#             Please write a comprehensive research report now.
#         """
#         self.logger.log_text("synthesis_prompt.txt", synthesis_prompt)
#         self.logger.log_json("synthesis_input_articles.json", relevant_information)
        
#         try:
#             self.logger.log(f"[SYNTHESIZER] Calling {self.model} for synthesis...")
#             response = self.client.chat.completions.create(
#                 messages=[{"role": "user", "content": synthesis_prompt}],
#                 **self.model_defaults
#             )
#             final_report = response.choices[0].message.content
#             self.logger.log(f"[SYNTHESIZER] Synthesis complete, response length: {len(final_report)} chars")
#             self.logger.log_text("final_report.md", final_report)
#             return final_report
#         except Exception as e:
#             self.logger.log(f"[SYNTHESIZER] ERROR in synthesis: {str(e)}")
#             return f"ERROR in synthesis: {str(e)}"
        

#!/usr/bin/env python3
"""
synthesizer.py
--------------

Synthesizer agent responsible for creating final research reports with proper citations.
"""

import json
from typing import List, Dict, Any
from openai import OpenAI
from .utils import FileLogger

class SynthesizerAgent:
    """Agent responsible for creating final research reports with proper citations."""
    
    def __init__(self, logger: FileLogger, model: str = "o4-mini"):
        self.client = OpenAI()
        self.logger = logger
        self.model = model
        self.model_defaults = {"model": self.model}
        # Developer instructions set as system prompt
        self.system_prompt = (
            "You are a research synthesizer agent. Your task is to create a comprehensive research report based on the collected information.\n"
            "The user message will include the original query, research guide, and available articles.\n"
            "Instructions:\n"
            "1. Create a well-structured research report that directly addresses the user's query.\n"
            "2. Use in-text citations in the format [Author, Year, PMID]. For multiple citations: [Author, Year, PMID; Author2, Year2, PMID2].\n"
            "3. Organize information logically with clear sections.\n"
            "4. Synthesize findings rather than just listing them.\n"
            "5. Highlight key insights and implications.\n"
            "6. Include limitations and areas for future research if applicable.\n"
            "7. Ensure all claims are properly cited.\n"
            "8. NEVER cite articles that are not in the provided list.\n"
            "9. Never fabricate citations or results.\n"
        )
    
    def synthesize_report(
        self,
        user_query: str,
        research_guide: str,
        relevant_information: List[Dict[str, Any]]
    ) -> str:
        """Synthesize a final research report based on collected information."""
        self.logger.log(f"[SYNTHESIZER] Starting synthesis for query: '{user_query}'")
        self.logger.log(f"[SYNTHESIZER] Research guide length: {len(research_guide)} chars")
        self.logger.log(f"[SYNTHESIZER] Relevant information count: {len(relevant_information)} articles")

        # Log prompts
        self.logger.log_text("system_prompt.txt", self.system_prompt)
        sorted_articles = sorted(
            relevant_information,
            key=lambda article: (
                article.get("relevance_score")
                if article.get("relevance_score") is not None
                else article.get("score", 0)
            ),
            reverse=True,
        )
        formatted_articles = []
        for idx, article in enumerate(sorted_articles, start=1):
            authors = article.get("authors")
            if isinstance(authors, list):
                authors_text = ", ".join(authors)
            elif authors:
                authors_text = str(authors)
            else:
                authors_text = "Unknown authors"
            formatted_articles.append(
                f"{idx}. Title: {article.get('title', 'Unknown title')}\n"
                f"   PMID: {article.get('pmid', 'N/A')}\n"
                f"   Authors: {authors_text}\n"
                f"   Text: {article.get('text', 'No text provided.')}\n"
                f"   Relevance Justification: {article.get('relevance_justification', 'No justification provided.')}"
            )
        articles_section = (
            "\n\n".join(formatted_articles) if formatted_articles else "No articles available."
        )
        user_content = (
            f"Original User Query: {user_query}\n\n"
            f"Research Guide from PI Agent: {research_guide}\n\n"
            f"Available Research Information:\n{articles_section}\n\n"
            "Please write a comprehensive research report now."
        )
        self.logger.log_text("synthesis_prompt.txt", user_content)
        self.logger.log_json("synthesis_input_articles.json", sorted_articles)

        try:
            self.logger.log(f"[SYNTHESIZER] Calling {self.model} for synthesis...")
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content}
            ]
            response = self.client.chat.completions.create(
                messages=messages,
                **self.model_defaults
            )
            final_report = response.choices[0].message.content
            self.logger.log(f"[SYNTHESIZER] Synthesis complete, response length: {len(final_report)} chars")
            self.logger.log_text("final_report_raw.md", final_report)

            # Post-process citations to numbered format with References section
            from .citation_processor import process_citations
            final_report = process_citations(final_report, relevant_information)
            self.logger.log(f"[SYNTHESIZER] Citation processing complete, final length: {len(final_report)} chars")
            self.logger.log_text("final_report.md", final_report)
            return final_report
        except Exception as e:
            self.logger.log(f"[SYNTHESIZER] ERROR in synthesis: {str(e)}")
            return f"ERROR in synthesis: {str(e)}"
