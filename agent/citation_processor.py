#!/usr/bin/env python3
"""
citation_processor.py
---------------------

Citation post-processing utilities for converting in-text citations
to numbered references with a formatted References section.
"""

import re
from typing import List, Dict


def extract_pmids_from_citation(citation_text: str) -> List[str]:
    """
    Extract all PMIDs from a citation string.

    Args:
        citation_text: Text inside brackets like 'Author, Year, PMID; Author2, Year2, PMID2'

    Returns:
        List of PMID strings (7-8 digit numbers)
    """
    return re.findall(r'\b(\d{7,8})\b', citation_text)


def format_single_reference(num: int, article: Dict, pmid: str) -> str:
    """
    Format a single reference entry.

    Format: [num] Authors (Year). Title. Journal. PMID: XXXXX

    Args:
        num: Reference number
        article: Article data dict with authors, year, title, journal
        pmid: PubMed ID

    Returns:
        Formatted reference string
    """
    authors = article.get('authors', [])
    if isinstance(authors, list) and authors:
        if len(authors) <= 2:
            author_str = ', '.join(authors)
        else:
            author_str = f"{authors[0]} et al."
    else:
        author_str = "Unknown"

    year = article.get('year', 'N/A')
    title = article.get('title', 'Unknown title')
    journal = article.get('journal', '')

    ref = f"[{num}] {author_str} ({year}). {title}."
    if journal:
        ref += f" {journal}."
    ref += f" PMID: {pmid}"

    return ref


def format_references(pmid_order: List[str], pmid_to_article: Dict) -> str:
    """
    Format the References section.

    Args:
        pmid_order: List of PMIDs in order of first appearance
        pmid_to_article: Dict mapping PMID to article data

    Returns:
        Formatted References section string
    """
    if not pmid_order:
        return ""

    lines = ["## References", ""]
    for i, pmid in enumerate(pmid_order, 1):
        article = pmid_to_article.get(pmid, {})
        lines.append(format_single_reference(i, article, pmid))

    return "\n".join(lines)


def process_citations(report: str, articles: List[Dict]) -> str:
    """
    Process a report to replace in-text citations with numbered references.

    Transforms citations like [Author, Year, PMID] or [Author, Year, PMID; Author2, Year2, PMID2]
    into numbered format [1], [1,2], etc. and appends a References section.

    Args:
        report: The report text with citations in [Author, Year, PMID] format
        articles: List of article dicts with pmid, authors, year, title, journal

    Returns:
        Processed report with numbered citations and References section
    """
    if not report or not articles:
        return report

    # Build PMID to article lookup
    pmid_to_article = {str(a.get('pmid')): a for a in articles if a.get('pmid')}

    # Pattern to match bracketed content: [...]
    citation_pattern = r'\[([^\[\]]+)\]'

    pmid_order = []  # Track order of first appearance
    pmid_to_refnum = {}

    def replace_citation(match):
        citation_content = match.group(1)
        pmids = extract_pmids_from_citation(citation_content)

        if not pmids:
            # Not a citation with PMID, return as-is
            return match.group(0)

        ref_nums = []
        for pmid in pmids:
            if pmid not in pmid_to_refnum:
                pmid_order.append(pmid)
                pmid_to_refnum[pmid] = len(pmid_order)
            ref_nums.append(str(pmid_to_refnum[pmid]))

        return '[' + ','.join(ref_nums) + ']'

    # Replace all citations
    processed_report = re.sub(citation_pattern, replace_citation, report)

    # Build and append References section
    references = format_references(pmid_order, pmid_to_article)

    if references:
        return processed_report + "\n\n" + references

    return processed_report
