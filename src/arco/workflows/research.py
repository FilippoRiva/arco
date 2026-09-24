"""A tool-using workflow for web research and general questions."""

from __future__ import annotations

import html
import json
import logging
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, unquote, urlparse

import requests
from langchain_core.tools import tool

from arco.core import AgentType, Workflow

if TYPE_CHECKING:
    from arco.core import Config, Graph

from arco.core.graph import END

logger = logging.getLogger(__name__)


class _ReadablePageParser(HTMLParser):
    """Extract readable text from an HTML page while skipping boilerplate code."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False
        self._SKIP_TAGS = {"script", "style", "noscript", "template", "svg"}
        self._BLOCK_TAGS = {
            "address",
            "article",
            "br",
            "dd",
            "div",
            "dl",
            "dt",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "li",
            "main",
            "p",
            "pre",
            "section",
            "table",
            "td",
            "th",
            "tr",
            "ul",
        }

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "title" and self._skip_depth == 0:
            self._in_title = True
        elif tag in self._BLOCK_TAGS and self._skip_depth == 0:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in self._BLOCK_TAGS and self._skip_depth == 0:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = " ".join(data.split())
        if not text:
            return
        if self._in_title:
            self.title += f" {text}"
        self.parts.append(text)


class _DuckDuckGoParser(HTMLParser):
    """Extract titles, links, and snippets from DuckDuckGo HTML results."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._field: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if tag == "a" and "result__a" in classes:
            self._current = {"title": "", "url": attributes.get("href") or ""}
            self._field = "title"
        elif tag == "a" and "result__snippet" in classes and self._current:
            self._field = "snippet"
            self._current.setdefault("snippet", "")

    def handle_data(self, data: str) -> None:
        if self._current is not None and self._field is not None:
            self._current[self._field] += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._current and self._field == "title":
            self.results.append(self._current)
            self._field = None
        elif tag == "a" and self._field == "snippet":
            self._field = None


def _resolve_result_url(url: str) -> str:
    """Resolve DuckDuckGo redirect URLs to their destination URL."""
    parsed = urlparse(html.unescape(url))
    uddg = parse_qs(parsed.query).get("uddg")
    return unquote(uddg[0]) if uddg else html.unescape(url)


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the public web and return titles, URLs, and snippets.

    Use this tool for current facts, source discovery, documentation, news,
    and any question where web research is more reliable than general model
    knowledge.
    """
    if not query.strip():
        return "Search failed: query cannot be empty."
    max_results = max(1, min(max_results, 10))
    try:
        response = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (compatible; ARCO research agent)"},
            timeout=20,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning(f"Web search failed for {query}: {exc}")
        return f"Search failed for {query!r}: {exc}"

    parser = _DuckDuckGoParser()
    parser.feed(response.text)
    results = []
    for result in parser.results[:max_results]:
        results.append(
            {
                "title": " ".join(result.get("title", "").split()),
                "url": _resolve_result_url(result.get("url", "")),
                "snippet": " ".join(result.get("snippet", "").split()),
            }
        )
    if not results:
        return f"No web results found for {query!r}."
    return "\n\n".join(
        f"[{index}] {result['title']}\nURL: {result['url']}\n"
        f"Snippet: {result['snippet']}"
        for index, result in enumerate(results, start=1)
    )


@tool
def fetch_web_page(url: str, max_chars: int = 12000) -> str:
    """Fetch a web page and return its readable text for detailed analysis.

    Use this after ``web_search`` when exact page content is needed instead of
    relying only on a search-result snippet. HTML markup, scripts, and styles
    are removed. The result is truncated to keep the context manageable.
    """
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "Page fetch failed: URL must use http or https."

    max_chars = max(1000, min(max_chars, 30000))
    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": "arco-research/0.1 (web page reader)",
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1",
            },
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        # Return the failure to the agent so it can try another source, but do
        # not print expected website blocks during normal benchmark runs.
        logger.debug("Page fetch failed for %s: %s", url, exc)
        return f"Page fetch failed for {url!r}: {exc}"

    content_type = response.headers.get("content-type", "").lower()
    if "html" not in content_type and "text/plain" not in content_type:
        return (
            f"Page fetch skipped for {url!r}: unsupported content type "
            f"{content_type or 'unknown'}."
        )

    parser = _ReadablePageParser()
    try:
        parser.feed(response.text)
        parser.close()
    except (AssertionError, ValueError) as exc:
        logger.debug("Could not parse page %s: %s", url, exc)
        return f"Page fetch failed for {url!r}: could not parse HTML"

    text = " ".join(" ".join(parser.parts).split())
    if not text:
        return f"No readable text found at {url!r}."

    title = " ".join(parser.title.split())
    prefix = f"Title: {title}\n" if title else ""
    truncated = len(text) > max_chars
    content = text[:max_chars]
    suffix = "\n… <page content truncated>" if truncated else ""
    return f"URL: {url}\n{prefix}Content:\n{content}{suffix}"


class WebSearchResearcher(Workflow):
    """Route a request through web research, summarization, or knowledge."""

    workflow_id = "websearch_researcher"
    description = (
        "Uses tool-calling agents to research the web, summarize sources, "
        "or answer from general knowledge."
    )

    @staticmethod
    def _route(state: Any) -> str:
        answer = state.get_last_answer(AgentType("ResearchRouter"))
        if answer is None:
            return "CommonKnowledge"
        raw_route = answer.agent_output.get("response", answer.message)
        route_text = str(raw_route).strip()
        if route_text.startswith("```"):
            route_text = route_text.strip("`").strip()
            if route_text.lower().startswith("json"):
                route_text = route_text[4:].strip()
        try:
            parsed_route = json.loads(route_text)
            if isinstance(parsed_route, dict):
                route_text = str(parsed_route.get("route", ""))
        except json.JSONDecodeError, AttributeError, TypeError:
            pass
        route = route_text.lower().replace("_", "").replace("-", "")
        if "websearch" in route or route == "search":
            return "WebSearch"
        if "summary" in route or "summar" in route:
            return "Summary"
        return "CommonKnowledge"

    def initialize(self, config: Config, graph: Graph) -> None:
        from arco.agents import ToolUseAgent

        router = ToolUseAgent(
            agent_name="ResearchRouter",
            role="""You are the routing agent for a web-research workflow.
The workflow contains three specialist agents:
- WebSearch: can search the public web and fetch pages; use it for current or source-based questions.
- Summary: has no tools and synthesizes the web-search results already in the workflow context.
- CommonKnowledge: has no tools and answers from general knowledge. It also understands this workflow's architecture and can explain it when asked.

Choose exactly one route. Use WebSearch for requests needing current information or sources. After a web search, choose WebSearch again if more queries are needed, otherwise choose Summary. Use CommonKnowledge for stable general-knowledge or architecture questions.

You are only a router, not a tool executor. Do not call tools and do not emit tool-call syntax. In particular, never output `multi_tool_use.parallel`, `functions.*`, `recipient`, `tool_uses`, a list of queries, or any other orchestration format. Do not return multiple routes. Your entire response must be exactly one plain JSON object with one of the three allowed route values and no markdown fences or explanation:
{"route": "WebSearch"}
{"route": "Summary"}
{"route": "CommonKnowledge"}""",
            tools=[],
        )
        researcher = ToolUseAgent(
            agent_name="WebSearch",
            role="""You are the web and academic research specialist.
Use web_search for current facts, official documentation, news, and broad public-web research.
Use fetch_web_page after web_search when exact page content is needed rather than relying on a snippet.
Search multiple times when different queries or source types are needed.
Prefer primary and authoritative sources, preserve stable URLs, and do not invent facts.
Once enough evidence has been collected, provide a concise research result with the sources for the Summary agent.""",
            tools=[web_search, fetch_web_page],
        )
        summary = ToolUseAgent(
            agent_name="Summary",
            role="""You are the research summary specialist.
You have no tools. Read the original user request and the complete web-search context from previous age. 
Synthesize a clear, accurate answer, distinguish sourced facts from uncertainty, and include the relevant source URLs.
Answer the user directly rather than describing your internal process.""",
            tools=[],
        )
        common_knowledge = ToolUseAgent(
            agent_name="CommonKnowledge",
            role="""You are the common-knowledge and architecture specialist. 
You have no tools. Answer stable questions from your knowled. 
You also know this workflow architecture: a routing tool-use agent chooses WebSearch, Summary, or CommonKnowledge; WebSearch can call web_search repeatedly; Summary synthesizes the collected web context; and the final specialist answer is returned to the user.
Explain that architecture when the user asks about it.
Do not pretend to have performed web research.""",
            tools=[],
        )

        for agent in [router, researcher, summary, common_knowledge]:
            graph.add_agent(agent)

        graph.set_entry_agent(router)
        graph.add_conditional_edges(
            source=router,
            path=self._route,
            path_map={
                researcher: researcher,
                summary: summary,
                common_knowledge: common_knowledge,
            },
        )
        graph.add_agent_edge(researcher, router)
        graph.add_agent_edge(summary, END)
        graph.add_agent_edge(common_knowledge, END)


__all__ = [
    "WebSearchResearcher",
    "fetch_web_page",
    "web_search",
]
