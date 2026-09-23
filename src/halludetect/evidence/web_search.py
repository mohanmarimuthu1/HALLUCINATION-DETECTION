"""Web-search evidence source (docs/contract.md: evidence_source == 'web').

Backed by Tavily, per the Phase 0.2 provider decision recorded in
docs/contract.md. Active only when a Tavily API key is configured; per
the contract, a missing key must behave identically to NoEvidenceSource,
and a live search failure must degrade to no evidence rather than crash
the request - absence of evidence is never papered over by falling back
to a different source or to the LLM's own knowledge.
"""
import httpx

from halludetect.evidence.base import Evidence
from halludetect.logging import get_logger

DEFAULT_TIMEOUT_S = 15.0
DEFAULT_BASE_URL = "https://api.tavily.com/search"
DEFAULT_MAX_RESULTS = 5

_logger = get_logger(__name__)


class WebSearchEvidence:
    def __init__(
        self,
        api_key: str | None,
        *,
        max_results: int = DEFAULT_MAX_RESULTS,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._api_key = api_key
        self._max_results = max_results
        self._base_url = base_url
        self._timeout = timeout

    def fetch(self, query: str) -> list[Evidence]:
        if not self._api_key:
            return []

        try:
            response = httpx.post(
                self._base_url,
                json={
                    "api_key": self._api_key,
                    "query": query,
                    "max_results": self._max_results,
                    "include_answer": False,
                },
                timeout=self._timeout,
            )
        except httpx.HTTPError as exc:
            _logger.warning("web_search_transport_error", error=str(exc))
            return []

        if response.status_code != 200:
            _logger.warning("web_search_http_error", status_code=response.status_code)
            return []

        try:
            results = response.json().get("results", [])
        except ValueError as exc:
            _logger.warning("web_search_invalid_json", error=str(exc))
            return []

        chunks: list[Evidence] = []
        for i, result in enumerate(results[: self._max_results]):
            content = (result.get("content") or "").strip()
            if not content:
                continue
            chunks.append(Evidence(chunk_id=f"web-{i}", text=content, source="web"))
        return chunks
