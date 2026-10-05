"""Streamlit demo UI for HALLUDETECT v2 (Phase 5.4).

    streamlit run app.py        (repo-root entry point)
    streamlit run src/halludetect/ui/app.py

Needs the API running (`uvicorn halludetect.api.main:app`). Presentation
only: every answer, verdict, label and quote shown here comes from the API
(`/v1/chat`, `/v1/verify`, `/v1/models/stats`) unchanged.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from halludetect.ui.client import (
    ApiError,
    chat,
    default_api_key,
    default_api_url,
    is_healthy,
    model_stats,
    parse_evidence,
    verify,
)

_MODES = ["Ask", "Check an answer", "Model performance"]

_EXAMPLES: dict[str, dict[str, str]] = {
    "Grounded - every claim is in the evidence": {
        "question": "How tall is the Eiffel Tower, when was it completed, and where is it?",
        "answer": (
            "The Eiffel Tower is 330 metres tall, was completed in 1889, "
            "and stands on the Champ de Mars in Paris."
        ),
        "evidence": (
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, "
            "France. It is 330 metres (1,083 ft) tall.\n\n"
            "Construction of the Eiffel Tower began in 1887 and it was completed in 1889 "
            "for the World Fair."
        ),
    },
    "Hallucinated - wrong facts, and one the evidence never mentions": {
        "question": "Tell me about the Eiffel Tower.",
        "answer": (
            "The Eiffel Tower is 450 metres tall, was completed in 1912, "
            "and was designed by Antoni Gaudi."
        ),
        "evidence": (
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, "
            "France. It is 330 metres (1,083 ft) tall.\n\n"
            "Construction of the Eiffel Tower began in 1887 and it was completed in 1889 "
            "for the World Fair."
        ),
    },
    "No evidence - the service must refuse to guess": {
        "question": "How tall is the Eiffel Tower?",
        "answer": "The Eiffel Tower is 330 metres tall, was completed in 1889, and is in Paris.",
        "evidence": "",
    },
}

_VERDICT_STYLE = {
    "GROUNDED": (
        st.success,
        "GROUNDED",
        "Every verifiable claim is supported by a quote found verbatim in the evidence.",
    ),
    "CONTRADICTED": (st.error, "CONTRADICTED", "At least one claim is contradicted by the evidence."),
    "NOT_ENOUGH_INFO": (
        st.warning,
        "NOT ENOUGH INFO",
        "At least one claim could not be confirmed from the evidence given.",
    ),
    "NOT_VERIFIABLE": (st.info, "NOT VERIFIABLE", ""),
}

_CHAT_NO_EVIDENCE = (
    "This answer was not checked: no sources were given and this deployment has no web search "
    "configured. Do not treat it as correct. Paste sources above to check the next answer against them."
)

_REASON_TEXT = {
    "no_evidence_configured": (
        "No evidence was supplied, so there was nothing to check the answer against. "
        "The service does not fall back on the model's own knowledge - paste the source "
        "passages the answer should be grounded in."
    ),
    "insufficient_verifiable_claims": (
        "Fewer than 3 factual claims were found, which is too few to score the answer as a "
        "whole. The per-claim results below are still valid - add more of the answer to get "
        "an overall verdict."
    ),
}

_LABEL_BADGE = {
    "SUPPORTED": ":green[SUPPORTED]",
    "CONTRADICTED": ":red[CONTRADICTED]",
    "NOT_ENOUGH_INFO": ":orange[NOT ENOUGH INFO]",
}


def _load_example() -> None:
    example = _EXAMPLES.get(st.session_state["example"])
    if example is None:
        return
    st.session_state["question"] = example["question"]
    st.session_state["answer"] = example["answer"]
    st.session_state["evidence"] = example["evidence"]


def _sidebar() -> tuple[str, str | None]:
    with st.sidebar:
        st.header("Connection")
        api_url = st.text_input("API URL", value=default_api_url()).rstrip("/")
        api_key = st.text_input(
            "API key",
            value=default_api_key() or "",
            type="password",
            help="One of CLIENT_API_KEYS from the service's .env. Filled in automatically when set.",
        )
        if is_healthy(api_url):
            st.success("API is running")
        else:
            st.error("API is not reachable")
            st.caption("Start it in another terminal:")
            st.code(".venv/Scripts/python.exe -m uvicorn halludetect.api.main:app", language="bash")
    return api_url, api_key or None


def _render_result(result: dict[str, Any], *, reason_text: dict[str, str] = _REASON_TEXT) -> None:
    verdict = result["verdict"]
    show, title, blurb = _VERDICT_STYLE.get(verdict, (st.info, verdict, ""))
    if verdict == "NOT_VERIFIABLE":
        blurb = reason_text.get(result.get("reason") or "", "The service abstained from giving a verdict.")
    show(f"**{title}**  \n{blurb}")

    n = result["n_verifiable_claims"]
    col_ground, col_claims, col_model, col_time = st.columns(4)
    if verdict == "NOT_VERIFIABLE":
        # docs/contract.md rule 1: no bare percentage for n <= 2.
        col_ground.metric("Groundedness", "-")
    else:
        low, high = result["groundedness_ci"]
        col_ground.metric(
            "Groundedness",
            f"{result['groundedness']:.0%}",
            help=f"95% Wilson interval: {low:.0%} - {high:.0%}",
        )
    col_claims.metric("Verifiable claims", n)
    col_model.metric("Model", result["model_used"]["model"].split("/")[-1])
    col_time.metric("Time", f"{result['timings_ms']['total'] / 1000:.1f}s")

    claims = result["claims"]
    if claims:
        st.subheader("Claims")
        for claim in claims:
            with st.container(border=True):
                st.markdown(f"{_LABEL_BADGE.get(claim['label'], claim['label'])}  {claim['text']}")
                if claim["quote"]:
                    mark = "verified verbatim in evidence" if claim["quote_verified"] else "NOT found in evidence"
                    st.caption(f"Quote ({mark}), from {', '.join(claim['evidence_chunk_ids']) or 'no chunk'}:")
                    st.markdown(f"> {claim['quote']}")
                else:
                    st.caption("No supporting quote in the evidence.")

    st.caption(
        f"p_hallucinated = {result['p_hallucinated']:.2f} from `{result['calibration_version']}`, "
        "fitted on 150 labeled golden-set answers - a rough guide; the per-claim labels "
        f"and quotes are the evidence. Cost ${result['cost_usd']:.4f}."
    )
    with st.expander("Raw API response"):
        st.json(result)


def _verify_mode(api_url: str, api_key: str | None) -> None:
    st.write(
        "Checks whether an answer is supported by the evidence you give it. Every claim marked "
        "supported carries a quote that was found verbatim in that evidence; with no evidence, "
        "it refuses to guess."
    )
    st.selectbox(
        "Try an example",
        ["-", *_EXAMPLES],
        key="example",
        on_change=_load_example,
    )

    st.text_input("Question (optional - improves claim extraction)", key="question")
    st.text_area("Answer to check", key="answer", height=120)
    st.text_area(
        "Evidence - one passage per paragraph, separated by a blank line",
        key="evidence",
        height=180,
    )

    if st.button("Verify", type="primary"):
        answer = st.session_state.get("answer", "").strip()
        if not answer:
            st.warning("Enter an answer to check.")
            return
        evidence = parse_evidence(st.session_state.get("evidence", ""))
        with st.spinner("Extracting claims and checking them against the evidence (usually 10-45s)..."):
            try:
                st.session_state["result"] = verify(
                    api_url,
                    api_key,
                    answer=answer,
                    question=st.session_state.get("question", "").strip() or None,
                    evidence=evidence,
                )
                st.session_state.pop("error", None)
            except ApiError as exc:
                st.session_state["error"] = exc
                st.session_state.pop("result", None)

    error = st.session_state.get("error")
    if error is not None:
        st.error(f"**{error.message}**" + (f"  \n{error.hint}" if error.hint else ""))
    elif "result" in st.session_state:
        _render_result(st.session_state["result"])


def _render_turn(turn: dict[str, Any]) -> None:
    with st.chat_message(turn["role"]):
        if "error" in turn:
            error = turn["error"]
            st.error(f"**{error.message}**" + (f"  \n{error.hint}" if error.hint else ""))
            return
        st.markdown(turn["content"])
        reply = turn.get("reply")
        if reply is None:
            return
        verification = reply["verification"]
        caption = f"Answered by {reply['answer_model']['model']} in {reply['answer_ms'] / 1000:.1f}s"
        # With no evidence the pipeline returns before calling any model.
        if verification.get("reason") != "no_evidence_configured":
            caption += f", checked by {verification['model_used']['model']}"
        st.caption(caption + ".")
        _render_result(reply["verification"], reason_text={**_REASON_TEXT, "no_evidence_configured": _CHAT_NO_EVIDENCE})


def _chat_mode(api_url: str, api_key: str | None) -> None:
    st.write(
        "Ask a question. A model answers it, then a different model checks every claim in the "
        "answer against the sources. Without sources (or web search on the server) the answer "
        "is shown but marked as not checked."
    )
    turns: list[dict[str, Any]] = st.session_state.setdefault("chat", [])
    with st.expander("Sources (optional) - one passage per paragraph"):
        st.text_area("Sources", key="chat_evidence", height=150, label_visibility="collapsed")
    if turns and st.button("New chat"):
        turns.clear()

    for past in turns:
        _render_turn(past)

    question = st.chat_input("Ask a question")
    if not question:
        return
    history = [
        {"role": t["role"], "content": t["content"]} for t in turns if "error" not in t and t.get("content")
    ]
    user_turn = {"role": "user", "content": question}
    turns.append(user_turn)
    _render_turn(user_turn)
    with st.spinner("Answering, then checking each claim (usually 20-90s)..."):
        try:
            reply = chat(
                api_url,
                api_key,
                question=question,
                history=history,
                evidence=parse_evidence(st.session_state.get("chat_evidence", "")),
            )
            turn: dict[str, Any] = {"role": "assistant", "content": reply["answer"], "reply": reply}
        except ApiError as exc:
            turn = {"role": "assistant", "error": exc}
    turns.append(turn)
    _render_turn(turn)


def _fmt_rate(value: float | None) -> str:
    return "-" if value is None else f"{value:.0%}"


def _fmt_seconds(value: float | None) -> str:
    return "-" if value is None else f"{value / 1000:.1f}s"


def _stats_mode(api_url: str, api_key: str | None) -> None:
    st.write(
        "How each model has done on this API instance: as the answering model, how often its "
        "answers had unsupported or contradicted claims; as the checking model, how fast and "
        "reliable it was."
    )
    if st.button("Refresh") or "stats" not in st.session_state:
        try:
            st.session_state["stats"] = model_stats(api_url, api_key)
            st.session_state.pop("stats_error", None)
        except ApiError as exc:
            st.session_state["stats_error"] = exc
            st.session_state.pop("stats", None)

    error = st.session_state.get("stats_error")
    if error is not None:
        st.error(f"**{error.message}**" + (f"  \n{error.hint}" if error.hint else ""))
        return
    stats = st.session_state["stats"]
    rows = sorted(stats["models"], key=lambda r: r["last_used"] or "", reverse=True)
    if not rows:
        st.info(f"No model has been called since {stats['since']}.")
        return
    st.dataframe(
        [
            {
                "Model": row["model"],
                "Provider": row["provider"],
                "Answers": f"{row['answer']['calls']} ({row['answer']['failures']} failed)",
                "Unsupported": _fmt_rate(row["answer"]["unsupported_rate"]),
                "Contradicted": _fmt_rate(row["answer"]["contradicted_rate"]),
                "Groundedness": _fmt_rate(row["answer"]["mean_groundedness"]),
                "Answer avg / p95": (
                    f"{_fmt_seconds(row['answer']['avg_latency_ms'])} / "
                    f"{_fmt_seconds(row['answer']['p95_latency_ms'])}"
                ),
                "Checks": f"{row['verify']['calls']} ({row['verify']['failures']} failed)",
                "Check avg": _fmt_seconds(row["verify"]["avg_latency_ms"]),
                "Checks unsupported": _fmt_rate(row["verify"]["unsupported_rate"]),
                "Status": "cooling down" if row["in_cooldown"] else (row["last_error"] or "ok"),
                "Last used": row["last_used"] or "-",
            }
            for row in rows
        ],
        hide_index=True,
    )
    st.caption(
        f"Since {stats['since']}, for this API instance only; the numbers reset when it restarts. "
        "Rates count answers with at least 3 checkable claims and some evidence. A high "
        "'Checks unsupported' rate points at the checking model, not the answers it checked."
    )


def main() -> None:
    st.set_page_config(page_title="HALLUDETECT", layout="wide")
    st.title("HALLUDETECT")
    api_url, api_key = _sidebar()

    mode = st.radio("Mode", _MODES, key="mode", horizontal=True, label_visibility="collapsed")
    if mode == "Ask":
        _chat_mode(api_url, api_key)
    elif mode == "Check an answer":
        _verify_mode(api_url, api_key)
    else:
        _stats_mode(api_url, api_key)


# Streamlit runs a script with __name__ == "__main__"; the guard keeps the
# repo-root app.py from rendering the page twice when it imports main().
if __name__ == "__main__":
    main()
