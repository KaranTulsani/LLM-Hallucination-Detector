"""
Correction Pipeline

Three strategies to fix hallucinated responses:
  1. Constrained re-prompting — cheapest, default first pass
  2. RAG-grounded regeneration — most accurate when KB is available
  3. Critic + Generator loop — most powerful, multi-round refinement

The corrector takes the original query, the flagged response, and the
detection artefacts (unverified claims, evidence) and produces a
revised response.
"""

from __future__ import annotations

import asyncio

from llm_client import LLMClient


class Corrector:
    """Hallucination correction pipeline."""

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient()

    # ── Strategy 1 — Constrained re-prompting ──────────────────────

    async def constrained_reprompt(
        self,
        query: str,
        original_response: str,
        unverified_claims: list[str],
    ) -> str:
        """Re-prompt the LLM, explicitly listing the claims it must revise."""

        claims_list = "\n".join(f"  - {c}" for c in unverified_claims)

        prompt = (
            f"A user asked: \"{query}\"\n\n"
            f"Your previous response contained these unverified or potentially "
            f"incorrect claims:\n{claims_list}\n\n"
            f"Please revise your response following these rules:\n"
            f"1. Remove or correct any claim you cannot verify.\n"
            f"2. If you are unsure about a claim, say so explicitly "
            f'(use phrases like "may", "approximately", "according to some sources").\n'
            f"3. Do NOT state anything you cannot verify.\n"
            f"4. Keep the response helpful and complete.\n\n"
            f"Revised response:"
        )

        return await asyncio.to_thread(
            self.llm.chat,
            [{"role": "user", "content": prompt}],
            temperature=0.0,
        )

    # ── Strategy 2 — RAG-grounded regeneration ─────────────────────

    async def rag_grounded_regeneration(
        self,
        query: str,
        context_docs: list[str],
    ) -> str:
        """Regenerate the answer grounded exclusively in provided docs."""

        docs_text = "\n\n---\n\n".join(context_docs)

        prompt = (
            f"Answer the following question using ONLY the information in "
            f"the provided documents. If the documents don't contain enough "
            f"information to fully answer, say so explicitly.\n\n"
            f"**Question:** {query}\n\n"
            f"**Documents:**\n{docs_text}\n\n"
            f"**Answer:**"
        )

        return await asyncio.to_thread(
            self.llm.chat,
            [{"role": "user", "content": prompt}],
            temperature=0.0,
        )

    # ── Strategy 3 — Critic + Generator loop ───────────────────────

    async def critic_generator_loop(
        self,
        query: str,
        initial_response: str,
        context_docs: list[str] | None = None,
        max_rounds: int = 3,
    ) -> str:
        """Multi-round refinement with a Critic and Generator."""

        current = initial_response
        docs_text = ""
        if context_docs:
            docs_text = "\n\n---\n\n".join(context_docs)

        for round_num in range(max_rounds):
            # ── Critic pass ──
            critic_prompt = (
                f"You are a strict fact-checking critic.\n\n"
                f"**User question:** {query}\n\n"
                f"**Current answer:**\n{current}\n\n"
            )
            if docs_text:
                critic_prompt += f"**Reference documents:**\n{docs_text}\n\n"

            critic_prompt += (
                "Identify specific problems:\n"
                "1. Claims that are factually incorrect\n"
                "2. Claims not supported by the reference documents\n"
                "3. Fabricated citations or references\n"
                "4. Overconfident statements on uncertain topics\n\n"
                "If the answer is satisfactory, respond with exactly: APPROVED\n"
                "Otherwise, list each problem on a new line."
            )

            critique = await asyncio.to_thread(
                self.llm.chat,
                [{"role": "user", "content": critic_prompt}],
                temperature=0.0,
            )

            if "APPROVED" in critique.strip().upper():
                break

            # ── Generator pass ──
            gen_prompt = (
                f"A user asked: \"{query}\"\n\n"
                f"Your previous answer:\n{current}\n\n"
                f"A fact-checker found these problems:\n{critique}\n\n"
            )
            if docs_text:
                gen_prompt += f"Reference documents:\n{docs_text}\n\n"

            gen_prompt += (
                "Please produce a corrected, accurate response. "
                "Fix every issue the critic identified. "
                "If unsure about something, hedge appropriately."
            )

            current = await asyncio.to_thread(
                self.llm.chat,
                [{"role": "user", "content": gen_prompt}],
                temperature=0.0,
            )

        return current