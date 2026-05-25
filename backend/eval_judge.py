"""DeepEval custom-model wrapper that grades with Gemini instead of OpenAI.

DeepEval metrics accept any subclass of `DeepEvalBaseLLM` via their `model=`
kwarg. We use `ChatGoogleGenerativeAI.with_structured_output(schema)`, which
configures Gemini's native responseSchema — that's the reliable fix for the
intermittent JSON parsing failures DeepEval hits when handed raw Gemini text
(see confident-ai/deepeval#982).

Picks `gemini-2.5-flash` by default (1M context window) because faithfulness
and contextual-relevancy prompts inline the full retrieval context and blow
through the chat model's smaller window.
"""

from __future__ import annotations

import os
from typing import Any

from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_google_genai import ChatGoogleGenerativeAI


class GeminiJudge(DeepEvalBaseLLM):
    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or os.getenv("EVAL_METRIC_MODEL", "gemini-2.5-flash")
        self._chat = ChatGoogleGenerativeAI(
            model=self._model_name,
            google_api_key=os.environ["GEMINI_API_KEY"],
            temperature=0,
        )

    def load_model(self) -> ChatGoogleGenerativeAI:
        return self._chat

    def get_model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, schema: Any) -> Any:
        structured = self._chat.with_structured_output(schema)
        return structured.invoke(prompt)

    async def a_generate(self, prompt: str, schema: Any) -> Any:
        structured = self._chat.with_structured_output(schema)
        return await structured.ainvoke(prompt)
