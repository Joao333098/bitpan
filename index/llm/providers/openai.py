import logging
from typing import List, Optional

from openai import AsyncOpenAI

from ..llm import BaseLLMProvider, LLMResponse, Message

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, model: str, reasoning_effort: Optional[str] = "low"):
        super().__init__(model=model)
        self.client = AsyncOpenAI()
        self.reasoning_effort = reasoning_effort

    async def call(
        self,
        messages: List[Message],
        temperature: float = 1.0,
    ) -> LLMResponse:

        args: dict = {"temperature": temperature}

        if self.model.startswith("o") and self.reasoning_effort:
            args["reasoning_effort"] = self.reasoning_effort
            args["temperature"] = 1

        openai_messages = [msg.to_openai_format() for msg in messages]

        # Nova built-in tools: web grounding + code interpreter
        nova_extra = {
            "system_tools": ["nova_grounding", "nova_code_interpreter"]
        }

        response = None
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                extra_body=nova_extra,
                **args,
            )
        except Exception as exc:
            logger.warning(f"Nova system_tools call failed ({exc}), retrying without extra_body.")

        if response is None:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                **args,
            )

        content = response.choices[0].message.content or ""

        return LLMResponse(
            content=content,
            raw_response=response,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        )
