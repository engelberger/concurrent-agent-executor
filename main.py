"""
generate a random number between 1 and 10, if it is greater than 5 say "eureka", otherwise say "sha"
"""

import aioconsole
import asyncio
import random

from typing import Any
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

from pydantic import BaseModel

from async_agent import AsyncAgentExecutor, BaseParallelizableTool, WaitTool
from async_agent.structured_chat import StructuredChatAgent

load_dotenv()


class RandomNumberToolSchema(BaseModel):
    a: int
    b: int


class RandomNumberTool(BaseParallelizableTool):
    is_parallelizable = True

    name = "RandomNumber"
    description = "Generates a random number between a and b"
    args_schema: RandomNumberToolSchema = RandomNumberToolSchema

    def _run(self, a, b):
        n = random.randint(a, b)
        return f"The random number is: {n}"

    def _arun(self, *args: Any, **kwargs: Any):
        return self._run(*args, **kwargs)


async def main():
    def on_message(who, message):
        print(f"\n{who}: {message}\n")

    llm = ChatOpenAI()

    tools = [
        RandomNumberTool(),
        WaitTool(),
    ]

    agent = StructuredChatAgent.from_llm_and_tools(
        llm=llm,
        tools=tools,
    )

    executor = AsyncAgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        # verbose=True,
        # return_intermediate_steps=True,
    )

    with executor:
        executor.emitter.on("message", on_message)

        while True:
            try:
                _input = await aioconsole.ainput(">>> ")
                executor({"input": _input})
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    asyncio.run(main())
