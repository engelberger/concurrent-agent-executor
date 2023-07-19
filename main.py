"""
generate a random number between 1 and 10, if it is greater than 5 say "eureka", otherwise say "sha"
"""

import aioconsole
import asyncio
import random
import time

from typing import Any
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import MessagesPlaceholder
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel
from colorama import Back, Style

from async_agent import AsyncAgentExecutor, BaseParallelizableTool
from async_agent.tools import WaitTool
from async_agent.structured_chat import StructuredChatAgent

load_dotenv()


class RandomNumberToolSchema(BaseModel):
    a: int
    b: int


class RandomNumberTool(BaseParallelizableTool):
    is_parallelizable = True

    name = "RandomNumber"
    description = "Schedules a random number generation between a and b; once invoked, you must wait for the result to be ready"
    args_schema: RandomNumberToolSchema = RandomNumberToolSchema

    def _run(self, a, b):
        try:
            time.sleep(5)
            n = random.randint(a, b)
            return f"The random number is: {n}"
        except Exception as e:
            return f"Error: {e}"

    def _arun(self, *args: Any, **kwargs: Any):
        return self._run(*args, **kwargs)


async def main():
    def on_message(who, message):
        print(f"\n{Back.GREEN}{who}{Style.RESET_ALL}: {message}\n")

    llm = ChatOpenAI()

    tools = [
        RandomNumberTool(),
        WaitTool(),
    ]

    chat_history = MessagesPlaceholder(variable_name="chat_history")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = StructuredChatAgent.from_llm_and_tools(
        llm=llm,
        tools=tools,
        memory_prompts=[chat_history],
        input_variables=["input", "agent_scratchpad", "chat_history"],
    )

    executor = AsyncAgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        # return_intermediate_steps=True,
        memory=memory,
    )

    with executor:
        executor.emitter.on("message", on_message)

        while True:
            try:
                _input = await aioconsole.ainput(">>> ")
                executor(
                    {
                        "input": _input,
                    }
                )
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    asyncio.run(main())
