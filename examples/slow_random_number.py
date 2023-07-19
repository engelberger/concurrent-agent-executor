"""
This example defines an intentionally-slow parallelizable tool that generates a random number between a and b. Showcases how the agent gets triggered twice, once for scheduling the job and once for processing the result.
"""

import random
import time

from typing import Any
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field
from colorama import Back, Style

from concurrent_agent_executor import ConcurrentAgentExecutor, BaseParallelizableTool
from concurrent_agent_executor.tools import WaitTool
from concurrent_agent_executor.structured_chat import ConcurrentStructuredChatAgent

load_dotenv()


class RandomNumberToolSchema(BaseModel):
    a: int = Field(
        ...,
        description="The lower bound of the random number generation (inclusive).",
    )
    b: int = Field(
        ...,
        description="The upper bound of the random number generation (inclusive).",
    )


class RandomNumberTool(BaseParallelizableTool):
    is_parallelizable = True

    name = "RandomNumber"
    description = "Schedules a random number generation between a and b; once invoked, you must wait for the result to be ready. Both a and b are integers."
    args_schema: RandomNumberToolSchema = RandomNumberToolSchema

    def _run(
        self,
        a: int,
        b: int,
    ):
        try:
            time.sleep(10)
            n = random.randint(a, b)
            return f"The random number is: {n}"
        except Exception as e:
            return f"Error: {e}"

    def _arun(self, *args: Any, **kwargs: Any):
        return self._run(*args, **kwargs)


def main():
    def on_message(who: str, message: str):
        if who.startswith("tool"):
            print(f"\n{Back.YELLOW}{who}{Style.RESET_ALL}: {message}\n")
        elif who.startswith("error"):
            print(f"\n{Back.RED}{who}{Style.RESET_ALL}: {message}\n")
        else:
            print(f"\n{Back.GREEN}{who}{Style.RESET_ALL}: {message}\n")

    llm = ChatOpenAI(
        temperature=0.3,
        model="gpt-4",
    )

    tools = [
        WaitTool(),
        RandomNumberTool(),
    ]

    chat_history = MessagesPlaceholder(variable_name="chat_history")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = ConcurrentStructuredChatAgent.from_llm_and_tools(
        llm=llm,
        tools=tools,
        memory_prompts=[chat_history],
        input_variables=["input", "agent_scratchpad", "chat_history"],
    )

    executor = ConcurrentAgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory,
        handle_parsing_errors=True,
        early_stopping_method="generate",
    )

    prompt = 'generate a random number between 1 and 2, if it is 1 say "odd", otherwise say "even". Both a and b are integers.'

    with executor:
        executor.emitter.on("message", on_message)
        executor({"input": prompt})


if __name__ == "__main__":
    main()
