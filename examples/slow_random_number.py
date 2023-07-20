"""
This example defines an intentionally-slow parallelizable tool that generates a random 
number between a and b. Showcases how the agent gets triggered twice, once for scheduling 
the job and once for processing the result.
"""

import random
import time

from typing import Any


from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

# pylint: disable=no-name-in-module
from pydantic import BaseModel, Field
from colorama import Back, Style

from concurrent_agent_executor import ConcurrentAgentExecutor, BaseParallelizableTool
from concurrent_agent_executor.tools import WaitTool
from concurrent_agent_executor.structured_chat import ConcurrentStructuredChatAgent

load_dotenv()


class RandomNumberToolSchema(BaseModel):
    """Schema for the RandomNumberTool tool."""

    a: int = Field(
        ...,
        description="The lower bound of the random number generation (inclusive).",
    )
    b: int = Field(
        ...,
        description="The upper bound of the random number generation (inclusive).",
    )


class RandomNumberTool(BaseParallelizableTool):
    """
    Schedules a random number generation between a and b; once invoked, you must wait
    for the result to be ready. Both a and b are integers.
    """

    is_parallelizable = True

    name = "RandomNumber"
    description = (
        "Schedules a random number generation between a and b; once invoked, you"
        "must wait for the result to be ready. Both a and b are integers."
    )
    args_schema: RandomNumberToolSchema = RandomNumberToolSchema

    # pylint: disable=arguments-differ
    def _run(
        self,
        a: int,
        b: int,
    ):
        try:
            time.sleep(10)
            return f"The random number is: {random.randint(a, b)}"
        # pylint: disable=broad-except
        except Exception as exception:
            return f"Error: {exception}"

    async def _arun(self, *args: Any, **kwargs: Any):
        return self._run(*args, **kwargs)


def main():
    """Main function for the example."""

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
    memory = ConversationBufferMemory(
        memory_key="chat_history", output_key="output", return_messages=True
    )

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
        return_intermediate_steps=True,
    )

    prompt = (
        'generate a random number between 1 and 2, if it is 1 say "odd", otherwise say "even". '
        "Both a and b are integers."
    )

    with executor:
        executor.emitter.on("message", on_message)
        result = executor({"input": prompt})
        print(f"{result=}")


if __name__ == "__main__":
    main()
