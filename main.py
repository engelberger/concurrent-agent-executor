import random
import asyncio
from typing import Any, Coroutine

from dotenv import load_dotenv
from promisio import promisify

# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel

from async_agent import AgentExecutor, BaseToolMaybePromise
from async_agent.structured_chat import StructuredChatAgent

load_dotenv()

llm = ChatOpenAI()


@promisify
async def get_random_number(a: int, b: int) -> int:
    asyncio.sleep(5)
    return random.randint(a, b)


class RandomNumberToolSchema(BaseModel):
    a: int
    b: int


class RandomNumberTool(BaseToolMaybePromise):
    name = "random_number"
    description = "Get a random number between a and b"
    # args_schema = RandomNumberToolSchema

    is_promise = True

    def _run(self, a: int, b: int) -> Any:
        return get_random_number(a, b)

    async def _arun(self, a: int, b: int) -> Coroutine[Any, Any, Any]:
        return await self._run(a, b)


tools = [RandomNumberTool()]

agent = StructuredChatAgent.from_llm_and_tools(
    llm=llm,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
)

executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools)

print(executor.run({"input": "Generate a random number between 1 and 10"}))
