import asyncio
import contextvars

import chainlit
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

from concurrent_agent_executor import (
    ConcurrentAgentExecutor,
    ConcurrentStructuredChatAgent,
    WaitTool,
)
from examples.slow_random_number import RandomNumberTool

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
    memory_key="chat_history",
    return_messages=True,
    output_key="output",
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

executor.start()


@chainlit.on_message
async def on_user_message(message: str):
    @executor.on_message
    def on_message(who: str, message: str):
        async def _on_message(who, message):
            await chainlit.Message(author=who, content=message).send()

        # asyncio.create_task(_on_message(who, message))
        # asyncio.ensure_future(_on_message(who, message))
        # ^ RuntimeError: There is no current event loop in thread 'Thread-3 (_handle_results)'.

        # loop = asyncio.new_event_loop()
        # loop.run_until_complete(_on_message(who, message))
        # loop.close()

        print("pre")
        asyncio.run(_on_message(who, message))
        print("post")

    await executor.acall({"input": message})


@chainlit.on_stop
async def on_stop():
    executor.stop()
