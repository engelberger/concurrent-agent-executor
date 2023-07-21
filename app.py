import asyncio

import chainlit

from chainlit.session import sessions_id
from chainlit.emitter import ChainlitEmitter
from chainlit.context import loop_var, emitter_var

from concurrent_agent_executor import initialize
from examples.slow_random_number import RandomNumberTool

loop = asyncio.get_event_loop()
executor = initialize(tools=[RandomNumberTool()])


@executor.on_message
def _executor_on_message(who: str, message: str):
    loop_var.set(loop)

    session = list(sessions_id.values())[0]
    emitter = ChainlitEmitter(session)
    emitter_var.set(emitter)

    message = chainlit.Message(
        author=who,
        content=message,
        indent=2 if who.startswith("tool") else 0,
    )

    chainlit.run_sync(message.send())


@chainlit.on_chat_start
async def on_chat_start():
    executor.start()


@chainlit.on_message
async def on_message(message: str):
    await executor.acall({"input": message})


@chainlit.on_stop
async def on_stop():
    executor.stop()
