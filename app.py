import asyncio
from typing import Any, Optional

import chainlit

from chainlit.session import sessions_id
from chainlit.emitter import ChainlitEmitter
from chainlit.context import loop_var, emitter_var
from langchain.agents import load_tools
from langchain.llms import OpenAI

from concurrent_agent_executor import initialize
from examples.slow_random_number import RandomNumberTool

_LOOP = asyncio.get_event_loop()
executor = initialize(
    tools=[RandomNumberTool(), *load_tools(["llm-math"], llm=OpenAI(temperature=0))],
    processes=4,
    model="gpt-4",
)

task_list: Optional[chainlit.TaskList] = None


def _context_hack():
    loop_var.set(_LOOP)
    session = list(sessions_id.values())[0]
    emitter = ChainlitEmitter(session)
    emitter_var.set(emitter)


def find(
    iterable,
    predicate,
    default=None,
):
    for item in iterable:
        if predicate(item):
            return item
    return default


@executor.on_message
def _executor_on_message(who: str, type: str, outputs: dict[str, Any]):
    _context_hack()

    message = outputs["output"]

    if who == "agent":
        message = chainlit.Message(
            author=who,
            content=message,
        )
    else:
        if type == "start":
            message = chainlit.Message(
                author=who,
                content=message,
                indent=1,
            )

            task = chainlit.Task(
                title=who,
                status=chainlit.TaskStatus.RUNNING,
            )
            chainlit.run_sync(task_list.add_task(task))
        else:
            message = chainlit.Message(
                author=who,
                content=message,
            )

            if type == "finish":
                task: chainlit.Task = find(task_list.tasks, lambda t: t.title == who)
                task.status = chainlit.TaskStatus.DONE
            elif type == "error":
                task: chainlit.Task = find(task_list.tasks, lambda t: t.title == who)
                task.status = chainlit.TaskStatus.FAILED
            else:
                raise ValueError(f"Unknown type: {type}")

        chainlit.run_sync(task_list.send())

    chainlit.run_sync(message.send())


@chainlit.on_chat_start
async def on_chat_start():
    global task_list

    executor.start()

    task_list = chainlit.TaskList()
    await task_list.send()


@chainlit.on_message
async def on_message(message: str):
    executor({"input": message})


@chainlit.on_stop
async def on_stop():
    executor.stop()
