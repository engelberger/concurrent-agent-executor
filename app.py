import asyncio

import chainlit

from chainlit.session import sessions
from chainlit.emitter import ChainlitEmitter
from chainlit.context import loop_var, emitter_var

from concurrent_agent_executor import initialize
from examples.slow_random_number import RandomNumberTool

_LOOP = asyncio.get_event_loop()
executor = initialize(tools=[RandomNumberTool()])

task_list: chainlit.TaskList = None


def _context_hack():
    loop_var.set(_LOOP)
    session = list(sessions.values())[0]
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
def _executor_on_message(who: str, type: str, message: str):
    _context_hack()

    if who == "agent":
        message = chainlit.Message(
            author=who,
            content=message,
        )
    else:
        message = chainlit.Message(
            author=who,
            content=message,
            indent=1,
        )

        if type == "start":
            task = chainlit.Task(
                title=who,
                status=chainlit.TaskStatus.RUNNING,
            )
            chainlit.run_sync(task_list.add_task(task))
        elif type == "finish":
            task: chainlit.Task = find(task_list.tasks, lambda t: t.title == who)
            task.status = chainlit.TaskStatus.DONE
        elif type == "error":
            task: chainlit.Task = find(task_list.tasks, lambda t: t.title == who)
            task.status = chainlit.TaskStatus.FAILED

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
    await executor.acall({"input": message})


@chainlit.on_stop
async def on_stop():
    executor.stop()
