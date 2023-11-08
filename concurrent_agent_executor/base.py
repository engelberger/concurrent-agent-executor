"""Agent executor that runs tools in parallel."""

from __future__ import annotations
from functools import wraps

import inspect
import asyncio
import time
import os
import logging
from datetime import datetime

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from threading import Thread, Event
from multiprocessing import Pool, Manager
from queue import PriorityQueue, Queue, Empty
from human_id import generate_id

from pyee import AsyncIOEventEmitter
from pydantic import Field

from langchain.agents.tools import InvalidTool, BaseTool
from langchain.agents.agent import (
    ExceptionTool,
    AgentExecutor,
)
from langchain.callbacks.manager import (
    CallbackManager,
    CallbackManagerForChainRun,
    Callbacks,
)
from langchain.input import get_color_mapping
from langchain.schema import (
    AgentFinish,
    OutputParserException,
)
from langchain.memory import ConversationBufferMemory
from langchain.load.dump import dumpd

from concurrent_agent_executor.structured_chat.base import ConcurrentStructuredChatAgent
from concurrent_agent_executor.structured_chat.prompt import START_BACKGROUND_JOB
from concurrent_agent_executor.models import (
    Interaction,
    InteractionType,
    AgentActionWithId,
    BaseParallelizableTool,
    StopMotive,
)
from concurrent_agent_executor.utils import time_it

import asyncio
import time
import logging
from datetime import datetime
from threading import Thread, Event
from multiprocessing import Pool, Manager
from queue import PriorityQueue, Queue, Empty
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from pyee import AsyncIOEventEmitter
from pydantic import Field
from langchain.agents.agent import AgentExecutor, ExceptionTool, InvalidTool, AgentFinish
from langchain.callbacks.manager import CallbackManager, CallbackManagerForChainRun, Callbacks
from langchain.schema import Interaction, InteractionType, AgentActionWithId
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
from concurrent_agent_executor.models import BaseParallelizableTool
from concurrent_agent_executor.structured_chat.base import ConcurrentStructuredChatAgent
import logging
from datetime import datetime

MessageCallback = Callable[[str, str, dict[str, Any]], None]
"""f(who: str, type: str, outputs: dict[str, Any]) -> None"""


class RunOnceGenerator:
    _finished: Event
    _queue: Queue[dict[str, Any]]
    _executor: ConcurrentAgentExecutor
    _inputs: dict[str, str]

    _auto_init: bool
    _start: bool

    _start_time: Optional[float]
    _end_time: Optional[float]

    _intermediate_steps: list[Any]
    _llm_generation_times: list[float]

    def __init__(
        self,
        executor: ConcurrentAgentExecutor,
        inputs: dict[str, str],
        *,
        auto_init: bool = False,
        start: bool = True,
    ) -> None:
        self._finished = Event()
        self._queue = Queue()
        self._executor = executor
        self._inputs = inputs

        self._auto_init = auto_init
        self._start = start

        self._start_time = None
        self._end_time = None

        self._intermediate_steps = []
        self._llm_generation_times = []

        if self._start:
            self.start()

    def _on_message(self, who: str, type: str, outputs: Dict[str, Any]) -> None:
        if "intermediate_steps" in outputs:
            self._intermediate_steps.extend(outputs["intermediate_steps"])

        if "llm_generation_time" in outputs:
            self._llm_generation_times.append(outputs["llm_generation_time"])

        if who == "agent" and len(self._executor.running_jobs) == 0:
            self._end_time = time.perf_counter()
            self.stop()

        self._queue.put_nowait(outputs)

    @property
    def running_time(self) -> float:
        if self._start_time is None:
            return 0
        elif self._end_time is None:
            return time.perf_counter() - self._start_time
        else:
            return self._end_time - self._start_time

    @property
    def intermediate_steps(self) -> list[Any]:
        return self._intermediate_steps

    @property
    def llm_generation_time(self) -> float:
        return sum(self._llm_generation_times)

    @property
    def finished(self) -> bool:
        return self._should_stop()

    def start(self):
        if self._auto_init:
            self._executor.start()
        self._executor.on_message(self._on_message)

        self._start_time = time.perf_counter()
        self._executor(self._inputs)

    def stop(self):
        if self._auto_init:
            self._executor.stop()
        self._finished.set()

    def _should_stop(self):
        return (
            self._queue.empty()
            and self._finished.is_set()
            and self._executor.queue.empty()
            and not self._executor.busy
        )

    def __iter__(self) -> "RunOnceGenerator":
        return self

    def __next__(self):
        if self._should_stop():
            raise StopIteration

        return self._queue.get()


class ConcurrentAgentExecutor(AgentExecutor):
    """Concurrent agent executor runtime."""

    agent: ConcurrentStructuredChatAgent
    """The agent definition."""

    tools: Sequence[Union[BaseParallelizableTool, BaseTool]]
    """The tools the agent can run/invoke."""

    memory: Optional[ConversationBufferMemory] = None
    """The memory of the agent."""

    processes: int = Field(
        default=4,
    )
    """Size of the process pool."""

    emitter: AsyncIOEventEmitter = Field(
        default_factory=lambda: AsyncIOEventEmitter(loop=asyncio.new_event_loop()),
    )
    """The event emitter. Handles post-generation logic."""

    queue: PriorityQueue[Interaction] = Field(
        default_factory=PriorityQueue,
    )
    """The queue of interactions."""

    busy: bool = Field(
        default=False,
    )
    """Whether the agent is busy."""

    finished: Event = Field(
        default_factory=Event,
    )
    """The event that signals the end of the agent."""

    thread: Optional[Thread]
    """The thread that runs the agent."""

    manager: Optional[Any]  # Optional[Manager]
    """The multiprocessing manager."""

    global_context: Optional[Any]  # Optional[dict[str, Any]]
    """The global context, shared across all processes."""

    pool: Optional[Any]  # Optional[Pool]
    """The process pool."""

    running_jobs: set[str] = Field(
        default_factory=set,
    )
    """The running jobs (id)."""

    def __enter__(self) -> ConcurrentAgentExecutor:
        self.start_logging()
        self.start()
        return self

    def start_logging(self) -> None:
        """
        Initialize logging to a file with the current date and time.
        """
        log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_agent_log.txt"
        log_filepath = os.path.join(os.getcwd(), log_filename)

        # Set up logging to the file
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_filepath),
                logging.StreamHandler()
            ]
        )

        # Log the start of the agent
        logging.info("Agent started")
    def stop(self) -> None:
        # Log the stop event before actually stopping the agent
        logging.info("Agent stopped")
        super().stop()

    def _on_message(self, who: str, type: str, outputs: Dict[str, Any]) -> None:
        # Log the message event
        logging.info(f"Message from {who}: {outputs}")
        super()._on_message(who, type, outputs)

    def emit_tool_start(
        self,
        tool: BaseParallelizableTool,
        job_id: str,
        input: Union[str, dict],
    ) -> None:
        # Log the tool start event
        logging.info(f"Tool {tool.name} with job_id {job_id} started")
        super().emit_tool_start(tool, job_id, input)

    def emit_tool_stop(
        self,
        tool: BaseParallelizableTool,
        job_id: str,
        motive: StopMotive,
        output: str,
    ) -> None:
        # Log the tool stop event
        logging.info(f"Tool {tool.name} with job_id {job_id} stopped with motive {motive}")
        super().emit_tool_stop(tool, job_id, motive, output)

    def __exit__(self, *args, **kwargs) -> None:
        self.stop()

    def __call__(
        self,
        inputs: Union[Dict[str, Any], Any],
        callbacks: Callbacks = None,
        *,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        inputs = self.prep_inputs(inputs)
        callback_manager = CallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose,
            tags,
            self.tags,
            metadata,
            self.metadata,
        )
        new_arg_supported = inspect.signature(self._call).parameters.get("run_manager")
        run_manager = callback_manager.on_chain_start(
            # NOTE: This chain is not serializable, since `multiprocessing.Pool` is not
            dumpd(self),
            inputs,
        )
        try:
            if new_arg_supported:
                self._call(inputs, run_manager=run_manager)
            else:
                self._call(inputs)
        except (KeyboardInterrupt, Exception) as e:
            run_manager.on_chain_error(e)
            raise e

    def _main_thread(self):
        while not self.finished.is_set():
            try:
                item = self.queue.get_nowait()
            except Empty:
                continue

            self.busy = True

            self._handle_call(
                item.inputs,
                interaction_type=item.interaction_type,
                who=item.who,
                # run_manager=item.run_manager,
            )

            self.busy = False

    def _handle_call(
        self,
        inputs: dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
        *,
        interaction_type: InteractionType = InteractionType.User,
        who: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}

        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools], excluded_colors=["green", "red"]
        )

        intermediate_steps: List[Tuple[AgentActionWithId, str]] = []

        # Let's start tracking the number of iterations and time elapsed
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()

        llm_generation_time = 0.0

        # We now enter the agent loop (until it returns something).
        while self._should_continue(iterations, time_elapsed):
            llm_generation_time, next_step_output = self._take_next_step(
                inputs,
                name_to_tool_map=name_to_tool_map,
                color_mapping=color_mapping,
                intermediate_steps=intermediate_steps,
                run_manager=run_manager,
                interaction_type=interaction_type,
                llm_generation_time=llm_generation_time,
            )

            if isinstance(next_step_output, AgentFinish):
                return self._return(
                    inputs,
                    next_step_output,
                    intermediate_steps,
                    run_manager=run_manager,
                    interaction_type=interaction_type,
                    who=who,
                    llm_generation_time=llm_generation_time,
                )

            intermediate_steps.extend(next_step_output)
            if len(next_step_output) == 1:
                next_step_action = next_step_output[0]
                # See if tool should return directly
                tool_return = self._get_tool_return(next_step_action)
                if tool_return is not None:
                    return self._return(
                        inputs,
                        tool_return,
                        intermediate_steps,
                        run_manager=run_manager,
                        interaction_type=interaction_type,
                        who=who,
                        llm_generation_time=llm_generation_time,
                    )
            iterations += 1
            time_elapsed = time.time() - start_time
        output = self.agent.return_stopped_response(
            self.early_stopping_method, intermediate_steps, **inputs
        )
        return self._return(
            inputs,
            output,
            intermediate_steps,
            run_manager=run_manager,
            interaction_type=interaction_type,
            who=who,
            llm_generation_time=llm_generation_time,
        )

    def arun(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def start(self) -> None:
        self.manager = Manager()
        self.global_context = self.manager.dict()
        self.pool = Pool(
            processes=self.processes,
        )
        self.thread = Thread(
            target=self._main_thread,
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        self.finished.set()
        self.thread.join()

        self.pool.close()
        self.pool.join()

    def reset(self) -> None:
        pass

    # NOTE: This is a decorator
    def on_message(self, func: MessageCallback) -> MessageCallback:
        @wraps(func)
        def inner(*args, **kwargs):
            return func(*args, **kwargs)

        self.emitter.on("message", inner)
        return inner

    def emit_message(self, who: str, type: str, outputs: dict[str, Any]) -> None:
        self.emitter.emit("message", who, type, outputs)

    def emit_tool_start(
        self,
        tool: BaseParallelizableTool,
        job_id: str,
        input: Union[str, dict],
    ) -> None:
        self.running_jobs.add(job_id)

        outputs = {"output": f"Tool {tool.name} with job_id {job_id} started"}

        self.memory.chat_memory.add_ai_message(outputs["output"])

        self.emit_message(
            f"{tool.name}:{job_id}",
            "start",
            outputs,
        )

    def emit_tool_stop(
        self,
        tool: BaseParallelizableTool,
        job_id: str,
        motive: StopMotive,
        output: str,
    ) -> None:
        self.running_jobs.remove(job_id)
        match motive:
            case StopMotive.Finished:
                self.emit_message(f"{tool.name}:{job_id}", "finish", {"output": output})
            case StopMotive.Error:
                self.emit_message(
                    f"{tool.name}:{job_id}",
                    "error",
                    {
                        "output": f"Tool {tool.name} with job_id {job_id} failed: {output}"
                    },
                )
            case _:
                raise ValueError(f"Unknown stop motive {motive}")

    def _return(
        self,
        inputs: Dict[str, Any],
        agent_finish: AgentFinish,
        intermediate_steps: list,
        run_manager: Optional[CallbackManagerForChainRun] = None,
        *,
        llm_generation_time: float = 0.0,
        interaction_type: InteractionType = InteractionType.User,
        who: Optional[str] = None,
    ) -> Dict[str, Any]:
        if run_manager:
            run_manager.on_agent_finish(
                agent_finish, color="green", verbose=self.verbose
            )

        outputs = agent_finish.return_values

        outputs["llm_generation_time"] = llm_generation_time

        # NOTE: This has to be serializable
        if self.return_intermediate_steps:
            # outputs["intermediate_steps"] = intermediate_steps
            outputs["intermediate_steps"] = []

        match interaction_type:
            case InteractionType.User:
                self.memory.save_context(inputs, outputs)
            case InteractionType.Tool:
                self.memory.chat_memory.add_ai_message(inputs["input"])
                self.memory.chat_memory.add_ai_message(outputs["output"])
            case InteractionType.Agent:
                raise NotImplementedError
            case _:
                raise ValueError(f"Unknown interaction type: {interaction_type}")

        self.emit_message("agent", "message", outputs)

        return outputs

    def _tool_callback(
        self,
        output: str,
        job_id: Optional[str] = None,
        tool: Optional[BaseParallelizableTool] = None,
    ) -> None:
        self.emit_tool_stop(tool, job_id, StopMotive.Finished, output)

        inputs = self.prep_inputs(
            {"input": f"Tool {tool.name} with job_id {job_id} finished: {output}"}
        )

        self.queue.put_nowait(
            Interaction(
                priority=1,
                interaction_type=InteractionType.Tool,
                who=f"{tool.name}:{job_id}",
                inputs=inputs,
                # run_manager=None,
            )
        )

    def _tool_error_callback(
        self,
        exception: Any,
        job_id: Optional[str] = None,
        tool: Optional[BaseParallelizableTool] = None,
    ):
        self.emit_tool_stop(tool, job_id, StopMotive.Error, str(exception))

        inputs = self.prep_inputs(
            {"input": f"Tool {tool.name} with job_id {job_id} failed: {exception}"}
        )

        self.queue.put_nowait(
            Interaction(
                priority=1,
                interaction_type=InteractionType.Tool,
                who=f"{tool.name}:{job_id}",
                inputs=inputs,
                # run_manager=None,
            )
        )

    def _start_tool(
        self,
        tool: BaseParallelizableTool,
        agent_action: AgentActionWithId,
        **tool_run_kwargs,
    ) -> Any:
        context = {
            "job_id": agent_action.job_id,
        }

        self.pool.apply_async(
            tool.invoke,
            args=(
                self.global_context,
                context,
                agent_action.tool_input,
            ),
            kwds=tool_run_kwargs,
            callback=lambda _: self._tool_callback(
                _,
                job_id=agent_action.job_id,
                tool=tool,
            ),
            error_callback=lambda _: self._tool_error_callback(
                _,
                job_id=agent_action.job_id,
                tool=tool,
            ),
        )

        self.emit_tool_start(
            tool,
            agent_action.job_id,
            agent_action.tool_input,
        )

        return START_BACKGROUND_JOB.format(
            tool_name=tool.name, job_id=agent_action.job_id
        )

    def _take_next_step(
        self,
        inputs: dict[str, str],
        *,
        name_to_tool_map: Dict[str, BaseParallelizableTool],
        color_mapping: Dict[str, str],
        intermediate_steps: List[Tuple[AgentActionWithId, str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
        interaction_type: InteractionType = InteractionType.User,
        llm_generation_time: float = 0.0,
    ) -> tuple[float, Union[AgentFinish, List[Tuple[AgentActionWithId, str]]]]:
        try:
            llm_generation_time, output = time_it(
                self.agent.plan,
                args=[
                    intermediate_steps,
                ],
                kwargs={
                    "callbacks": run_manager.get_child() if run_manager else None,
                    "interaction_type": interaction_type,
                    **inputs,
                },
                current_time=llm_generation_time,
            )
        except OutputParserException as exception:
            if isinstance(self.handle_parsing_errors, bool):
                raise_error = not self.handle_parsing_errors
            else:
                raise_error = False
            if raise_error:
                raise exception
            text = str(exception)
            if isinstance(self.handle_parsing_errors, bool):
                if exception.send_to_llm:
                    observation = str(exception.observation)
                    text = str(exception.llm_output)
                else:
                    observation = "Invalid or incomplete response"
            elif isinstance(self.handle_parsing_errors, str):
                observation = self.handle_parsing_errors
            elif callable(self.handle_parsing_errors):
                # pylint: disable=not-callable
                observation = self.handle_parsing_errors(exception)
            else:
                # pylint: disable=raise-missing-from
                raise ValueError("Got unexpected type of `handle_parsing_errors`")
            output = AgentActionWithId("_Exception", observation, text, generate_id())
            if run_manager:
                run_manager.on_agent_action(output, color="green")
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = ExceptionTool().run(
                output.tool_input,
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
            return llm_generation_time, [(output, observation)]
        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            return llm_generation_time, output
        actions: List[AgentActionWithId]
        if isinstance(output, AgentActionWithId):
            actions = [output]
        else:
            actions = output
        result = []
        for agent_action in actions:
            if run_manager:
                run_manager.on_agent_action(agent_action, color="green")
            # Otherwise we lookup the tool
            if agent_action.tool in name_to_tool_map:
                tool = name_to_tool_map[agent_action.tool]
                return_direct = tool.return_direct
                color = color_mapping[agent_action.tool]
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()

                if return_direct:
                    tool_run_kwargs["llm_prefix"] = ""

                if hasattr(tool, "is_parallelizable") and tool.is_parallelizable:
                    observation = self._start_tool(
                        tool,
                        agent_action,
                        # color,
                        # run_manager,
                        **tool_run_kwargs,
                    )
                else:
                    # We then call the tool on the tool input to get an observation
                    observation = tool.run(
                        agent_action.tool_input,
                        verbose=self.verbose,
                        color=color,
                        callbacks=run_manager.get_child() if run_manager else None,
                        **tool_run_kwargs,
                    )
            else:
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                observation = InvalidTool().run(
                    agent_action.tool,
                    verbose=self.verbose,
                    color=None,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
            result.append((agent_action, observation))
        return llm_generation_time, result

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
        *,
        priority: int = 0,
        interaction_type: InteractionType = InteractionType.User,
    ):
        self.queue.put_nowait(
            Interaction(
                priority=priority,
                interaction_type=interaction_type,
                who="user",
                inputs=inputs,
                # run_manager=run_manager,
            )
        )

    def run_once(
        self,
        inputs: Dict[str, str],
    ) -> RunOnceGenerator:
        return RunOnceGenerator(
            executor=self,
            inputs=inputs,
        )
