"""Agent executor that runs tools in parallel."""

from __future__ import annotations

import inspect
import asyncio
import time

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from threading import Thread, Event
from multiprocessing import Pool
from queue import PriorityQueue, Queue, Empty

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

MessageCallback = Callable[[str, str, dict[str, Any]], None]
"""f(who: str, type: str, outputs: dict[str, Any]) -> None"""


class RunOnceGenerator:
    finished: Event
    queue: Queue[dict[str, Any]]
    executor: ConcurrentAgentExecutor
    inputs: dict[str, str]

    def __init__(
        self,
        executor: ConcurrentAgentExecutor,
        inputs: dict[str, str],
        *,
        start: bool = True,
    ) -> None:
        self.finished = Event()
        self.queue = Queue()
        self.executor = executor
        self.inputs = inputs

        if start:
            self.start()

    def _on_message(self, who: str, type: str, outputs: Dict[str, Any]) -> None:
        if who == "agent" and len(self.executor.running_jobs) == 0:
            self.stop()

        self.queue.put_nowait(outputs)

    def start(self):
        # self.executor.start()
        self.executor.on_message(self._on_message)
        self.executor(self.inputs)

    def stop(self):
        # self.executor.stop()
        self.finished.set()

    def _should_stop(self):
        return (
            self.queue.empty()
            and self.finished.is_set()
            and self.executor.queue.empty()
            and not self.executor.busy
        )

    def __iter__(self) -> "RunOnceGenerator":
        return self

    def __next__(self):
        if self._should_stop():
            raise StopIteration

        return self.queue.get()


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

    pool: Optional[Any]  # Optional[Pool]
    """The process pool."""

    running_jobs: set[str] = Field(
        default_factory=set,
    )
    """The running jobs (id)."""

    def __enter__(self) -> ConcurrentAgentExecutor:
        self.start()
        return self

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
        inputs: Dict[str, str],
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
        # We now enter the agent loop (until it returns something).
        while self._should_continue(iterations, time_elapsed):
            next_step_output = self._take_next_step(
                name_to_tool_map,
                color_mapping,
                inputs,
                intermediate_steps,
                run_manager=run_manager,
                interaction_type=interaction_type,
            )

            if isinstance(next_step_output, AgentFinish):
                return self._return(
                    inputs,
                    next_step_output,
                    intermediate_steps,
                    run_manager=run_manager,
                    interaction_type=interaction_type,
                    who=who,
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
        )

    def arun(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def start(self) -> None:
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

    # NOTE: This is a decorator
    def on_message(self, func: MessageCallback) -> MessageCallback:
        self.emitter.on("message", func)
        return func

    def emit_message(self, who: str, type: str, outputs: dict[str, Any]) -> None:
        self.emitter.emit("message", who, type, outputs)

    def emit_tool_start(
        self,
        tool: BaseParallelizableTool,
        job_id: str,
        input: Union[str, dict],
    ) -> None:
        self.running_jobs.add(job_id)
        self.emit_message(
            f"{tool.name}:{job_id}",
            "start",
            {"output": f"Tool {tool.name} with job_id {job_id} started"},
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
        output: AgentFinish,
        intermediate_steps: list,
        run_manager: Optional[CallbackManagerForChainRun] = None,
        *,
        interaction_type: InteractionType = InteractionType.User,
        who: Optional[str] = None,
    ) -> Dict[str, Any]:
        if run_manager:
            run_manager.on_agent_finish(output, color="green", verbose=self.verbose)

        final_output = output.return_values
        if self.return_intermediate_steps:
            final_output["intermediate_steps"] = intermediate_steps

        match interaction_type:
            case InteractionType.User:
                self.memory.save_context(inputs, final_output)
            case InteractionType.Tool:
                self.memory.chat_memory.add_ai_message(inputs["input"])
                self.memory.chat_memory.add_ai_message(final_output["output"])
            case InteractionType.Agent:
                raise NotImplementedError
            case _:
                raise ValueError(f"Unknown interaction type: {interaction_type}")

        self.emit_message("agent", "message", final_output)

        return final_output

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
        name_to_tool_map: Dict[str, BaseParallelizableTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentActionWithId, str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
        interaction_type: InteractionType = InteractionType.User,
    ) -> Union[AgentFinish, List[Tuple[AgentActionWithId, str]]]:
        try:
            # Call the LLM to see what to do.
            output = self.agent.plan(
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                interaction_type=interaction_type,
                **inputs,
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
            output = AgentActionWithId("_Exception", observation, text)
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
            return [(output, observation)]
        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            return output
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
        return result

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
