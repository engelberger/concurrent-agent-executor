"""Agent executor that runs tools in parallel."""

from __future__ import annotations

import asyncio
import time

from threading import Thread, Event
from multiprocessing import Pool
from queue import PriorityQueue
from pyee import AsyncIOEventEmitter

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from langchain.agents.tools import InvalidTool, BaseTool
from langchain.agents.agent import (
    ExceptionTool,
    AgentExecutor,
)
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.input import get_color_mapping
from langchain.schema import (
    AgentAction,
    AgentFinish,
    OutputParserException,
)
from langchain.utilities.asyncio import asyncio_timeout
from human_id import generate_id

from concurrent_agent_executor.structured_chat.base import ConcurrentStructuredChatAgent
from concurrent_agent_executor.structured_chat.prompt import START_BACKGROUND_JOB
from concurrent_agent_executor.tools import BaseParallelizableTool


class ConcurrentAgentExecutor(AgentExecutor):
    """Consists of an async agent using tools."""

    agent: ConcurrentStructuredChatAgent
    """The agent to run for creating a plan and determining actions
    to take at each step of the execution loop."""
    tools: Sequence[Union[BaseParallelizableTool, BaseTool]]
    """The valid tools the agent can call."""

    processes: Optional[int]

    emitter: Any  # emitter: AsyncIOEventEmitter

    finished: Any  # finished: Event

    thread: Any  # Optional[Thread]
    queue: Any  # Optional[PriorityQueue]
    pool: Any  # pool: Pool

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.emitter = AsyncIOEventEmitter(loop=asyncio.new_event_loop())

    def __enter__(self) -> ConcurrentAgentExecutor:
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()

    def _event_loop(self) -> None:
        while True:
            _, who, inputs, run_manager = self.queue.get()
            if self.finished.is_set():
                break
            self.__call(inputs, who=who, run_manager=run_manager)

    def start(self) -> None:
        self.finished = Event()

        self.queue = PriorityQueue()

        self.pool = Pool(processes=self.processes)

        self.thread = Thread(target=self._event_loop)
        self.thread.start()

    def stop(self) -> None:
        self.finished.set()
        self.thread.join()

        self.pool.close()
        self.pool.join()

    def on_message(self, func: Callable) -> Callable:
        self.emitter.on("message", func)
        return func

    def _return(
        self,
        output: AgentFinish,
        intermediate_steps: list,
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        if run_manager:
            run_manager.on_agent_finish(output, color="green", verbose=self.verbose)

        final_output = output.return_values
        if self.return_intermediate_steps:
            final_output["intermediate_steps"] = intermediate_steps

        self.emitter.emit("message", "agent", "message", final_output["output"])

        return final_output

    def _tool_callback(
        self,
        output: Any,
        job_id: Optional[str] = None,
        tool: Optional[BaseParallelizableTool] = None,
        # agent_action: Optional[AgentAction] = None,
    ) -> None:
        self.emitter.emit("message", f"{tool.name}:{job_id}", "finish", output)

        inputs = self.prep_inputs(
            {"input": f"Tool {tool.name} with job_id {job_id} finished: {output}"}
        )

        self.queue.put((1, f"{tool.name}:{job_id}", inputs, None))

    def _tool_error_callback(
        self,
        exception: Any,
        job_id: Optional[str] = None,
        tool: Optional[BaseParallelizableTool] = None,
        # agent_action: Optional[AgentAction] = None,
    ):
        self.emitter.emit("message", f"{tool.name}:{job_id}", "error", exception)

        inputs = self.prep_inputs(
            {"input": f"Tool {tool.name} with job_id {job_id} failed: {exception}"}
        )

        self.queue.put((1, f"{tool.name}:{job_id}", inputs, None))

    def _start_parallelizable_tool(
        self,
        tool: BaseParallelizableTool,
        agent_action: AgentAction,
        # color: Optional[str] = None,
        # run_manager: Optional[CallbackManagerForChainRun] = None,
        **tool_run_kwargs,
    ) -> Any:
        # ! TODO: This does not provide a way of having tracing with callbacks
        job_id = generate_id()

        self.pool.apply_async(
            tool.run,
            args=(agent_action.tool_input,),
            kwds={
                "job_id": job_id,
                **tool_run_kwargs,
            },
            callback=lambda _: self._tool_callback(
                _,
                job_id=job_id,
                tool=tool,
                # agent_action=agent_action,
            ),
            error_callback=lambda _: self._tool_error_callback(
                _,
                job_id=job_id,
                tool=tool,
                # agent_action=agent_action,
            ),
        )

        self.emitter.emit("message", f"{tool.name}:{job_id}", "start", "started")

        return START_BACKGROUND_JOB.format(tool_name=tool.name, job_id=job_id)

    def _take_next_step(
        self,
        name_to_tool_map: Dict[str, BaseParallelizableTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
        who: str = "agent",
    ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
        """Take a single step in the thought-action-observation loop.

        Override this to take control of how the agent makes and acts on choices.
        """
        try:
            # Call the LLM to see what to do.
            output = self.agent.plan(
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                who=who,
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
            output = AgentAction("_Exception", observation, text)
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
        actions: List[AgentAction]
        if isinstance(output, AgentAction):
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
                    observation = self._start_parallelizable_tool(
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

    def __call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
        who: str = "agent",
    ) -> Dict[str, Any]:
        """Run text through and get agent response."""

        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools], excluded_colors=["green", "red"]
        )
        intermediate_steps: List[Tuple[AgentAction, str]] = []
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
                who=who,
            )

            if isinstance(next_step_output, AgentFinish):
                return self._return(
                    next_step_output, intermediate_steps, run_manager=run_manager
                )

            intermediate_steps.extend(next_step_output)
            if len(next_step_output) == 1:
                next_step_action = next_step_output[0]
                # See if tool should return directly
                tool_return = self._get_tool_return(next_step_action)
                if tool_return is not None:
                    return self._return(
                        tool_return, intermediate_steps, run_manager=run_manager
                    )
            iterations += 1
            time_elapsed = time.time() - start_time
        output = self.agent.return_stopped_response(
            self.early_stopping_method, intermediate_steps, **inputs
        )
        return self._return(output, intermediate_steps, run_manager=run_manager)

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
        priority: int = 0,
        who: str = "agent",
    ) -> Dict[str, Any]:
        """Run text through and get agent response."""

        self.queue.put((priority, who, inputs, run_manager))

    async def _acall(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        """Run text through and get agent response."""
        raise NotImplementedError
