"""Agent executor that runs tools in parallel."""

from __future__ import annotations

import inspect
import asyncio
import time

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from threading import Thread, Event
from multiprocessing import Pool
from queue import PriorityQueue
from dataclasses import dataclass, field

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
    AgentAction,
    AgentFinish,
    OutputParserException,
)
from langchain.memory import ConversationBufferMemory
from langchain.load.dump import dumpd
from human_id import generate_id
from concurrent_agent_executor.models import InteractionType

from concurrent_agent_executor.structured_chat.base import ConcurrentStructuredChatAgent
from concurrent_agent_executor.structured_chat.prompt import START_BACKGROUND_JOB
from concurrent_agent_executor.tools import BaseParallelizableTool


@dataclass(order=True)
class PrioritizedItem:
    priority: int

    interaction_type: InteractionType = field(compare=False)
    who: str = field(compare=False)
    inputs: dict[str, Any] = field(compare=False)


class ConcurrentAgentExecutor(AgentExecutor):
    agent: ConcurrentStructuredChatAgent
    tools: Sequence[Union[BaseParallelizableTool, BaseTool]]
    memory: Optional[ConversationBufferMemory] = None

    processes: int = Field(
        default=4,
    )
    emitter: AsyncIOEventEmitter = Field(
        default_factory=lambda: AsyncIOEventEmitter(loop=asyncio.new_event_loop()),
    )
    queue: PriorityQueue[PrioritizedItem] = Field(
        default_factory=PriorityQueue,
    )
    finished: Event = Field(
        default_factory=Event,
    )

    thread: Optional[Thread]
    pool: Optional[Any]

    def __enter__(self) -> ConcurrentAgentExecutor:
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def __call__(
        self,
        inputs: Union[Dict[str, Any], Any],
        callbacks: Callbacks = None,
        *,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
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
        # run_manager.on_chain_end(outputs)
        # final_outputs: Dict[str, Any] = self.prep_outputs(
        #     inputs, outputs, return_only_outputs
        # )
        # if include_run_info:
        #     final_outputs[RUN_KEY] = RunInfo(run_id=run_manager.run_id)
        # return final_outputs

    def _main_thread(self):
        while True:
            item = self.queue.get()
            if self.finished.is_set():
                break
            self._handle_call(
                item.inputs,
                interaction_type=item.interaction_type,
                who=item.who,
                # run_manager=item.run_manager,
            )

    def _handle_call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
        interaction_type: InteractionType = InteractionType.User,
        who: Optional[str] = None,
    ) -> Dict[str, Any]:
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

    def arun(
        self,
        *args,
        **kwargs,
    ):
        raise NotImplementedError

    def start(self):
        self.pool = Pool(processes=self.processes)
        self.thread = Thread(target=self._main_thread)
        self.thread.start()

    def stop(self):
        self.finished.set()
        self.thread.join()

        self.pool.close()
        self.pool.join()

    # NOTE: This is a decorator
    def on_message(self, func: Callable) -> Callable:
        self.emitter.on("message", func)
        return func

    def _return(
        self,
        inputs: Dict[str, Any],
        output: AgentFinish,
        intermediate_steps: list,
        run_manager: Optional[CallbackManagerForChainRun] = None,
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

        self.emitter.emit("message", "agent", "message", final_output["output"])

        return final_output

    def _tool_callback(
        self,
        output: Any,
        job_id: Optional[str] = None,
        tool: Optional[BaseParallelizableTool] = None,
    ) -> None:
        self.emitter.emit("message", f"{tool.name}:{job_id}", "finish", output)

        inputs = self.prep_inputs(
            {"input": f"Tool {tool.name} with job_id {job_id} finished: {output}"}
        )

        self.queue.put(
            PrioritizedItem(
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
        self.emitter.emit("message", f"{tool.name}:{job_id}", "error", exception)

        inputs = self.prep_inputs(
            {"input": f"Tool {tool.name} with job_id {job_id} failed: {exception}"}
        )

        self.queue.put(
            PrioritizedItem(
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
        agent_action: AgentAction,
        **tool_run_kwargs,
    ) -> Any:
        job_id = generate_id()

        context = {
            "job_id": job_id,
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
                job_id=job_id,
                tool=tool,
            ),
            error_callback=lambda _: self._tool_error_callback(
                _,
                job_id=job_id,
                tool=tool,
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
        interaction_type: InteractionType = InteractionType.User,
    ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
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
        priority: int = 0,
        interaction_type: InteractionType = InteractionType.User,
    ):
        self.queue.put(
            PrioritizedItem(
                priority=priority,
                interaction_type=interaction_type,
                who="user",
                inputs=inputs,
                # run_manager=run_manager,
            )
        )
