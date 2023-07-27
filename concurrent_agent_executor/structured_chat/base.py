"""Base class for structured chat agents."""

import re
from typing import Any, List, Optional, Sequence, Tuple, Union

from pydantic import Field

from langchain.agents.agent import Agent, AgentOutputParser

from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import AgentAction, AgentFinish
from langchain.callbacks.manager import Callbacks
from concurrent_agent_executor.models import InteractionType

from concurrent_agent_executor.models import BaseParallelizableTool
from concurrent_agent_executor.structured_chat.output_parser import (
    StructuredChatOutputParserWithRetries,
)
from concurrent_agent_executor.structured_chat.prompt import (
    FORMAT_INSTRUCTIONS,
    PREFIX,
    SUFFIX,
)

HUMAN_MESSAGE_TEMPLATE = "{input}\n\n{agent_scratchpad}"

SYSTEM_MESSAGE_TEMPLATE = "{input}\n\n{agent_scratchpad}"


class ConcurrentStructuredChatAgent(Agent):
    """Base class for structured chat agents."""

    system_llm_chain: LLMChain

    output_parser: AgentOutputParser = Field(
        default_factory=StructuredChatOutputParserWithRetries
    )

    @property
    def system_prefix(self) -> str:
        """Prefix to append the system message with.

        This is used when the agent decides to use a tool that returns a promise,
        as the agent will invoke it but the system will run it and respond with
        the result.
        """
        return "System: "

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Observation: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "Thought:"

    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> str:
        agent_scratchpad = super()._construct_scratchpad(intermediate_steps)
        if not isinstance(agent_scratchpad, str):
            raise ValueError("agent_scratchpad should be of type string.")
        if agent_scratchpad:
            return (
                f"This was your previous work "
                f"(but I haven't seen any of it! I only see what "
                f"you return as final answer):\n{agent_scratchpad}"
            )
        else:
            return agent_scratchpad

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseParallelizableTool]) -> None:
        pass

    @classmethod
    def _get_default_output_parser(
        cls, llm: Optional[Any] = None, **kwargs: Any
    ) -> AgentOutputParser:
        return StructuredChatOutputParserWithRetries.from_llm(llm=llm)

    @property
    def _stop(self) -> List[str]:
        return ["Observation:"]

    @staticmethod
    def create_tools_description(tools: Sequence[BaseParallelizableTool]):
        """Create a description of the tools."""

        tool_strings = []
        for tool in tools:
            # ? NOTE: Tools must include schema for args
            args_schema = re.sub("}", "}}}}", re.sub("{", "{{{{", str(tool.args)))

            _ = f"{tool.name}: {tool.description}, args: {args_schema}"

            if hasattr(tool, "is_parallelizable") and tool.is_parallelizable:
                _ += ", this tool runs in the background SO YOU MUST WAIT FOR IT TO FINISH"

            tool_strings.append(_)
        return "\n".join(tool_strings)

    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseParallelizableTool],
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        human_message_template: str = HUMAN_MESSAGE_TEMPLATE,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        input_variables: Optional[List[str]] = None,
        memory_prompts: Optional[List[Any]] = None,
    ) -> Any:
        tool_names = ", ".join([tool.name for tool in tools])
        formatted_tools = cls.create_tools_description(tools)
        format_instructions = format_instructions.format(tool_names=tool_names)
        template = "\n\n".join([prefix, formatted_tools, format_instructions, suffix])
        if input_variables is None:
            input_variables = ["input", "agent_scratchpad"]
        _memory_prompts = memory_prompts or []
        messages = [
            SystemMessagePromptTemplate.from_template(template),
            *_memory_prompts,
            HumanMessagePromptTemplate.from_template(human_message_template),
        ]
        return ChatPromptTemplate(input_variables=input_variables, messages=messages)

    @classmethod
    def create_system_prompt(
        cls,
        tools: Sequence[BaseParallelizableTool],
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        system_message_template: str = SYSTEM_MESSAGE_TEMPLATE,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        input_variables: Optional[List[str]] = None,
        memory_prompts: Optional[List[Any]] = None,
    ) -> Any:
        """Create a prompt for the system to use."""

        formatted_tools = cls.create_tools_description(tools)
        tool_names = ", ".join([tool.name for tool in tools])
        format_instructions = format_instructions.format(tool_names=tool_names)
        template = "\n\n".join([prefix, formatted_tools, format_instructions, suffix])
        if input_variables is None:
            input_variables = ["input", "agent_scratchpad"]
        _memory_prompts = memory_prompts or []
        messages = [
            SystemMessagePromptTemplate.from_template(template),
            *_memory_prompts,
            SystemMessagePromptTemplate.from_template(system_message_template),
        ]
        return ChatPromptTemplate(input_variables=input_variables, messages=messages)

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: Any,
        tools: Sequence[BaseParallelizableTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        output_parser: Optional[AgentOutputParser] = None,
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        human_message_template: str = HUMAN_MESSAGE_TEMPLATE,
        system_message_template: str = SYSTEM_MESSAGE_TEMPLATE,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        input_variables: Optional[List[str]] = None,
        memory_prompts: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> Agent:
        """Construct an agent from an LLM and tools."""
        cls._validate_tools(tools)

        prompt = cls.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix,
            human_message_template=human_message_template,
            format_instructions=format_instructions,
            input_variables=input_variables,
            memory_prompts=memory_prompts,
        )

        system_prompt = cls.create_system_prompt(
            tools,
            prefix=prefix,
            suffix=suffix,
            system_message_template=system_message_template,
            format_instructions=format_instructions,
            input_variables=input_variables,
            memory_prompts=memory_prompts,
        )

        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )

        system_llm_chain = LLMChain(
            llm=llm,
            prompt=system_prompt,
            callback_manager=callback_manager,
        )

        tool_names = [tool.name for tool in tools]
        _output_parser = output_parser or cls._get_default_output_parser(llm=llm)
        return cls(
            llm_chain=llm_chain,
            system_llm_chain=system_llm_chain,
            allowed_tools=tool_names,
            output_parser=_output_parser,
            **kwargs,
        )

    @property
    def _agent_type(self) -> str:
        raise ValueError

    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        interaction_type: InteractionType = InteractionType.User,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        full_inputs = self.get_full_inputs(intermediate_steps, **kwargs)

        match interaction_type:
            case InteractionType.User:
                full_output = self.llm_chain.predict(callbacks=callbacks, **full_inputs)
            case InteractionType.Tool:
                full_output = self.system_llm_chain.predict(
                    callbacks=callbacks, **full_inputs
                )
            case InteractionType.Agent:
                raise NotImplementedError
            case _:
                raise ValueError(f"Unknown interaction type: {interaction_type}")

        return self.output_parser.parse(full_output)
