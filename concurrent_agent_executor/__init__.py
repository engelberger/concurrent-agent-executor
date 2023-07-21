"""
An concurrent runtime for tool-enhanced language agents.
"""

from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

from concurrent_agent_executor.base import ConcurrentAgentExecutor
from concurrent_agent_executor.tools import BaseParallelizableTool, WaitTool
from concurrent_agent_executor.structured_chat.base import ConcurrentStructuredChatAgent

__all__ = [
    "ConcurrentAgentExecutor",
    "BaseParallelizableTool",
    "WaitTool",
    "ConcurrentStructuredChatAgent",
    "initialize",
]


def initialize(
    llm=None,
    tools=None,
    **kwargs,
):
    """
    Initialize the concurrent_agent_executor module.
    """

    if llm is None:
        llm = ChatOpenAI(
            temperature=0.3,
            model="gpt-4",
        )

    if tools is None:
        tools = []

    tools = [
        WaitTool(),
        *tools,
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
        **kwargs,
    )

    return executor
