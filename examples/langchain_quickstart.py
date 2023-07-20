"""
Basic "Getting Started" langchain example.
"""

from dotenv import load_dotenv
from colorama import Back, Style
from langchain.agents import load_tools
from langchain.llms import OpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

from concurrent_agent_executor import (
    ConcurrentAgentExecutor,
    ConcurrentStructuredChatAgent,
    WaitTool,
)

load_dotenv()


def main():
    """Main function."""

    def on_message(who: str, message: str):
        if who.startswith("tool"):
            print(f"\n{Back.YELLOW}{who}{Style.RESET_ALL}: {message}\n")
        elif who.startswith("error"):
            print(f"\n{Back.RED}{who}{Style.RESET_ALL}: {message}\n")
        else:
            print(f"\n{Back.GREEN}{who}{Style.RESET_ALL}: {message}\n")

    # The language model we're going to use to control the agent.
    llm = OpenAI(temperature=0)

    # The tools we'll give the Agent access to. Note that the 'llm-math' tool uses an LLM,
    # so we need to pass that in.
    tools = load_tools(["serpapi", "llm-math"], llm=llm)
    tools.append(WaitTool())

    # Finally, let's initialize an agent with the tools, the language model, and the type
    # of agent we want to use.
    chat_history = MessagesPlaceholder(variable_name="chat_history")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

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
    )

    with executor:
        executor.emitter.on("message", on_message)

        # Let's test it out!
        executor.run(
            (
                "What was the high temperature in SF yesterday in Fahrenheit? "
                "What is that number raised to the .023 power?"
            )
        )


if __name__ == "__main__":
    main()
