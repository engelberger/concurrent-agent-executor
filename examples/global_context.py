import random

from dotenv import load_dotenv

from concurrent_agent_executor import initialize, BaseParallelizableTool

load_dotenv()


class SetRandomVariableTool(BaseParallelizableTool):
    is_parallelizable = True

    name = "random"
    description = "Sets a variable to a random value."

    def _run(self, variable: str):
        value = random.randint(1, 10)
        self.global_context[variable] = value
        return f"Set {variable} to a random value. Use the 'get' tool to get it."


class GetVariableTool(BaseParallelizableTool):
    is_parallelizable = True

    name = "get"
    description = "Gets a variable's value."

    def _run(self, variable: str):
        return str(self.global_context[variable])


def main():
    executor = initialize(
        tools=[
            SetRandomVariableTool(),
            GetVariableTool(),
        ],
    )

    prompt = 'Set the variable "x" to a random number, then get it.'

    try:
        executor.start()
        run = executor.run_once({"input": prompt})
        outputs = run.tail()

        print(outputs)
        print(f"Used {run.intermediate_steps}")
        print(f"Took {run.running_time} seconds")
        print(f"LLM took {run.llm_generation_time} seconds")
    finally:
        executor.stop()


if __name__ == "__main__":
    main()
