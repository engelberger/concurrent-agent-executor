import random

from typing import Any

from dotenv import load_dotenv

from concurrent_agent_executor import initialize, BaseParallelizableTool

load_dotenv()


class RandomNumberTool(BaseParallelizableTool):
    """
    Schedules a random number generation between a and b; once invoked, you must wait
    for the result to be ready. Both a and b are integers.
    """

    is_parallelizable = True

    name = "RandomNumber"
    description = (
        "Schedules a random number generation between a and b; once invoked, you"
        "must wait for the result to be ready. Both a and b are integers."
    )

    # pylint: disable=arguments-differ
    def _run(
        self,
        a: int,
        b: int,
    ):
        try:
            # time.sleep(5)
            return f"The random number is: {random.randint(a, b)}"
        # pylint: disable=broad-except
        except Exception as exception:
            return f"Error: {exception}"

    async def _arun(self, *args: Any, **kwargs: Any):
        return self._run(*args, **kwargs)


def main():
    """Main function for the example."""

    executor = initialize(
        tools=[RandomNumberTool()],
    )

    prompt = 'generate a random number between 1 and 10, if it is 1 say "odd", otherwise say "even". '

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
