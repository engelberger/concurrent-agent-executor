import random
import time

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
            time.sleep(10)
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
        generator = executor.run_once({"input": prompt})
        for outputs in generator:
            print(outputs["output"], end="\n\n")
    finally:
        executor.stop()


if __name__ == "__main__":
    main()
