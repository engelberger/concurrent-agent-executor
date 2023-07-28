"""
This example defines an intentionally-slow parallelizable tool that generates a random 
number between a and b. Showcases how the agent gets triggered twice, once for scheduling 
the job and once for processing the result.
"""

import random
import time

from typing import Any


from dotenv import load_dotenv

from colorama import Back, Style

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

    def on_message(who: str, type: str, outputs: dict[str, Any]):
        message = outputs["output"]

        if who.startswith("tool"):
            print(f"\n{Back.YELLOW}{who}{Style.RESET_ALL}: {message}\n")
        elif who.startswith("error"):
            print(f"\n{Back.RED}{who}{Style.RESET_ALL}: {message}\n")
        else:
            print(f"\n{Back.GREEN}{who}{Style.RESET_ALL}: {message}\n")

    executor = initialize(
        tools=[RandomNumberTool()],
    )

    prompt = 'generate a random number between 1 and 10, if it is 1 say "odd", otherwise say "even". '

    with executor:
        executor.emitter.on("message", on_message)
        executor({"input": prompt})


if __name__ == "__main__":
    main()
