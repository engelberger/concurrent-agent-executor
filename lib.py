import asyncio
import aioconsole

from pyee import AsyncIOEventEmitter
from promisio import promisify


class Agent:
    emitter: AsyncIOEventEmitter
    lock: asyncio.Lock

    def __init__(self):
        self.emitter = AsyncIOEventEmitter()
        self.lock = asyncio.Lock()

    def _should_continue():
        pass

    async def _system_reply(self, message):
        async with self.lock:
            self.emitter.emit("message", "system", message)

    def system_test(self):
        """Assumes you have the lock."""

        @promisify
        async def _system_test(self):
            await asyncio.sleep(5)
            await self._system_reply("Test complete!")

        self.emitter.emit("message", "system", "Test started!")
        return _system_test(self)

    async def run(self, message):
        promises = []

        async with self.lock:
            self.emitter.emit("message", "user", message)
            promise = self.system_test()
            promises.append(promise)

        return promises


async def main():
    def on_message(who, message):
        print(f"\n{who}: {message}\n")

    agent = Agent()
    agent.emitter.on("message", on_message)

    promises = []

    while True:
        try:
            message = await aioconsole.ainput('>>> ')
            if message == "exit": break

            promises.extend(await agent.run(message))
        except:
            break

    await asyncio.gather(*promises)


if __name__ == "__main__":
    asyncio.run(main())
