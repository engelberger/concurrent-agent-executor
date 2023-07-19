import asyncio

from promisio import promisify

from textual import on
from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer, Container
from textual.widgets import Header, Footer, Input, Label, Static, Button


class AgentMessage(Container):
    def compose(self) -> ComposeResult:
        yield Label("Agent")
        yield Label("Hello!")


class UserMessage(Container):
    def compose(self) -> ComposeResult:
        yield Label("Hello!")
        yield Label("User")


# class Job(Static):
#     def compose(self) -> ComposeResult:
#         yield Button("Start", id="start", variant="success")
#         yield Label("0", id="time")


class AsyncAgentApp(App):
    def compose(self) -> ComposeResult:
        yield Header()

        yield ScrollableContainer(
            id="body",
        )

        yield Input()

        yield Footer()

    @on(Input.Submitted)
    def handle_input_submitted(self, event: Input.Submitted) -> None:
        @promisify
        async def _():
            await asyncio.sleep(5)
            label = AgentMessage()
            self.query_one("#body").mount(label)
            label.scroll_visible()

        message = UserMessage()
        self.query_one("#body").mount(message)
        message.scroll_visible()
        event.input.value = ""
        _()


if __name__ == "__main__":
    app = AsyncAgentApp()
    app.run()
