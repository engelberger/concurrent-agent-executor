import asyncio

from promisio import promisify

from textual import on
from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer
from textual.widgets import Header, Footer, Input, Label


class AsyncAgentApp(App):
    def compose(self) -> ComposeResult:
        yield Header()

        yield ScrollableContainer(
            id="body",
        )

        yield Input()

        # yield Footer()

    @on(Input.Submitted)
    def handle_input_submitted(self, event: Input.Submitted) -> None:
        @promisify
        async def _():
            await asyncio.sleep(5)
            label = Label("test")
            self.query_one("#body").mount(label)
            label.scroll_visible()

        label = Label(event.value)
        self.query_one("#body").mount(label)
        label.scroll_visible()
        event.input.value = ""
        _()


if __name__ == "__main__":
    app = AsyncAgentApp()
    app.run()
