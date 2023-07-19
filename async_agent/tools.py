from __future__ import annotations

from typing import Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.base import BaseTool


class BaseParallelizableTool(BaseTool):
    is_parallelizable: bool = False


class WaitTool(BaseParallelizableTool):
    name = "_Wait"
    description = "Wait tool"
    return_direct = True

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        return "Waiting"

    async def _arun(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return self._run(run_manager=run_manager)
