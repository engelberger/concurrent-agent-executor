from __future__ import annotations

from typing import Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.base import BaseTool
from pydantic import BaseModel, Field


class BaseParallelizableTool(BaseTool):
    is_parallelizable: bool = Field(
        default=False,
        const=True,
        description="Whether this tool can be run in parallel with other tools.",
    )


class WaitToolSchema(BaseModel):
    job_id: str = Field(
        ...,
        description="The job ID to wait for.",
    )


class WaitTool(BaseParallelizableTool):
    name = "_Wait"
    description = (
        "This tool waits for other tools that run in the background to finish."
    )
    return_direct = True
    args_schema: WaitToolSchema = WaitToolSchema

    def _run(
        self,
        job_id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        return f"Waiting for job {job_id}"

    async def _arun(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return self._run(run_manager=run_manager)


class CancelTool(BaseParallelizableTool):
    name = "_Cancel"
    description = "This tool cancels other tools that run in the background."

    def _run(
        self,
        job_id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        # ! TODO: Implement cancellation logic
        # return f"Cancelling job {job_id}"
        raise NotImplementedError

    async def _arun(
        self,
        job_id: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return self._run(job_id=job_id, run_manager=run_manager)


class StatusTool(BaseParallelizableTool):
    name = "_Status"
    description = (
        "This tool checks the status of other tools that run in the background."
    )

    def _run(
        self,
        job_id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        # ! TODO: Implement status logic
        return f"Checking status of job {job_id}"

    async def _arun(
        self,
        job_id: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return self._run(job_id=job_id, run_manager=run_manager)
