"""Extends the base tool class to provide tools to interact with jobs running in the background."""

from __future__ import annotations

from langchain.tools.base import BaseTool

# pylint: disable=no-name-in-module
from pydantic import BaseModel, Field


class BaseParallelizableTool(BaseTool):
    """Base class for tools that can be run in parallel with other tools."""

    is_parallelizable: bool = Field(
        default=False,
        const=True,
        description="Whether this tool can be run in parallel with other tools.",
    )


class WaitToolSchema(BaseModel):
    """Schema for the WaitTool tool."""

    job_id: str = Field(
        ...,
        description="The job ID to wait for.",
    )


class WaitTool(BaseParallelizableTool):
    """Waits for other tools that run in the background to finish."""

    name = "_Wait"
    description = (
        "This tool waits for other tools that run in the background to finish."
    )
    return_direct = True
    args_schema: WaitToolSchema = WaitToolSchema

    # pylint: disable=arguments-differ
    def _run(
        self,
        job_id: str,
        # run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        return f"Waiting for job {job_id}"

    # pylint: disable=arguments-differ
    async def _arun(
        self,
        job_id: str,
        # run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return self._run(job_id=job_id)


class CancelTool(BaseParallelizableTool):
    """Cancels other tools that run in the background."""

    name = "_Cancel"
    description = "This tool cancels other tools that run in the background."

    # pylint: disable=arguments-differ
    def _run(
        self,
        job_id: str,
        # run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        # ! TODO: Implement cancellation logic
        # return f"Cancelling job {job_id}"
        raise NotImplementedError

    # pylint: disable=arguments-differ
    async def _arun(
        self,
        job_id: str,
        # run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return self._run(job_id=job_id)


class StatusTool(BaseParallelizableTool):
    """Checks the status of other tools that run in the background."""

    name = "_Status"
    description = (
        "This tool checks the status of other tools that run in the background."
    )

    # pylint: disable=arguments-differ
    def _run(
        self,
        job_id: str,
        # run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        # ! TODO: Implement status logic
        # return f"Checking status of job {job_id}"
        raise NotImplementedError

    # pylint: disable=arguments-differ
    async def _arun(
        self,
        job_id: str,
        # run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return self._run(job_id=job_id)
