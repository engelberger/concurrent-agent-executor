"""Extends the base tool class to provide tools to interact with jobs running in the background."""

from __future__ import annotations

from typing import Union

# pylint: disable=no-name-in-module
from pydantic import BaseModel, Field

from concurrent_agent_executor.models import BaseParallelizableTool


class WaitToolSchema(BaseModel):
    """Schema for the WaitTool tool."""

    job_id_or_ids: Union[str, list[str]] = Field(
        ...,
        description="The job IDs to wait for.",
    )


class WaitTool(BaseParallelizableTool):
    """Waits for other tools that run in the background to finish."""

    name = "_Wait"
    description = (
        "This tool waits for other tools that run in the background to finish."
    )
    return_direct = True
    args_schema: WaitToolSchema = WaitToolSchema

    def _run(
        self,
        job_id_or_ids: str,
    ) -> str:
        return f"Waiting for job(s) {job_id_or_ids}"

    async def _arun(
        self,
        job_id_or_ids: str,
    ) -> str:
        return self._run(job_id=job_id_or_ids)


class CancelTool(BaseParallelizableTool):
    """Cancels other tools that run in the background."""

    name = "_Cancel"
    description = "This tool cancels other tools that run in the background."

    def _run(
        self,
        job_id: str,
    ) -> str:
        # ! TODO: Implement cancellation logic
        raise NotImplementedError

    async def _arun(
        self,
        job_id: str,
    ) -> str:
        return self._run(job_id=job_id)


class StatusTool(BaseParallelizableTool):
    """Checks the status of other tools that run in the background."""

    name = "_Status"
    description = (
        "This tool checks the status of other tools that run in the background."
    )

    def _run(
        self,
        job_id: str,
    ) -> str:
        raise NotImplementedError

    async def _arun(
        self,
        job_id: str,
    ) -> str:
        return self._run(job_id=job_id)
