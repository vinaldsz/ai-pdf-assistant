"""In-memory job store for background ingestion tasks."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import StrEnum


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    status: JobStatus
    source_url: str
    doc_id: str | None = None
    chunk_count: int | None = None
    error: str | None = None


_jobs: dict[str, Job] = {}


def create_job(source_url: str) -> Job:
    job = Job(id=str(uuid.uuid4()), status=JobStatus.QUEUED, source_url=source_url)
    _jobs[job.id] = job
    return job


def get_job(job_id: str) -> Job | None:
    return _jobs.get(job_id)


def update_job(job_id: str, **kwargs: object) -> None:
    job = _jobs.get(job_id)
    if job:
        for k, v in kwargs.items():
            setattr(job, k, v)
