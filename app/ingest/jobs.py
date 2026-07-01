"""In-memory job store for background ingestion tasks."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum

_MAX_JOBS = 10_000  # hard cap — prevents OOM on the 512 MB Fly.io VM


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
    _created_order: int = field(default=0, repr=False, compare=False)


_jobs: dict[str, Job] = {}
_job_counter: int = 0


def _evict_completed() -> None:
    """Remove the oldest DONE/FAILED jobs until we're under the cap."""
    terminal = [j for j in _jobs.values() if j.status in (JobStatus.DONE, JobStatus.FAILED)]
    terminal.sort(key=lambda j: j._created_order)
    for j in terminal[: len(_jobs) - _MAX_JOBS + 1]:
        _jobs.pop(j.id, None)


def create_job(source_url: str) -> Job:
    global _job_counter
    if len(_jobs) >= _MAX_JOBS:
        _evict_completed()
    _job_counter += 1
    job = Job(
        id=str(uuid.uuid4()),
        status=JobStatus.QUEUED,
        source_url=source_url,
        _created_order=_job_counter,
    )
    _jobs[job.id] = job
    return job


def get_job(job_id: str) -> Job | None:
    return _jobs.get(job_id)


def update_job(job_id: str, **kwargs: object) -> None:
    job = _jobs.get(job_id)
    if job:
        for k, v in kwargs.items():
            setattr(job, k, v)
