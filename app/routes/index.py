"""POST /index — accept a PDF URL, enqueue background ingestion, return job ID.
GET  /jobs/{job_id} — check ingestion status.
"""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.ingest import jobs
from app.ingest.pdf import ingest_from_url

router = APIRouter()


class IndexRequest(BaseModel):
    url: str


class IndexResponse(BaseModel):
    job_id: str
    status: str


class JobResponse(BaseModel):
    job_id: str
    status: str
    source_url: str
    doc_id: str | None
    chunk_count: int | None
    error: str | None


@router.post("/index", response_model=IndexResponse, status_code=202)
async def index_endpoint(body: IndexRequest, background_tasks: BackgroundTasks) -> IndexResponse:
    job = jobs.create_job(source_url=body.url)
    background_tasks.add_task(_run_ingestion, job.id, body.url)
    return IndexResponse(job_id=job.id, status=job.status)


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str) -> JobResponse:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(
        job_id=job.id,
        status=job.status,
        source_url=job.source_url,
        doc_id=job.doc_id,
        chunk_count=job.chunk_count,
        error=job.error,
    )


async def _run_ingestion(job_id: str, url: str) -> None:
    jobs.update_job(job_id, status=jobs.JobStatus.RUNNING)
    try:
        result = await ingest_from_url(url)
        jobs.update_job(
            job_id,
            status=jobs.JobStatus.DONE,
            doc_id=result.doc_id,
            chunk_count=result.chunk_count,
        )
    except Exception as exc:
        jobs.update_job(job_id, status=jobs.JobStatus.FAILED, error=str(exc))
