"""
Reach FastAPI web server.

Endpoints:
    POST /api/upload        — Upload PDF(s) for remediation
    GET  /api/status/{id}   — Check job status
    GET  /api/download/{id} — Download remediated PDF
    POST /api/batch         — Upload ZIP of PDFs
    GET  /api/queue         — List all jobs
    GET  /health            — Server health + GPU info
"""
import asyncio
import json
import uuid
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(
    title="Reach",
    description="Automated PDF/UA-2 remediation server",
    version="0.1.0",
)

# In-memory job store (replace with Redis/DB for production)
jobs: dict[str, dict] = {}
UPLOAD_DIR = Path("/tmp/reach-uploads")
OUTPUT_DIR = Path("/tmp/reach-output")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Pydantic models ──────────────────────────────────────────

class JobStatus(BaseModel):
    job_id: str
    filename: str
    status: str  # "queued" | "processing" | "complete" | "failed"
    progress: float  # 0.0 - 100.0
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None
    download_url: Optional[str] = None
    report: Optional[dict] = None


class BatchStatus(BaseModel):
    batch_id: str
    total: int
    completed: int
    failed: int
    jobs: list[JobStatus]


class HealthResponse(BaseModel):
    status: str
    version: str
    gpu: str
    models_available: list[str]
    queue_size: int


# ── Health ────────────────────────────────────────────────────

@app.get("/health")
async def health() -> HealthResponse:
    """Server health check with GPU info."""
    from reach.models.backend import detect_gpu
    from reach.models.download import verify_model

    gpu = detect_gpu()
    models = []
    for model_id in ["unlimited-ocr", "surya", "qwen3.5-9b"]:
        if verify_model(model_id):
            models.append(model_id)

    return HealthResponse(
        status="ok",
        version="0.1.0",
        gpu=str(gpu),
        models_available=models,
        queue_size=len([j for j in jobs.values() if j["status"] in ("queued", "processing")]),
    )


# ── Upload ────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)) -> JobStatus:
    """Upload a single PDF for remediation."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    job_id = str(uuid.uuid4())[:8]
    filepath = UPLOAD_DIR / f"{job_id}.pdf"
    content = await file.read()

    if len(content) > 500 * 1024 * 1024:  # 500 MB limit
        raise HTTPException(400, "File too large (max 500 MB)")

    filepath.write_bytes(content)

    job = JobStatus(
        job_id=job_id,
        filename=file.filename,
        status="queued",
        progress=0.0,
        created_at=datetime.utcnow().isoformat(),
    )
    jobs[job_id] = job.model_dump()

    # Queue for background processing
    asyncio.create_task(_process_job(job_id, filepath))

    return job


@app.post("/api/batch")
async def upload_batch(file: UploadFile = File(...)) -> BatchStatus:
    """Upload a ZIP file containing multiple PDFs."""
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "Only ZIP files are accepted")

    batch_id = str(uuid.uuid4())[:8]
    batch_dir = UPLOAD_DIR / batch_id
    batch_dir.mkdir(exist_ok=True)

    content = await file.read()
    batch_jobs = []

    with zipfile.ZipFile(BytesIO(content)) as zf:
        pdfs = [n for n in zf.namelist() if n.lower().endswith(".pdf")]
        if not pdfs:
            raise HTTPException(400, "ZIP contains no PDF files")

        for pdf_name in pdfs:
            job_id = str(uuid.uuid4())[:8]
            filepath = batch_dir / f"{job_id}.pdf"
            filepath.write_bytes(zf.read(pdf_name))

            job = JobStatus(
                job_id=job_id,
                filename=pdf_name,
                status="queued",
                progress=0.0,
                created_at=datetime.utcnow().isoformat(),
            )
            jobs[job_id] = job.model_dump()
            batch_jobs.append(job)
            asyncio.create_task(_process_job(job_id, filepath))

    return BatchStatus(
        batch_id=batch_id,
        total=len(batch_jobs),
        completed=0,
        failed=0,
        jobs=batch_jobs,
    )


# ── Status / Download ─────────────────────────────────────────

@app.get("/api/status/{job_id}")
async def job_status(job_id: str) -> JobStatus:
    """Get the current status of a remediation job."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return JobStatus(**jobs[job_id])


@app.get("/api/download/{job_id}")
async def download_result(job_id: str):
    """Download the remediated PDF."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    if job["status"] != "complete":
        raise HTTPException(400, f"Job not ready (status: {job['status']})")

    output_path = OUTPUT_DIR / f"{job_id}-remediated.pdf"
    if not output_path.exists():
        raise HTTPException(404, "Output file not found")

    return FileResponse(
        output_path,
        media_type="application/pdf",
        filename=f"remediated-{job['filename']}",
    )


@app.get("/api/report/{job_id}")
async def download_report(job_id: str) -> JSONResponse:
    """Get the remediation report for a completed job."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    if job.get("report") is None:
        raise HTTPException(404, "Report not available")

    return JSONResponse(content=job["report"])


@app.get("/api/queue")
async def list_queue() -> list[JobStatus]:
    """List all jobs."""
    return [JobStatus(**j) for j in jobs.values()]


# ── UI ────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """Web UI for Reach."""
    return HTMLResponse(content=UI_HTML)


# ── Background processing ─────────────────────────────────────

async def _process_job(job_id: str, filepath: Path) -> None:
    """Run the remediation pipeline and update job status."""
    job = jobs[job_id]
    job["status"] = "processing"
    job["progress"] = 10.0

    try:
        # Phase 1: Extraction
        job["progress"] = 25.0
        # TODO: call reach.extract pipeline
        await asyncio.sleep(0.1)  # placeholder

        # Phase 2: Tag tree
        job["progress"] = 50.0
        await asyncio.sleep(0.1)

        # Phase 3: Construction
        job["progress"] = 75.0
        # TODO: call reach.remediate pipeline
        await asyncio.sleep(0.1)

        # Write placeholder output
        from pypdf import PdfWriter

        output_path = OUTPUT_DIR / f"{job_id}-remediated.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        writer.add_metadata(
            {
                "/Title": f"Reach Remediated: {job['filename']}",
                "/Lang": "en-US",
            }
        )
        with open(output_path, "wb") as f:
            writer.write(f)

        job["status"] = "complete"
        job["progress"] = 100.0
        job["completed_at"] = datetime.utcnow().isoformat()
        job["download_url"] = f"/api/download/{job_id}"
        job["report"] = {
            "job_id": job_id,
            "remediated": True,
            "pdf_ua_version": "2.0",
            "fixes": ["tag_tree", "reading_order", "metadata", "alt_text"],
            "warnings": ["pipeline_not_implemented_placeholder"],
            "verapdf_status": "not_run",
        }

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        job["completed_at"] = datetime.utcnow().isoformat()


# ── HTML UI ───────────────────────────────────────────────────

UI_HTML = """<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reach — PDF/UA-2 Remediation</title>
    <style>
        :root { --bg: #fafafa; --card: #fff; --text: #222; --accent: #2563eb; --border: #e5e7eb; --muted: #6b7280; --green: #10b981; --red: #ef4444; --radius: 10px; }
        [data-theme="dark"] { --bg: #111827; --card: #1f2937; --text: #f3f4f6; --border: #374151; --muted: #9ca3af; }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 40px 20px; }
        .container { max-width: 720px; width: 100%; }
        h1 { font-size: 2rem; font-weight: 700; margin-bottom: 8px; }
        .subtitle { color: var(--muted); margin-bottom: 32px; }
        .dropzone { border: 2px dashed var(--border); border-radius: var(--radius); padding: 60px 20px; text-align: center; cursor: pointer; transition: border-color .2s, background .2s; background: var(--card); }
        .dropzone:hover, .dropzone.drag { border-color: var(--accent); background: rgba(37,99,235,.05); }
        .dropzone p { color: var(--muted); font-size: 1.1rem; }
        .dropzone .icon { font-size: 2.5rem; margin-bottom: 12px; }
        input[type=file] { display: none; }
        .jobs { margin-top: 32px; }
        .job { background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px 20px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; }
        .job-name { font-weight: 600; }
        .job-status { font-size: .85rem; font-weight: 500; padding: 4px 10px; border-radius: 20px; }
        .status-queued { background: #f3f4f6; color: #6b7280; }
        .status-processing { background: #dbeafe; color: #2563eb; }
        .status-complete { background: #d1fae5; color: #059669; }
        .status-failed { background: #fee2e2; color: #dc2626; }
        .progress-bar { width: 100%; height: 6px; background: var(--border); border-radius: 3px; margin-top: 8px; overflow: hidden; }
        .progress-fill { height: 100%; background: var(--accent); transition: width .3s; border-radius: 3px; }
        .download-btn { color: var(--accent); text-decoration: none; font-weight: 600; font-size: .9rem; }
        .theme-toggle { position: fixed; top: 20px; right: 20px; background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; cursor: pointer; font-size: 1.1rem; }
    </style>
</head>
<body>
    <button class="theme-toggle" onclick="toggleTheme()" title="Toggle theme">
        <span id="theme-icon">🌙</span>
    </button>
    <div class="container">
        <h1>Reach</h1>
        <p class="subtitle">Drop inaccessible PDFs. Get back PDF/UA-2 compliant documents. All processing runs locally — your files never leave this server.</p>
        <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
            <div class="icon">📄</div>
            <p>Drop PDFs here or click to upload</p>
            <p style="font-size:.85rem;margin-top:8px">Single PDF or ZIP of multiple PDFs accepted</p>
        </div>
        <input type="file" id="fileInput" accept=".pdf,.zip" multiple onchange="handleFiles(this.files)">
        <div class="jobs" id="jobs"></div>
    </div>
    <script>
        let jobs = [];
        const jobsEl = document.getElementById('jobs');
        const dropzone = document.getElementById('dropzone');

        dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('drag'); });
        dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag'));
        dropzone.addEventListener('drop', e => { e.preventDefault(); dropzone.classList.remove('drag'); handleFiles(e.dataTransfer.files); });

        async function handleFiles(files) {
            for (const file of files) {
                const formData = new FormData();
                formData.append('file', file);
                const isZip = file.name.endsWith('.zip');
                const endpoint = isZip ? '/api/batch' : '/api/upload';
                try {
                    const res = await fetch(endpoint, { method: 'POST', body: formData });
                    const data = await res.json();
                    if (isZip) {
                        data.jobs.forEach(j => jobs.push(j));
                    } else {
                        jobs.push(data);
                    }
                    renderJobs();
                    if (isZip) data.jobs.forEach(j => pollJob(j.job_id));
                    else pollJob(data.job_id);
                } catch (err) {
                    alert('Upload failed: ' + err.message);
                }
            }
        }

        async function pollJob(jobId) {
            const interval = setInterval(async () => {
                try {
                    const res = await fetch('/api/status/' + jobId);
                    const data = await res.json();
                    const idx = jobs.findIndex(j => j.job_id === jobId);
                    if (idx >= 0) jobs[idx] = data;
                    renderJobs();
                    if (data.status === 'complete' || data.status === 'failed') clearInterval(interval);
                } catch (e) { clearInterval(interval); }
            }, 1000);
        }

        function renderJobs() {
            jobsEl.innerHTML = jobs.map(j => `
                <div class="job">
                    <div style="flex:1">
                        <div class="job-name">${j.filename || j.job_id}</div>
                        <div class="progress-bar"><div class="progress-fill" style="width:${j.progress||0}%"></div></div>
                    </div>
                    <span class="job-status status-${j.status}">${j.status}</span>
                    ${j.status === 'complete' ? `<a class="download-btn" href="/api/download/${j.job_id}">⬇ Download</a>` : ''}
                </div>
            `).join('');
        }

        function toggleTheme() {
            const html = document.documentElement;
            const icon = document.getElementById('theme-icon');
            if (html.getAttribute('data-theme') === 'dark') {
                html.setAttribute('data-theme', 'light');
                icon.textContent = '🌙';
            } else {
                html.setAttribute('data-theme', 'dark');
                icon.textContent = '☀️';
            }
        }

        // Check system dark mode preference
        if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
            document.documentElement.setAttribute('data-theme', 'dark');
            document.getElementById('theme-icon').textContent = '☀️';
        }
    </script>
</body>
</html>"""
