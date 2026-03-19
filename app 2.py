import os
import subprocess
from pathlib import Path


def _safe_read_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return ""


def _format_tool_result(tool_name: str, responses_path: str) -> str:
    content = _safe_read_text(responses_path)
    if content:
        return (
            f"{tool_name} finished successfully.\n"
            f"responses_file: {responses_path}\n\n"
            f"responses_content:\n{content}"
        )
    return (
        f"{tool_name} finished successfully.\n"
        f"responses_file: {responses_path}\n"
        f"No readable content found in the responses file."
    )


def compressor_accent_call_run(dirpath: str) -> str:
    print("[STATUS] Starting ACCENT workflow", flush=True)

    scratch_path = os.getenv("SCRATCH")
    if not scratch_path:
        raise RuntimeError("SCRATCH environment variable is not set")

    path_var = os.getenv("PATH", "")
    os.environ["PATH"] = f"/hpc/local/pwc/pw50647/pwc/prod/script:{path_var}"

    folderp = f"{scratch_path}/centrifugal_test_case"
    folder = f"{folderp}/project_data"
    fde_project_folder_path = f"{folder}/project_folder"

    print("[STATUS] Preparing scratch folders", flush=True)
    os.makedirs(folderp, exist_ok=True)
    os.makedirs(folder, exist_ok=True)

    print("[STATUS] Copying project files", flush=True)
    source_dir = dirpath
    destination_dir = folder
    subprocess.run(["cp", "-r", f"{source_dir}/.", destination_dir], check=True)

    print("[STATUS] Moving setup files", flush=True)
    os.chdir(folderp)

    source_file = os.path.join(folder, "initial_factors.txt")
    if os.path.exists(source_file):
        subprocess.run(["mv", source_file, "factors.txt"], check=True)

    source_file = os.path.join(folder, "initial_meta.txt")
    if os.path.exists(source_file):
        subprocess.run(["mv", source_file, "meta.txt"], check=True)

    source_file = os.path.join(folder, "fde_settings.txt")
    if os.path.exists(source_file):
        subprocess.run(["mv", source_file, "fde_settings.txt"], check=True)

    settings_file = Path("fde_settings.txt")
    if settings_file.exists():
        lines = settings_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        new_lines = []
        replaced = False
        for line in lines:
            if line.strip().startswith("project_folder ="):
                new_lines.append(f"project_folder = {fde_project_folder_path}")
                replaced = True
            else:
                new_lines.append(line)
        if not replaced:
            new_lines.append(f"project_folder = {fde_project_folder_path}")
        settings_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    print("[STATUS] Submitting ACCENT job", flush=True)
    print("[STATUS] ACCENT job running on HPC", flush=True)
    command = ["fde_centrifugal.cmd", "call_accent_service", "-a", "a_fluid"]
    subprocess.run(command, check=True)

    responses_filepath = os.path.join(folderp, "responses.txt")
    print("[STATUS] Reading ACCENT response", flush=True)

    if not os.path.exists(responses_filepath):
        Path(responses_filepath).write_text("[DONE]", encoding="utf-8")

    print("[STATUS] ACCENT workflow complete", flush=True)
    return _format_tool_result("accent_service", responses_filepath)


def compressor_caps_call_run(dirpath: str) -> str:
    print("[STATUS] Starting CAPS workflow", flush=True)

    scratch_path = os.getenv("SCRATCH")
    if not scratch_path:
        raise RuntimeError("SCRATCH environment variable is not set")

    path_var = os.getenv("PATH", "")
    os.environ["PATH"] = f"/hpc/local/pwc/pw50647/pwc/prod/script:{path_var}"

    folderp = f"{scratch_path}/centrifugal_test_case"
    os.makedirs(folderp, exist_ok=True)

    print("[STATUS] Preparing CAPS working directory", flush=True)
    os.chdir(folderp)

    print("[STATUS] Submitting CAPS job", flush=True)
    print("[STATUS] CAPS job running on HPC", flush=True)
    command = ["fde_centrifugal.cmd", "call_caps_service", "-a", "a_fluid"]
    subprocess.run(command, check=True)

    responses_filepath = os.path.join(folderp, "responses.txt")
    print("[STATUS] Reading CAPS response", flush=True)

    if not os.path.exists(responses_filepath):
        Path(responses_filepath).write_text("[DONE]", encoding="utf-8")

    print("[STATUS] CAPS workflow complete", flush=True)
    return _format_tool_result("caps_service", responses_filepath)






import os
import json
import asyncio
import queue
import threading
from contextlib import redirect_stdout, redirect_stderr

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent import run_agent_query

app = FastAPI(title="Agentic MDAO Flexible Service")

DEFAULT_JOB_DIR = os.getenv(
    "MDAO_DEFAULT_DIR",
    "/scratch/pwc/pw50647/centrifugal_test_case/project_data"
)


class RunRequest(BaseModel):
    prompt: str
    job_dir: str | None = None


class QueueWriter:
    def __init__(self, q: queue.Queue):
        self.q = q
        self.buf = ""

    def write(self, s: str):
        if not s:
            return
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            line = line.strip()
            if line:
                self.q.put(line)

    def flush(self):
        if self.buf.strip():
            self.q.put(self.buf.strip())
        self.buf = ""


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run")
async def run(req: RunRequest):
    job_dir = req.job_dir or DEFAULT_JOB_DIR

    def generate():
        q: queue.Queue = queue.Queue()
        result = {"answer": "", "error": None}

        def worker():
            writer = QueueWriter(q)
            try:
                with redirect_stdout(writer), redirect_stderr(writer):
                    print("[STATUS] Request received", flush=True)
                    print(f"[STATUS] Using working directory: {job_dir}", flush=True)
                    print("[STATUS] Launching agent", flush=True)

                    answer = asyncio.run(
                        run_agent_query(
                            user_prompt=req.prompt,
                            job_dir=job_dir
                        )
                    )

                    result["answer"] = str(answer)

            except Exception as e:
                result["error"] = str(e)
            finally:
                writer.flush()
                q.put("__STREAM_DONE__")

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        while True:
            item = q.get()
            if item == "__STREAM_DONE__":
                break

            yield json.dumps({
                "type": "progress",
                "message": item
            }) + "\n"

        if result["error"]:
            yield json.dumps({
                "type": "error",
                "status": "failed",
                "job_dir": job_dir,
                "message": result["error"],
                "answer": result["answer"]
            }) + "\n"
        else:
            yield json.dumps({
                "type": "final",
                "status": "success",
                "job_dir": job_dir,
                "answer": result["answer"]
            }) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


import os
import json
import asyncio
import queue
import threading
from contextlib import redirect_stdout, redirect_stderr

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent import run_agent_query

app = FastAPI(title="Agentic MDAO Flexible Service")

DEFAULT_JOB_DIR = os.getenv(
    "MDAO_DEFAULT_DIR",
    "/scratch/pwc/pw50647/centrifugal_test_case/project_data"
)


class RunRequest(BaseModel):
    prompt: str
    job_dir: str | None = None


class QueueWriter:
    def __init__(self, q: queue.Queue):
        self.q = q
        self.buf = ""

    def write(self, s: str):
        if not s:
            return
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            line = line.strip()
            if line:
                self.q.put(line)

    def flush(self):
        if self.buf.strip():
            self.q.put(self.buf.strip())
        self.buf = ""


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run")
async def run(req: RunRequest):
    job_dir = req.job_dir or DEFAULT_JOB_DIR

    def generate():
        q: queue.Queue = queue.Queue()
        result = {"answer": "", "error": None}

        def worker():
            writer = QueueWriter(q)
            try:
                with redirect_stdout(writer), redirect_stderr(writer):
                    print("[STATUS] Request received", flush=True)
                    print(f"[STATUS] Using working directory: {job_dir}", flush=True)
                    print("[STATUS] Launching agent", flush=True)

                    answer = asyncio.run(
                        run_agent_query(
                            user_prompt=req.prompt,
                            job_dir=job_dir
                        )
                    )

                    result["answer"] = str(answer)

            except Exception as e:
                result["error"] = str(e)
            finally:
                writer.flush()
                q.put("__STREAM_DONE__")

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        while True:
            item = q.get()
            if item == "__STREAM_DONE__":
                break

            yield json.dumps({
                "type": "progress",
                "message": item
            }) + "\n"

        if result["error"]:
            yield json.dumps({
                "type": "error",
                "status": "failed",
                "job_dir": job_dir,
                "message": result["error"],
                "answer": result["answer"]
            }) + "\n"
        else:
            yield json.dumps({
                "type": "final",
                "status": "success",
                "job_dir": job_dir,
                "answer": result["answer"]
            }) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")






import os
import json
import requests
import streamlit as st

MDAO_AGENT_STREAM_URL = os.getenv("MDAO_AGENT_STREAM_URL", "http://localhost:8001/run")


def _friendly_status(msg: str) -> str:
    m = msg.lower()

    if "request received" in m or "launching agent" in m:
        return "Starting task"
    if "using working directory" in m or "preparing scratch" in m or "preparing caps working directory" in m:
        return "Preparing workspace"
    if "copying project files" in m:
        return "Copying files"
    if "moving setup files" in m:
        return "Organizing input files"
    if "submitting accent job" in m or "submitting caps job" in m:
        return "Submitting job"
    if "job running" in m:
        return "Job running"
    if "reading accent response" in m or "reading caps response" in m:
        return "Reading results"
    if "workflow complete" in m:
        return "Finalizing answer"

    return "Working"


def run_agentic_mode():
    st.subheader("Agentic MDAO")

    for msg in st.session_state.agent_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Enter instructions for Agentic MDAO...", key="agent_input")
    if not prompt:
        return

    st.session_state.agent_history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        final_answer = ""

        try:
            with st.status("Starting task...", expanded=False) as status:
                with requests.post(
                    MDAO_AGENT_STREAM_URL,
                    json={"prompt": prompt},
                    stream=True,
                    timeout=3600
                ) as resp:
                    resp.raise_for_status()

                    current_label = "Starting task..."

                    for raw in resp.iter_lines(decode_unicode=True):
                        if not raw:
                            continue

                        event = json.loads(raw)
                        event_type = event.get("type", "")

                        if event_type == "progress":
                            msg = event.get("message", "")
                            new_label = _friendly_status(msg) + "..."

                            if new_label != current_label:
                                current_label = new_label
                                status.update(label=current_label, state="running")

                        elif event_type == "final":
                            final_answer = event.get("answer", "Workflow completed.")
                            status.update(label="Completed", state="complete")
                            break

                        elif event_type == "error":
                            err = event.get("message", "Unknown error")
                            partial = event.get("answer", "")
                            final_answer = f"Error: {err}"
                            if partial:
                                final_answer += f"\n\n{partial}"
                            status.update(label="Failed", state="error")
                            break

            if not final_answer:
                final_answer = "Workflow completed."

            st.markdown(final_answer)

        except Exception as e:
            final_answer = f"Error calling HPC agent service: {e}"
            st.error(final_answer)

    st.session_state.agent_history.append({
        "role": "assistant",
        "content": final_answer
    })





MDAO_AGENT_STREAM_URL = "http://localhost:8001/run"

















Change only these two places in rag_core.py.

1) Replace _cosine_top() with this
def _cosine_top(
    question_vec: List[float],
    items: Dict[str, Dict],
    top_n: int,
    min_score: float = 0.705,
) -> List[str]:
    if not items:
        return []

    ids = list(items.keys())
    summaries = [items[i].get("summary", "") for i in ids]
    if not any(summaries):
        return []

    vecs = _embed(summaries)
    sims = [_dot(question_vec, v) / (norm(question_vec) * norm(v) + 1e-9) for v in vecs]

    ranked = sorted(zip(ids, sims), key=lambda x: x[1], reverse=True)

    # only keep strongly relevant media
    return [i for i, s in ranked if s >= min_score][:top_n]
2) In smart_query(), replace this part

Old:

top_img_ids = _cosine_top(q_vec, imgs_all, top_n=1)
top_tbl_ids = _cosine_top(q_vec, tbls_all, top_n=1)

img_item = imgs_all[top_img_ids[0]] if top_img_ids else None
tbl_item = tbls_all[top_tbl_ids[0]] if top_tbl_ids else None

New:

top_img_ids = _cosine_top(q_vec, imgs_all, top_n=1, min_score=0.705)
top_tbl_ids = _cosine_top(q_vec, tbls_all, top_n=1, min_score=0.705)

img_item = imgs_all[top_img_ids[0]] if top_img_ids else None
tbl_item = tbls_all[top_tbl_ids[0]] if top_tbl_ids else None

That is enough to make image/table optional. If nothing is above 0.705, both become None, and your code will show only text.






