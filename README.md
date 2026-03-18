# Malaria_classification
Classification of Infected and Non-infected classes of Malaria from Cell images of humans

Dataset link:- https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
Go to the HPC terminal and run these in the same environment where your current agent.py already works.

1) Go to repo
cd /path/to/agentic-mdao-hackathon
2) Activate your environment

Use whatever you already use. Example:

source .venv/bin/activate

or conda:

conda activate your_env_name
3) Install API packages if missing
pip install fastapi uvicorn requests
4) Export environment variables

Adjust paths for your HPC.

export PYTHONPATH=$PWD:$PYTHONPATH
export LLM_MODEL=$PWD/examples/config_granite_example.json
export SCRATCH=/scratch/pwc/pw50647
export MDAO_DEFAULT_DIR=/scratch/pwc/pw50647/centrifugal_test_case/project_data

If your cluster needs extra paths, also export them here.

First test before any API

Before starting the API, make sure the agent itself works on HPC.

python turbo/compressor/agent.py "I have files in the project directory. Run the compressor analysis and summarize the result."

If this works, the API wrapping will also work.

COMMON FILES FOR BOTH VERSIONS

These two files are shared by both Version 1 and Version 2.

turbo/compressor/functions.py
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


def describe_directory(dirpath: str) -> str:
    p = Path(dirpath)
    if not p.exists():
        return f"Directory does not exist: {dirpath}"

    lines = []
    for item in sorted(p.iterdir()):
        if item.is_file():
            lines.append(f"FILE: {item.name}")
        elif item.is_dir():
            lines.append(f"DIR:  {item.name}")

    if not lines:
        return f"Directory is empty: {dirpath}"

    return f"Directory inventory for {dirpath}:\n" + "\n".join(lines)


def compressor_accent_call_run(dirpath: str) -> str:
    scratch_path = os.getenv("SCRATCH")
    if not scratch_path:
        raise RuntimeError("SCRATCH environment variable is not set")

    path_var = os.getenv("PATH", "")
    os.environ["PATH"] = f"/hpc/local/pwc/pw50647/pwc/prod/script:{path_var}"

    folderp = f"{scratch_path}/centrifugal_test_case"
    os.makedirs(folderp, exist_ok=True)

    folder = f"{folderp}/project_data"
    os.makedirs(folder, exist_ok=True)

    fde_project_folder_path = f"{folder}/project_folder"

    source_dir = dirpath
    destination_dir = f"{scratch_path}/centrifugal_test_case/project_data"
    subprocess.run(["cp", "-r", f"{source_dir}/.", destination_dir], check=True)

    os.chdir(folderp)

    source_file = os.path.join(folder, "initial_factors.txt")
    destination_file = "factors.txt"
    if os.path.exists(source_file):
        subprocess.run(["mv", source_file, destination_file], check=True)

    source_file = os.path.join(folder, "initial_meta.txt")
    destination_file = "meta.txt"
    if os.path.exists(source_file):
        subprocess.run(["mv", source_file, destination_file], check=True)

    source_file = os.path.join(folder, "fde_settings.txt")
    destination_file = "fde_settings.txt"
    if os.path.exists(source_file):
        subprocess.run(["mv", source_file, destination_file], check=True)

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

    command = ["fde_centrifugal.cmd", "call_accent_service", "-a", "a_fluid"]
    subprocess.run(command, check=True)

    responses_filepath = os.path.join(folderp, "responses.txt")
    if not os.path.exists(responses_filepath):
        Path(responses_filepath).write_text("[DONE]", encoding="utf-8")

    return _format_tool_result("accent_service", responses_filepath)


def compressor_caps_call_run(dirpath: str) -> str:
    scratch_path = os.getenv("SCRATCH")
    if not scratch_path:
        raise RuntimeError("SCRATCH environment variable is not set")

    path_var = os.getenv("PATH", "")
    os.environ["PATH"] = f"/hpc/local/pwc/pw50647/pwc/prod/script:{path_var}"

    folderp = f"{scratch_path}/centrifugal_test_case"
    os.makedirs(folderp, exist_ok=True)

    os.chdir(folderp)

    command = ["fde_centrifugal.cmd", "call_caps_service", "-a", "a_fluid"]
    subprocess.run(command, check=True)

    responses_filepath = os.path.join(folderp, "responses.txt")
    if not os.path.exists(responses_filepath):
        Path(responses_filepath).write_text("[DONE]", encoding="utf-8")

    return _format_tool_result("caps_service", responses_filepath)


def compressor_auto_run(dirpath: str) -> str:
    accent_result = compressor_accent_call_run(dirpath)
    caps_result = compressor_caps_call_run(dirpath)

    return (
        "Auto workflow completed.\n\n"
        "Step 1: Accent\n"
        f"{accent_result}\n\n"
        "Step 2: Caps\n"
        f"{caps_result}"
    )
turbo/compressor/agent.py
import json
import sys
import os
import asyncio
from pathlib import Path

from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import FunctionTool
from llm_pwc.llm import load_from_config

from functions import (
    compressor_accent_call_run,
    compressor_caps_call_run,
    compressor_auto_run,
    describe_directory,
)

llm_model_path = os.getenv("LLM_MODEL")
if not llm_model_path:
    raise RuntimeError("LLM_MODEL environment variable is not set")

config = json.loads(Path(llm_model_path).read_text(encoding="utf-8"))
llm = load_from_config(config["llm"])
local_llm = llm.llama_index_llm()

tools = [
    FunctionTool.from_defaults(compressor_accent_call_run),
    FunctionTool.from_defaults(compressor_caps_call_run),
    FunctionTool.from_defaults(compressor_auto_run),
    FunctionTool.from_defaults(describe_directory),
]

SYSTEM_PROMPT = """
You are an AI assistant for compressor workflow execution.

Rules:
1. The user may not mention tool names explicitly.
2. Infer the correct workflow from the request and the directory context.
3. Use describe_directory first if the directory contents are unclear.
4. If the user asks to run the compressor workflow from a prepared project directory,
   you may use the automatic workflow tool.
5. Always summarize what you did and include the important result content if available.
6. Do not ask the user which internal tool name to use.
"""

agent = ReActAgent(
    tools=tools,
    llm=local_llm,
    verbose=True,
    system_prompt=SYSTEM_PROMPT,
)


def build_wrapped_prompt(user_prompt: str, job_dir: str | None = None) -> str:
    parts = []

    if job_dir:
        parts.append(
            f"The working directory for this task is:\n{job_dir}\n"
            "The required files are expected to already exist in or under this directory.\n"
            "Do not ask the user for tool names. Infer the correct action yourself."
        )

    parts.append(f"User request:\n{user_prompt}")
    return "\n\n".join(parts)


async def run_agent_query(user_prompt: str, job_dir: str | None = None) -> str:
    prompt = build_wrapped_prompt(user_prompt, job_dir)
    response = await agent.run(prompt)
    return str(response)


async def main(queries: list[str]):
    job_dir = os.getenv("MDAO_JOB_DIR")
    for q in queries:
        print(f"Running query: {q}")
        response = await run_agent_query(q, job_dir=job_dir)
        print(response)


if __name__ == "__main__":
    queries = sys.argv[1:]
    if not queries:
        print("Please provide at least one query as an argument.")
    else:
        asyncio.run(main(queries))
VERSION 1 — SIMPLE / MORE HARDCODED

This is the first version I recommend you start with.

Idea

one default HPC directory

frontend sends only the plain prompt

backend silently injects the directory

user does not mention tool names

turbo/compressor/agent_service_simple.py
import os
from fastapi import FastAPI
from pydantic import BaseModel

from agent import run_agent_query

app = FastAPI(title="Agentic MDAO Simple Service")

DEFAULT_JOB_DIR = os.getenv(
    "MDAO_DEFAULT_DIR",
    "/scratch/pwc/pw50647/centrifugal_test_case/project_data"
)


class RunRequest(BaseModel):
    prompt: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run")
async def run(req: RunRequest):
    answer = await run_agent_query(
        user_prompt=req.prompt,
        job_dir=DEFAULT_JOB_DIR
    )
    return {
        "status": "success",
        "job_dir": DEFAULT_JOB_DIR,
        "answer": answer
    }
Commands to run Version 1 on HPC
Start the service

Run from repo root:

cd /path/to/agentic-mdao-hackathon
source .venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH
export LLM_MODEL=$PWD/examples/config_granite_example.json
export SCRATCH=/scratch/pwc/pw50647
export MDAO_DEFAULT_DIR=/scratch/pwc/pw50647/centrifugal_test_case/project_data

uvicorn turbo.compressor.agent_service_simple:app --host 0.0.0.0 --port 8001
Health check on HPC

Open another terminal on HPC:

curl http://localhost:8001/health

Expected:

{"status":"ok"}
Test the API on HPC
curl -X POST http://localhost:8001/run \
  -H "Content-Type: application/json" \
  -d '{"prompt":"I have files in the project directory. Run the compressor analysis and summarize the result."}'
If frontend is on your laptop: create SSH tunnel

Run this on your laptop terminal:

ssh -L 8001:localhost:8001 your_hpc_username@your_hpc_host

Then on your laptop, the service will be reachable as:

http://localhost:8001/run
Streamlit frontend code for Version 1

Add requests import near the top of your Application/app.py:

import requests

Set the URL:

MDAO_AGENT_URL = os.getenv("MDAO_AGENT_URL", "http://localhost:8001/run")

Replace your run_agentic_mode() with this:

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
        placeholder = st.empty()
        try:
            placeholder.markdown("Running on HPC...")
            resp = requests.post(
                MDAO_AGENT_URL,
                json={"prompt": prompt},
                timeout=600
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data.get("answer", "No response returned.")
            placeholder.markdown(answer)
        except Exception as e:
            answer = f"Error calling HPC agent service: {e}"
            placeholder.markdown(answer)

    st.session_state.agent_history.append({"role": "assistant", "content": answer})
Prompt to use in Version 1

User types something simple like:

I have files in the project directory. Run the compressor analysis and summarize the result.

or

Please analyze the files in the working directory and tell me what was done and what the result says.
VERSION 2 — LESS HARDCODED / MORE FLEXIBLE
Idea

frontend can optionally pass a working directory

default still exists

user still does not say tool names

good for experimentation

turbo/compressor/agent_service_flexible.py
import os
from fastapi import FastAPI
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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run")
async def run(req: RunRequest):
    job_dir = req.job_dir or DEFAULT_JOB_DIR
    answer = await run_agent_query(
        user_prompt=req.prompt,
        job_dir=job_dir
    )
    return {
        "status": "success",
        "job_dir": job_dir,
        "answer": answer
    }
Commands to run Version 2 on HPC
cd /path/to/agentic-mdao-hackathon
source .venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH
export LLM_MODEL=$PWD/examples/config_granite_example.json
export SCRATCH=/scratch/pwc/pw50647
export MDAO_DEFAULT_DIR=/scratch/pwc/pw50647/centrifugal_test_case/project_data

uvicorn turbo.compressor.agent_service_flexible:app --host 0.0.0.0 --port 8001
Health check
curl http://localhost:8001/health
Flexible API test
curl -X POST http://localhost:8001/run \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Please analyze the files in the working directory and summarize the result.","job_dir":"/scratch/pwc/pw50647/centrifugal_test_case/project_data"}'
Streamlit frontend code for Version 2

Again, make sure you have:

import requests

and:

MDAO_AGENT_URL = os.getenv("MDAO_AGENT_URL", "http://localhost:8001/run")

Replace run_agentic_mode() with this:

def run_agentic_mode():
    st.subheader("Agentic MDAO")

    with st.expander("Advanced Agent Settings", expanded=False):
        job_dir = st.text_input(
            "Working directory on HPC",
            value=os.getenv(
                "MDAO_DEFAULT_DIR",
                "/scratch/pwc/pw50647/centrifugal_test_case/project_data"
            ),
            key="agent_job_dir"
        )

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
        placeholder = st.empty()
        try:
            placeholder.markdown("Running on HPC...")
            resp = requests.post(
                MDAO_AGENT_URL,
                json={
                    "prompt": prompt,
                    "job_dir": job_dir
                },
                timeout=600
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data.get("answer", "No response returned.")
            placeholder.markdown(answer)
        except Exception as e:
            answer = f"Error calling HPC agent service: {e}"
            placeholder.markdown(answer)

    st.session_state.agent_history.append({"role": "assistant", "content": answer})
Prompt to use in Version 2

User can still type:

Please analyze the files in the working directory and tell me what was done and what the result says.

and the UI-provided directory goes separately.

How to keep the service running on HPC
Option A: use tmux
tmux new -s mdao

Inside tmux, run the uvicorn command.

Detach:

Ctrl+b then d

Reattach later:

tmux attach -t mdao
Option B: use nohup
mkdir -p logs
nohup uvicorn turbo.compressor.agent_service_simple:app --host 0.0.0.0 --port 8001 > logs/mdao_agent.log 2>&1 &

Check process:

ps -ef | grep uvicorn

Check logs:

tail -f logs/mdao_agent.log
What to do first

Start with this order:

Step 1

Replace:

turbo/compressor/functions.py

turbo/compressor/agent.py

Step 2

Test direct CLI:

python turbo/compressor/agent.py "I have files in the project directory. Run the compressor analysis and summarize the result."
Step 3

Create agent_service_simple.py

Step 4

Run:

uvicorn turbo.compressor.agent_service_simple:app --host 0.0.0.0 --port 8001
Step 5

Test:

curl http://localhost:8001/health

and then:

curl -X POST http://localhost:8001/run \
  -H "Content-Type: application/json" \
  -d '{"prompt":"I have files in the project directory. Run the compressor analysis and summarize the result."}'
Step 6

Update your Streamlit run_agentic_mode() with the Version 1 frontend code

Step 7

If it works, then move to Version 2

My recommendation

Use Version 1 first.
It is the fastest to get working and easiest to debug.

After that, move to Version 2 once the API path is stable.

The next best step is for me to give you a single final merged Application/app.py with the Agentic MDAO API version already inserted into your current Streamlit app.
