from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parent.parent
AI_ROOT = PROJECT_ROOT / "ai_service"
AI_HOST = "127.0.0.1"
AI_PORT = int(os.getenv("PHISHGUARD_INTERNAL_AI_PORT", "8100"))
PUBLIC_PORT = int(os.getenv("PORT", "10000"))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def configure_environment() -> dict[str, str]:
    runtime_root = Path(os.getenv("PHISHGUARD_RUNTIME_DIR", "/tmp/phishguard"))
    runtime_root.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PHISHGUARD_RUNTIME_DIR", str(runtime_root))
    os.environ.setdefault("PHISHGUARD_RULE_ENGINE_PATH", str(PROJECT_ROOT / "rule_engine"))
    os.environ.setdefault("PHISHGUARD_AI_SERVICE_URL", f"http://{AI_HOST}:{AI_PORT}")
    os.environ.setdefault("PHISHING_APP_HOME", str(runtime_root / "ai"))
    os.environ.setdefault("LLM_ENABLED", "true")
    os.environ.setdefault("ALLOW_DETERMINISTIC_FALLBACK", "false")
    os.environ.setdefault("LLM_PROVIDER", "openai")
    os.environ.setdefault("LLM_VENDOR", "deepseek")
    os.environ.setdefault("OPENAI_BASE_URL", "https://api.deepseek.com")
    os.environ.setdefault("OPENAI_MODEL", "deepseek-v4-flash")
    os.environ.setdefault("OCR_ENGINE", "tesseract")
    os.environ.setdefault("EASYOCR_GPU", "false")

    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(AI_ROOT), child_env.get("PYTHONPATH", ""))
        if value
    )
    return child_env


def wait_for_ai(process: subprocess.Popen[bytes], timeout_seconds: int = 120) -> None:
    health_url = f"http://{AI_HOST}:{AI_PORT}/api/v1/health"
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"AI service stopped during startup (exit {process.returncode})")
        try:
            with urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError):
            time.sleep(0.5)
    raise RuntimeError("AI service did not become healthy within 120 seconds")


def stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main() -> None:
    child_env = configure_environment()
    ai_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api:app",
            "--host",
            AI_HOST,
            "--port",
            str(AI_PORT),
            "--workers",
            "1",
        ],
        cwd=AI_ROOT,
        env=child_env,
    )

    def terminate(_signum: int, _frame: object) -> None:
        stop_process(ai_process)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    try:
        wait_for_ai(ai_process)
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=PUBLIC_PORT,
            workers=1,
            proxy_headers=True,
            forwarded_allow_ips="*",
        )
    finally:
        stop_process(ai_process)


if __name__ == "__main__":
    main()
