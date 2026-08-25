"""Install and audit Atelier generation engines one at a time on Colab.

This controller deliberately starts each provider in a child process and
clears CUDA between providers. A failed provider is recorded and does not
prevent the remaining engines from being checked.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageStat


ROOT = Path(os.environ.get("ATELIER_MODEL_ROOT", "/content/atelier-models"))
REFERENCE = Path(os.environ.get("ATELIER_AUDIT_REFERENCE", "/content/atelier-audit-reference.png"))
PROMPT = "A professional studio photograph of a leather handbag on a warm neutral background."


ENGINES = [
    {
        "id": "flux-schnell",
        "repo": "black-forest-labs/FLUX.1-schnell",
        "path": ROOT / "flux-schnell",
        "command": "python /content/flux_schnell_runner.py --model {model} --prompt {prompt} --reference {reference} --output {output} --width 768 --height 768 --steps 4 --strength 0.55 --seed 42",
    },
    {
        "id": "sdxl",
        "repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "path": ROOT / "sdxl",
        "command": "python /content/sdxl_runner.py --model {model} --prompt {prompt} --reference {reference} --output {output} --width 768 --height 768 --steps 8 --guidance 5 --strength 0.55 --seed 42",
    },
    {
        "id": "qwen-edit",
        "repo": "Qwen/Qwen-Image-Edit-2511",
        "path": ROOT / "qwen-edit",
        "command": "python /content/qwen_image_edit_runner.py --model {model} --prompt {prompt} --output {output} --width 768 --height 768 --steps 8 --guidance 4 --seed 42 --references {reference}",
    },
    {
        "id": "fooocus",
        "repo": os.environ.get("ATELIER_FOOOCUS_REPO", ""),
        "path": ROOT / "fooocus",
        "command": os.environ.get("ATELIER_AUDIT_COMMAND_FOOOCUS", ""),
    },
    {
        "id": "hidream",
        "repo": os.environ.get("ATELIER_HIDREAM_REPO", ""),
        "path": ROOT / "hidream",
        "command": os.environ.get("ATELIER_AUDIT_COMMAND_HIDREAM", ""),
    },
    {
        "id": "flux2",
        "repo": "black-forest-labs/FLUX.2-dev",
        "path": ROOT / "flux2",
        "command": os.environ.get("ATELIER_AUDIT_COMMAND_FLUX2", ""),
    },
    {
        "id": "flux2-klein",
        "repo": "black-forest-labs/FLUX.2-klein-4B",
        "path": ROOT / "flux2-klein",
        "command": os.environ.get("ATELIER_AUDIT_COMMAND_FLUX2_KLEIN", ""),
        "larger_gpu": True,
    },
]


def _make_reference() -> None:
    if REFERENCE.is_file():
        return
    REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (768, 768), (242, 238, 230))
    image.save(REFERENCE, format="PNG")


def _download(spec: dict[str, object]) -> None:
    repo = str(spec["repo"])
    if not repo:
        raise RuntimeError("MODEL_REPO_NOT_CONFIGURED")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo,
        local_dir=str(spec["path"]),
        token=os.environ.get("HUGGINGFACE_HUB_TOKEN") or None,
        ignore_patterns=["*.bin", "*.msgpack"],
    )


def _validate_output(output: Path) -> dict[str, object]:
    if not output.is_file():
        return {"passed": False, "reason": "OUTPUT_MISSING"}
    with Image.open(output) as image:
        image.load()
        stat = ImageStat.Stat(image.convert("RGB"))
        blank = max(stat.stddev) < 1.0
        return {
            "passed": image.width > 512 and image.height > 512 and not blank,
            "width": image.width,
            "height": image.height,
            "blank": blank,
        }


def _clear_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _audit(spec: dict[str, object], output_root: Path) -> dict[str, object]:
    engine_id = str(spec["id"])
    result: dict[str, object] = {"id": engine_id, "installed": False, "inference_passed": False}
    if spec.get("larger_gpu") and os.environ.get("ATELIER_ALLOW_LARGE_MODELS") != "1":
        result.update({"status": "requires_larger_gpu", "reason": "Explicit larger-GPU gate; not loaded on a T4."})
        return result
    command = str(spec.get("command") or "").strip()
    if not command and not (engine_id == "flux-schnell" and os.environ.get("ATELIER_PROVIDER_COMMAND")):
        result.update({"status": "skipped", "reason": "No provider runner configured; weights were not downloaded."})
        return result
    try:
        _download(spec)
        result["installed"] = True
    except Exception as exc:
        result.update({"status": "install_failed", "reason": f"{type(exc).__name__}: {exc}"})
        return result
    output = output_root / f"{engine_id}.png"
    adapter_command = os.environ.get("ATELIER_PROVIDER_COMMAND") if engine_id == "flux-schnell" else None
    if adapter_command:
        request = json.dumps({"action": "probe", "provider": engine_id, "output": str(output)})
        completed = subprocess.run(adapter_command, input=request, shell=True, text=True, capture_output=True, timeout=1800, check=False)
        try:
            adapter_result = json.loads(completed.stdout.splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            adapter_result = {}
        result["provider_result"] = adapter_result
    else:
        rendered = command.format(
            model=shlex.quote(str(spec["path"])),
            prompt=shlex.quote(PROMPT),
            reference=shlex.quote(str(REFERENCE)),
            output=shlex.quote(str(output)),
        )
        completed = subprocess.run(rendered, shell=True, text=True, capture_output=True, timeout=1800, check=False)
    validation = _validate_output(output)
    result.update({
        "status": "verified_inference" if completed.returncode == 0 and validation["passed"] else "audit_failed",
        "inference_passed": completed.returncode == 0 and validation["passed"],
        "output_validation": validation,
        "stderr": completed.stderr[-2000:],
    })
    _clear_cuda()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", action="append", help="Audit only this engine id; repeatable.")
    parser.add_argument("--report", default="/content/atelier-sequential-audit.json")
    parser.add_argument("--provider-command", help="JSON adapter command used by the worker for the selected provider.")
    args = parser.parse_args()
    if args.provider_command:
        os.environ["ATELIER_PROVIDER_COMMAND"] = args.provider_command
    _make_reference()
    output_root = Path(args.report).parent / "atelier-audit-images"
    output_root.mkdir(parents=True, exist_ok=True)
    selected = set(args.only or [str(spec["id"]) for spec in ENGINES])
    results = []
    for spec in ENGINES:
        if str(spec["id"]) not in selected:
            continue
        print(f"[{spec['id']}] install/load/audit/unload", flush=True)
        result = _audit(spec, output_root)
        results.append(result)
        print(json.dumps(result), flush=True)
    Path(args.report).write_text(json.dumps({"engines": results}, indent=2))
    return 0 if all(result.get("status") not in {"install_failed", "audit_failed"} for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())