from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from scene_create.io import read_json
from scene_create.runtime import WorkerRunner


def diagnose(config):
    root = Path(config["root"])
    manifest_path = root / "third_parts" / "manifest.json"
    pins = (
        read_json(manifest_path).get("repositories", {})
        if manifest_path.is_file()
        else {}
    )
    report = {
        "schema_version": 1,
        "ready": True,
        "models": {},
        "notes": [
            "Module discovery is not a real-model inference test.",
            "Weights, licenses and CUDA extensions must be provisioned in each configured worker environment.",
        ],
    }
    with tempfile.TemporaryDirectory(prefix=".scene-doctor-", dir=root) as temporary:
        runner = WorkerRunner(config, temporary)
        for name, model in config["models"].items():
            issues, warnings = [], []
            item = {
                "python": model["python"],
                "repository": model["repo_path"],
                "model_path": model.get("model_path"),
            }
            repo = Path(model["repo_path"])
            if not repo.is_dir():
                issues.append("Third-party repository is missing")
            else:
                command = subprocess.run(
                    ["git", "-C", str(repo), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if command.returncode == 0:
                    item["commit"] = command.stdout.strip()
                    expected = pins.get(name, {}).get("commit")
                    if expected and item["commit"] != expected:
                        issues.append(
                            "Source commit differs from third_parts/manifest.json; adapter compatibility must be revalidated"
                        )
                else:
                    warnings.append("Cannot establish the source revision")
            if name != "maskgaussian":
                weights = model.get("model_path")
                if not weights:
                    issues.append(f"Configure models.{name}.model_path")
                elif not model.get("allow_download") and not Path(weights).exists():
                    issues.append("Local weights do not exist; downloads are disabled")
                elif model.get("allow_download") and not Path(weights).exists():
                    warnings.append(
                        "Weights will require network access, sufficient disk space and any necessary license approval"
                    )
            if name == "trellis":
                for key in ("dino_model_path", "rembg_model_path"):
                    if not model.get(key) or not Path(model[key]).is_file():
                        issues.append(f"Configure a local models.trellis.{key}")
                if not (Path(model.get("dino_repo_path", "")) / "hubconf.py").is_file():
                    issues.append("The local DINOv2 source repository is missing")
            try:
                probe = runner.call(name, {}, Path(temporary) / name, probe=True)
                item["environment"] = probe
                missing = [
                    key for key, present in probe["modules"].items() if not present
                ]
                if missing:
                    issues.append("Missing modules: " + ", ".join(missing))
                if probe.get("torch_error"):
                    issues.append("Torch/device probe failed: " + probe["torch_error"])
                if model["device"].startswith("cuda") and not probe["cuda_available"]:
                    issues.append("CUDA is unavailable in this worker environment")
                if name == "sam3" and tuple(
                    int(p) for p in probe["python"].split(".")[:2]
                ) < (3, 12):
                    issues.append("Pinned SAM3 requires Python >= 3.12")
                if name == "trellis" and probe.get("memory_gib", 100) < 16:
                    warnings.append(
                        "Available GPU memory is below TRELLIS's documented 16GB prerequisite"
                    )
                if (
                    model.get("dtype") == "bfloat16"
                    and probe.get("capability", [0])[0] < 8
                ):
                    issues.append(
                        "Configured bfloat16 requires a compatible GPU; this GPU does not support native BF16"
                    )
            except Exception as error:
                issues.append(str(error))
            item.update(ready=not issues, issues=issues, warnings=warnings)
            report["models"][name] = item
            report["ready"] = report["ready"] and item["ready"]
    return report
