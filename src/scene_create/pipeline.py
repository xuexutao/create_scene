from __future__ import annotations

import time
import uuid
from pathlib import Path

from scene_create.config import STAGES
from scene_create.inputs import ingest, resolve_input
from scene_create.io import atomic_json, file_digest, fingerprint
from scene_create.runtime import (
    StageStore,
    WorkerRunner,
    configure_logging,
    output_lock,
)

DEPENDENCIES = {
    "ingest": [],
    "segment": ["ingest"],
    "depth": ["ingest"],
    "objects": ["ingest", "segment", "depth"],
    "complete": ["ingest", "objects"],
    "assets": ["complete"],
    "register": ["assets"],
    "background": ["register", "depth"],
    "assemble": ["register", "background", "depth"],
}
MODEL_FOR_STAGE = {
    "segment": "sam3",
    "depth": "da3",
    "complete": "flux2",
    "assets": "trellis",
    "background": "maskgaussian",
}
CONFIG_FOR_STAGE = {
    "ingest": [],
    "segment": [],
    "depth": ["geometry"],
    "objects": ["geometry", "fusion"],
    "complete": ["completion"],
    "assets": [],
    "register": ["registration"],
    "background": ["background"],
    "assemble": [],
}


class ScenePipeline:
    def __init__(self, config, output):
        self.config = config
        self.output = Path(output).expanduser().resolve()

    def stage_config(self, stage, imported):
        subset = {key: self.config[key] for key in CONFIG_FOR_STAGE[stage]}
        subset["runtime"] = {
            key: self.config["runtime"][key]
            for key in ("seed", "continue_on_object_error")
        }
        model = MODEL_FOR_STAGE.get(stage)
        if (
            model
            and not (imported and stage in ("segment", "depth"))
            and not (
                stage == "background"
                and self.config["background"]["mode"] == "init-only"
            )
        ):
            subset["model"] = self.config["models"][model]
            inventory = []
            for key in ("model_path", "dino_model_path", "rembg_model_path"):
                weights = subset["model"].get(key)
                if weights and Path(weights).exists():
                    path = Path(weights)
                    files = (
                        [path]
                        if path.is_file()
                        else sorted(p for p in path.rglob("*") if p.is_file())
                    )
                    inventory.extend(
                        {
                            "path": str(p),
                            "size": p.stat().st_size,
                            "mtime_ns": p.stat().st_mtime_ns,
                        }
                        for p in files
                    )
            subset["weight_inventory"] = inventory
            repo = Path(subset["model"]["repo_path"])
            subset["source_fingerprint"] = (
                fingerprint(
                    {
                        str(path.relative_to(repo)): file_digest(path)
                        for path in sorted(repo.rglob("*.py"))
                    }
                )
                if repo.is_dir()
                else None
            )
            manifest = Path(self.config["root"]) / "third_parts" / "manifest.json"
            if manifest.is_file():
                subset["source_manifest"] = file_digest(manifest)
        return subset

    def run(self, source, through="assemble", resume=False):
        if through not in STAGES:
            raise ValueError(f"Unknown pipeline stage: {through}")
        inputs = resolve_input(source)
        if (
            self.output.exists()
            and not (self.output / "run_manifest.json").exists()
            and any(self.output.iterdir())
        ):
            raise ValueError(
                "Output directory is nonempty and is not owned by this pipeline; choose a new directory"
            )
        for view in inputs["views"]:
            if self.output == Path(view["source_path"]).parent:
                raise ValueError("Output directory cannot be the input image directory")
        run_id = time.strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
        with output_lock(self.output):
            configure_logging(self.output, run_id)
            store = StageStore(self.output, run_id, resume=resume)
            runner = WorkerRunner(self.config, self.output, run_id=run_id)
            atomic_json(self.output / "metadata" / "config.resolved.json", self.config)
            package = Path(__file__).parent
            code = fingerprint(
                {
                    str(path.relative_to(package)): file_digest(path)
                    for path in sorted(package.rglob("*.py"))
                }
            )
            results, identities = {}, {}
            for stage in STAGES:
                settings = self.stage_config(
                    stage, inputs["provenance"] == "external_observations"
                )
                digest = fingerprint(
                    {
                        "stage": stage,
                        "code": code,
                        "settings": settings,
                        "inputs": inputs if stage == "ingest" else None,
                        "dependencies": {
                            name: identities[name] for name in DEPENDENCIES[stage]
                        },
                    }
                )
                previous = store.data["stages"].get(stage, {})
                if previous.get("fingerprint") != digest or previous.get("status") in (
                    "failed",
                    "partial",
                ):
                    invalid = {stage}
                    for descendant in STAGES:
                        if invalid.intersection(DEPENDENCIES[descendant]):
                            invalid.add(descendant)
                            if descendant in store.data["stages"]:
                                store.data["stages"][descendant]["status"] = "stale"
                result = store.run(
                    stage,
                    digest,
                    lambda destination, name=stage: self.execute(
                        name, destination, inputs, results, runner
                    ),
                )
                results[stage] = result
                identities[stage] = fingerprint(
                    {
                        "fingerprint": digest,
                        "result": result,
                        "artifacts": store.data["stages"][stage]["artifacts"],
                    }
                )
                if stage == through:
                    break
            failures = any(result.get("failures") for result in results.values())
            approximate = results.get("background", {}).get("approximate", False)
            insufficient = any(
                item.get("status") == "insufficient_geometry"
                for item in results.get("objects", {}).get("objects", [])
            )
            status = (
                "partial"
                if failures or insufficient
                else "approximate"
                if approximate
                else "completed"
            )
            store.data["completed_through"] = through
            store.finish(status)
            return {
                "status": status,
                "through": through,
                "output": str(self.output),
                "manifest": str(store.path),
                "result": results[through],
            }

    def execute(self, stage, destination, inputs, results, runner):
        if stage == "ingest":
            return ingest(inputs, destination, self.output)
        if stage in ("segment", "depth"):
            from scene_create.observations import geometry, segmentation

            if stage == "segment":
                return segmentation(results["ingest"], self.config, runner, destination)
            return geometry(
                results["ingest"], self.config, runner, destination, self.output
            )
        if stage == "objects":
            from scene_create.objects import build_objects

            return build_objects(
                results["segment"],
                results["depth"],
                results["ingest"],
                self.config,
                destination,
                self.output,
            )
        if stage in ("complete", "assets", "register"):
            from scene_create.reconstruction import (
                complete_objects,
                generate_assets,
                register_assets,
            )

            if stage == "complete":
                return complete_objects(
                    results["objects"],
                    results["ingest"],
                    self.config,
                    runner,
                    self.output,
                    destination,
                )
            if stage == "assets":
                return generate_assets(
                    results["complete"], self.config, runner, self.output, destination
                )
            return register_assets(
                results["assets"], self.config, self.output, destination
            )
        if stage == "background":
            from scene_create.background import reconstruct_background

            return reconstruct_background(
                results["register"],
                results["depth"],
                self.config,
                runner,
                destination,
                self.output,
            )
        if stage == "assemble":
            from scene_create.assembly import assemble_scene

            return assemble_scene(
                results["register"],
                results["background"],
                results["depth"],
                self.output,
            )
        raise ValueError(stage)
