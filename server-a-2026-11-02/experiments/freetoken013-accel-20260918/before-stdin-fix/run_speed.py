"""Run one admission-guarded FreeToken speed mode on server-a.

This adapter never builds, downloads, removes containers/images, changes
production configuration, starts Qwen, or retries a valid case.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

ROOT = Path("/srv/ai-server/artifacts/experiments/freetoken013-accel-20260918")
BASE_ROOT = Path("/srv/ai-server/artifacts/experiments/freetoken013-qwen38-nvfp4-20260918")
BENCH_ROOT = Path("/srv/ai-server/artifacts/experiments/freetoken013-benchmark120-20260918")
NETWORK = "freetoken013-nvfp4-loopback"
NETWORK_OPTIONS = {"com.docker.network.bridge.enable_ip_masquerade": "false",
                   "com.docker.network.bridge.enable_icc": "false"}
COMMIT = "cac247a860e316e06580d05aeb05f2e647bde214"
BASE_IMAGE = "sha256:10bc9d29f79941c71bb27a77974be4b2312ae64f02636dd5445db20e77675134"
ORIGINAL_BUNDLE_SHA256 = "b0bbaf4fddce284d2987b8dcabcc47d295c2f27686a7e4ff1ca45d246102dc95"
QWEN_ID = "15fc9d27b9c035e8d1304c0fcfc5c6a3170b4dc2c00e32a3b178696465f2ea5f"
QWEN_IMAGE = "sha256:5b14f62188d90d394b6e517bc76b964b511cde81fbb572ac5a87eaf41f5ed9c0"
COMPOSE = Path("/opt/homelab/ai-server/deploy/compose.qwen.yaml")
COMPOSE_SHA = "88c7de590dd7dea693dba225242d81e52387415a1e4e0f5796b568dfc4468c3c"
MODEL = "/srv/ai-server/staging/experiments/qwen38-uncensored-nvfp4-37b5130a/model"
WEIGHT = "/srv/ai-server/model-store/sha256/bc642c1159ab9b0fcd2fdc65be4fee72fa560cf0570604741db50061ea063277/model.safetensors"
TOKENIZER = "/srv/ai-server/staging/experiments/qwen38-syv-uncensored/leminkozey/f958acb153eb8c7bbd25638149c8c75185559226/model/tokenizer.json"
MODES = ("baseline", "native", "fi")
MODE_NAMES = {mode: "ai-server-freetoken-q38-013-speed-" + mode for mode in MODES}
MEASURED_CASES = ("expanded-he-075", "expanded-he-081", "expanded-he-119")
BASE_SERVE = ["serve", "--model", "/model", "--served-model-name", "qwen38-nvfp4-freetoken",
              "--host", "0.0.0.0", "--port", "1919", "--max-running-requests", "1",
              "--max-seq-len-override", "32768", "--num-tokens", "32768",
              "--max-output-tokens", "512", "--max-prefill-length", "8192"]


def serve_args(mode: str) -> list[str]:
    attention = "fi" if mode == "fi" else "triton"
    return BASE_SERVE + ["--attention-backend", attention,
                         "--quant-backend", "linear.nvfp4=triton",
                         "--cache-type", "radix", "--enable-cache-report",
                         "--text-model-only"]


def run(args, timeout=20, **kwargs):
    return subprocess.run(args, timeout=timeout, check=True, capture_output=True, **kwargs)


def inspect(name):
    return json.loads(run(["docker", "inspect", name]).stdout)[0]


def existing_candidate(mode: str):
    name = MODE_NAMES[mode]
    ids = run(["docker", "ps", "-aq", "--filter", "name=^/" + name + "$"]).stdout.split()
    if len(ids) > 1:
        raise RuntimeError("ambiguous_candidate_identity")
    return None if not ids else inspect(ids[0].decode())


def production_safe():
    qwen = inspect("ai-server-qwen")
    if not (qwen["Id"] == QWEN_ID and qwen["Image"] == QWEN_IMAGE
            and not qwen["State"]["Running"] and qwen["State"]["Status"] == "exited"
            and qwen["HostConfig"]["RestartPolicy"]["Name"] == "no"
            and hashlib.sha256(COMPOSE.read_bytes()).hexdigest() == COMPOSE_SHA):
        raise RuntimeError("production_qwen_prestate_changed")
    gateway = inspect("ai-server-gateway")
    if not gateway["State"]["Running"] or gateway["State"].get("Health", {}).get("Status") != "healthy":
        raise RuntimeError("gateway_not_healthy")
    if run(["systemctl", "is-active", "ai-server-supervisor.service"]).stdout.strip() != b"active":
        raise RuntimeError("supervisor_not_active")
    return {"qwen": "stopped", "container_id": QWEN_ID,
            "compose_sha256": COMPOSE_SHA, "gateway": "healthy"}


def gpu_idle():
    raw = run(["nvidia-smi", "--query-compute-apps=pid",
               "--format=csv,noheader,nounits"]).stdout.strip()
    if raw:
        raise RuntimeError("gpu_has_compute_work")


def memory_available():
    values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return {"mem_available_bytes": int(values["MemAvailable"].split()[0]) * 1024,
            "swap_free_bytes": int(values["SwapFree"].split()[0]) * 1024,
            "container_limit_bytes": 24 * 2**30}


def request_json(path: str):
    conn = http.client.HTTPConnection("127.0.0.1", 19191, timeout=10)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        body = response.read(1024 * 1024 + 1)
        if response.status != 200 or len(body) > 1024 * 1024:
            raise RuntimeError("http_not_ready_" + path)
        return json.loads(body)
    finally:
        conn.close()


def require_capacity(stats: dict, cache: dict):
    model, geometry = stats.get("model") or {}, cache.get("geometry") or {}
    if (model.get("id") != "qwen38-nvfp4-freetoken" or model.get("ctx") != 32768
            or model.get("input_modalities") != ["text"] or cache.get("state") != "serving"
            or geometry.get("num_pages", 0) * geometry.get("page_size", 0) < 32768):
        raise RuntimeError("requested_model_context_or_kv_capacity_not_proven")


def mounts():
    return {MODEL: "/model", WEIGHT: WEIGHT, TOKENIZER: TOKENIZER,
            str(BASE_ROOT / "instrumentation"): "/original-instrumentation",
            str(ROOT / "instrumentation"): "/instrumentation",
            str(ROOT / "speed_client.py"): "/speed_client.py",
            str(BENCH_ROOT / "harness/quality_expanded_v2.json"): "/speed-cases.json"}


def candidate_matches(value, mode: str, image_id: str, bundle_sha: str,
                      experiment_sha: str) -> bool:
    cfg, host = value["Config"], value["HostConfig"]
    actual_mounts = {x["Source"]: x["Destination"] for x in value["Mounts"]
                     if x["Type"] == "bind"}
    devices = host.get("DeviceRequests") or []
    gpu_zero_only = (len(devices) == 1 and devices[0].get("DeviceIDs") == ["0"]
                     and devices[0].get("Count") == 0
                     and devices[0].get("Driver") in ("", "nvidia")
                     and devices[0].get("Capabilities") == [["gpu"]])
    env = dict(x.split("=", 1) for x in cfg.get("Env", []) if "=" in x)
    labels = cfg.get("Labels", {})
    return (value["Name"] == "/" + MODE_NAMES[mode] and value["Image"] == image_id
            and labels.get("ai.server.experiment.bundle") == bundle_sha
            and labels.get("ai.server.experiment.speed-mode") == mode
            and labels.get("ai.server.experiment.manifest") == experiment_sha
            and cfg.get("Cmd") == serve_args(mode) and cfg.get("User") == "65532:65532"
            and cfg.get("Entrypoint") == ["/opt/freetoken013/bin/ft"]
            and host["RestartPolicy"]["Name"] == "no" and host["ReadonlyRootfs"] is True
            and host.get("NetworkMode") == NETWORK and host.get("Privileged") is False
            and host.get("Memory") == 24 * 2**30 and host.get("MemorySwap") == 24 * 2**30
            and host.get("NanoCpus") == 6 * 10**9 and host.get("PidsLimit") == 512
            and host.get("ShmSize") == 2**30 and gpu_zero_only
            and {x.upper() for x in host.get("CapDrop") or []} == {"ALL"}
            and not host.get("CapAdd") and not host.get("Devices")
            and host.get("SecurityOpt") in (["no-new-privileges"], ["no-new-privileges:true"])
            and host.get("Tmpfs") == {"/tmp": "rw,uid=65532,gid=65532,size=4g",
                                     "/cache": "rw,exec,nosuid,nodev,uid=65532,gid=65532,size=2g",
                                     "/app/models": "ro,uid=65532,gid=65532,size=1m"}
            and env.get("PYTHONPATH") == "/instrumentation"
            and env.get("FT013_JIT_TRACE") == "/cache/jit-trace.jsonl"
            and env.get("TORCHINDUCTOR_CACHE_DIR") == "/cache/torchinductor"
            and host.get("PortBindings") == {"1919/tcp": [{"HostIp": "127.0.0.1",
                                                            "HostPort": "19191"}]}
            and actual_mounts == mounts()
            and all(not x["RW"] for x in value["Mounts"] if x["Type"] == "bind")
            and all(x["Type"] in ("bind", "tmpfs") for x in value["Mounts"]))


def create_args(mode: str, image_id: str, bundle_sha: str,
                experiment_sha: str) -> list[str]:
    args = ["docker", "create", "--name", MODE_NAMES[mode], "--restart=no",
            "--gpus=device=0", "--read-only", "--user=65532:65532", "--cap-drop=ALL",
            "--security-opt=no-new-privileges", "--network=" + NETWORK,
            "--publish=127.0.0.1:19191:1919", "--memory=24g", "--memory-swap=24g",
            "--shm-size=1g", "--cpus=6", "--pids-limit=512",
            "--tmpfs=/tmp:rw,uid=65532,gid=65532,size=4g",
            "--tmpfs=/cache:rw,exec,nosuid,nodev,uid=65532,gid=65532,size=2g",
            "--tmpfs=/app/models:ro,uid=65532,gid=65532,size=1m",
            "--env=PYTHONPATH=/instrumentation",
            "--env=FT013_JIT_TRACE=/cache/jit-trace.jsonl",
            "--env=TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor",
            "--label=ai.server.experiment.bundle=" + bundle_sha,
            "--label=ai.server.experiment.speed-mode=" + mode,
            "--label=ai.server.experiment.manifest=" + experiment_sha,
            "--log-opt=max-size=16m", "--log-opt=max-file=2"]
    for source, destination in mounts().items():
        args += ["--mount", f"type=bind,source={source},target={destination},readonly"]
    return args + [image_id] + serve_args(mode)


def require_custody(path: Path, root: Path, expected_sha: str | None = None):
    info = path.lstat()
    if (not path.resolve().is_relative_to(root) or path.is_symlink() or info.st_uid != 0
            or info.st_mode & 0o022):
        raise RuntimeError("source_custody_" + path.name)
    if expected_sha is not None and hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise RuntimeError("source_hash_changed_" + path.name)


def validate_experiment_manifest(experiment_sha: str) -> dict:
    manifest_path = ROOT / "experiment-inputs.json"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != experiment_sha:
        raise RuntimeError("experiment_manifest_changed")
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest.items():
        require_custody(ROOT / name, ROOT, digest)
    for required in ("run_speed.py", "speed_client.py"):
        if required not in manifest:
            raise RuntimeError("experiment_manifest_missing_" + required)
    return manifest


def validate_image(mode: str, image_id: str, bundle_sha: str, manifest: dict, experiment_sha: str) -> dict:
    require_custody(BASE_ROOT / "installed-image.json", BASE_ROOT)
    base_receipt = json.loads((BASE_ROOT / "installed-image.json").read_text())
    if (base_receipt.get("id") != BASE_IMAGE
            or base_receipt.get("bundle_sha256") != ORIGINAL_BUNDLE_SHA256):
        raise RuntimeError("base_image_receipt_changed")
    if mode == "baseline":
        if image_id != BASE_IMAGE:
            raise RuntimeError("baseline_image_not_exact")
        receipt = base_receipt
    else:
        require_custody(ROOT / "installed-image.json", ROOT)
        receipt = json.loads((ROOT / "installed-image.json").read_text())
        artifact_rows = json.loads((ROOT / "artifact-manifest.json").read_text())["artifacts"]
        if (receipt.get("base_id") != BASE_IMAGE
                or receipt.get("base_layers") != base_receipt["layers"]
                or receipt.get("original_bundle_sha256") != bundle_sha
                or receipt.get("bundle_sha256") != experiment_sha
                or receipt.get("wheel_manifest_sha256") != manifest["artifact-manifest.json"]
                or receipt.get("packages_added") != {r["name"]:r["version"] for r in artifact_rows}):
            raise RuntimeError("derived_image_receipt_invalid")
    if (receipt.get("id") != image_id
            or (mode == "baseline" and receipt.get("bundle_sha256") != bundle_sha)):
        raise RuntimeError("image_not_bound_to_reviewed_build")
    image = inspect(image_id)
    layers = image.get("RootFS", {}).get("Layers")
    if layers != receipt.get("layers"):
        raise RuntimeError("image_layers_changed")
    base_layers = base_receipt.get("layers")
    if mode != "baseline" and (layers[:len(base_layers)] != base_layers
                                or len(layers) <= len(base_layers)):
        raise RuntimeError("derived_image_not_exact_base_prefix")
    labels = image["Config"].get("Labels", {})
    if (labels.get("org.opencontainers.image.revision") != COMMIT
            or labels.get("ai.server.freetoken.version") != "0.1.3"):
        raise RuntimeError("unverified_runtime_image")
    return {**receipt, "network_id":base_receipt["network_id"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--experiment-sha256", required=True)
    args = parser.parse_args()
    if (os.geteuid() != 0 or Path(__file__).resolve().parent != ROOT
            or not re.fullmatch(r"sha256:[a-f0-9]{64}", args.image_id)
            or not re.fullmatch(r"[a-f0-9]{64}", args.bundle_sha256)
            or not re.fullmatch(r"[a-f0-9]{64}", args.experiment_sha256)
            or args.bundle_sha256 != ORIGINAL_BUNDLE_SHA256):
        raise RuntimeError("execution_identity_or_path")
    for directory in (ROOT, *ROOT.parents):
        info = directory.lstat()
        if directory.is_symlink() or info.st_uid != 0 or info.st_mode & 0o022:
            raise RuntimeError("experiment_directory_custody")
    manifest = validate_experiment_manifest(args.experiment_sha256)
    for directory in (BASE_ROOT, *BASE_ROOT.parents):
        info = directory.lstat()
        if directory.is_symlink() or info.st_uid != 0 or info.st_mode & 0o022:
            raise RuntimeError("original_directory_custody")
    require_custody(BASE_ROOT / "candidate-inputs.json", BASE_ROOT, ORIGINAL_BUNDLE_SHA256)
    for name,digest in json.loads((BASE_ROOT / "candidate-inputs.json").read_text()).items():
        require_custody(BASE_ROOT / name, BASE_ROOT, digest)
    require_custody(BENCH_ROOT / "harness/quality_expanded_v2.json", BENCH_ROOT,
                    json.loads((ROOT / "case-provenance.json").read_text())["source_suite_sha256"])
    sys.path.insert(0, str(BASE_ROOT))
    from admission_window import exclusive_window
    from smoke_client import Vram
    receipt = validate_image(args.mode, args.image_id, args.bundle_sha256, manifest, args.experiment_sha256)
    network = json.loads(run(["docker", "network", "inspect", NETWORK]).stdout)[0]
    if (network["Id"] != receipt["network_id"] or network["Internal"] is not False
            or network["Driver"] != "bridge" or network.get("Containers")
            or network.get("Options") != NETWORK_OPTIONS):
        raise RuntimeError("candidate_network_not_isolated_or_idle")
    verified = json.loads((BASE_ROOT / "nv-preflight-data.json").read_text())["weights"]
    current = Path(WEIGHT).stat()
    if (current.st_size, current.st_ino, current.st_mtime_ns) != (
            verified["size"], verified["inode"], verified["mtime_ns"]):
        raise RuntimeError("verified_weight_file_changed")
    for row in json.loads((BASE_ROOT / "nv-integrity.json").read_text())["files"]:
        path = Path(MODEL) / row["file"]
        if str(path.resolve()) != row["resolved"] or path.stat().st_size != row["size"]:
            raise RuntimeError("model_reference_changed")
        if row["file"] != "model.safetensors" and hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError("small_model_file_changed")
    sys.path.insert(0, "/opt/homelab/ai-server")
    from storage_guard.storage_guard import StoragePolicy, admit, measure
    if not admit(measure(Path("/srv/ai-server")), 512 * 2**20,
                 StoragePolicy(root_free_floor_bytes=32 * 2**30)).allowed:
        raise RuntimeError("runtime_storage_admission_denied")
    output = ROOT / ("run-" + args.mode + "-" +
                     datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(mode=0o700)
    initial_free = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize

    def record(event, value):
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps({"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                     "event": event, "data": value}) + "\n")

    record("prestate", production_safe())
    outcome = {"status": "FAIL", "phase": "preflight", "terminal_class": None,
               "restoration_verified": False, "mode": args.mode, "image_id": args.image_id}
    owned = {"id": None, "create_attempted": False}
    value = existing_candidate(args.mode)
    if value is not None:
        if (not candidate_matches(value, args.mode, args.image_id, args.bundle_sha256,
                                  args.experiment_sha256)
                or value["State"]["Running"]
                or value["State"]["Status"] not in ("created", "exited")):
            raise RuntimeError("existing_candidate_not_safe_to_reuse")
        owned["id"] = value["Id"]

    def restore():
        if owned["id"] is None and not owned["create_attempted"]:
            gpu_idle()
            record("production_restored", production_safe())
            return
        current_candidate = existing_candidate(args.mode)
        if current_candidate is not None:
            if not candidate_matches(current_candidate, args.mode, args.image_id,
                                     args.bundle_sha256, args.experiment_sha256):
                raise RuntimeError("foreign_or_changed_candidate_no_stop")
            if owned["id"] is not None and current_candidate["Id"] != owned["id"]:
                raise RuntimeError("candidate_identity_changed_no_stop")
            owned["id"] = current_candidate["Id"]
            if current_candidate["State"]["Running"]:
                try:
                    trace = run(["docker", "exec", MODE_NAMES[args.mode], "cat",
                                 "/cache/jit-trace.jsonl"], timeout=10).stdout
                    (output / "jit-trace.jsonl").write_bytes(trace)
                except Exception as exc:
                    record("trace_capture_error", type(exc).__name__)
                try:
                    run(["docker", "stop", "--time=30", current_candidate["Id"]], timeout=45)
                except Exception:
                    if inspect(current_candidate["Id"])["State"]["Running"]:
                        raise RuntimeError("candidate_stop_unknown")
            final = inspect(current_candidate["Id"])
            if final["State"]["Running"] or final["State"]["Status"] not in ("exited", "created"):
                raise RuntimeError("candidate_not_stopped")
            record("candidate_final", {"id": final["Id"], "state": final["State"],
                                       "host_memory": memory_available()})
        elif owned["id"] is not None:
            raise RuntimeError("owned_candidate_disappeared")
        gpu_idle()
        record("production_restored", production_safe())

    def interrupted(signum, _frame):
        raise RuntimeError("interrupted_signal_" + str(signum))

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        with exclusive_window(restore, record):
            production_safe()
            gpu_idle()
            record("host_memory_before_start", memory_available())
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 19191))
            value = existing_candidate(args.mode)
            if value is None:
                owned["create_attempted"] = True
                run(create_args(args.mode, args.image_id, args.bundle_sha256,
                                args.experiment_sha256), timeout=60)
                value = existing_candidate(args.mode)
            if (value is None or not candidate_matches(value, args.mode, args.image_id,
                                                       args.bundle_sha256, args.experiment_sha256)
                    or value["State"]["Running"]):
                raise RuntimeError("candidate_config_or_prestate_invalid")
            owned["id"] = value["Id"]
            outcome["phase"] = "startup"
            startup_sample = Vram()
            startup_sample.thread.start()
            started = time.perf_counter()
            try:
                run(["docker", "start", owned["id"]], timeout=60)
                deadline = time.monotonic() + 900
                while True:
                    state = inspect(owned["id"])["State"]
                    if not state["Running"]:
                        raise RuntimeError("candidate_exited_during_load")
                    try:
                        health = request_json("/health")
                        if health.get("status") == "loading":
                            raise RuntimeError("model_still_loading")
                        stats = request_json("/v1/stats")
                        cache = request_json("/v1/cache/status")
                    except (OSError, http.client.HTTPException, RuntimeError, ValueError):
                        if time.monotonic() >= deadline:
                            raise RuntimeError("load_deadline_not_met")
                        time.sleep(5)
                        continue
                    if health.get("status") != "ok" or health.get("maintenance") != "serving":
                        raise RuntimeError("backend_not_healthy_" + str(health))
                    require_capacity(stats, cache)
                    ports = inspect(owned["id"])["NetworkSettings"]["Ports"]
                    if ports.get("1919/tcp") != [{"HostIp": "127.0.0.1", "HostPort": "19191"}]:
                        raise RuntimeError("effective_loopback_binding_not_proven")
                    break
                record("ready", {"startup_seconds": time.perf_counter() - started,
                                 "stats": stats, "health": health, "cache": cache,
                                 "effective_ports": ports,
                                 "vram_samples": startup_sample.samples,
                                 "sampling_errors": startup_sample.errors})
            finally:
                startup_sample.stop.set()
                startup_sample.thread.join(timeout=6)
                record("startup_observation", {"seconds": time.perf_counter() - started,
                                               "vram_samples": startup_sample.samples,
                                               "sampling_errors": startup_sample.errors})
            outcome["phase"] = "matrix"
            matrix_sample = Vram()
            matrix_sample.thread.start()
            try:
                with (output / "matrix.jsonl").open("wb") as stdout, \
                        (output / "matrix.stderr").open("wb") as stderr:
                    completed = subprocess.run(["docker", "exec", MODE_NAMES[args.mode],
                                                "/opt/freetoken013/bin/python", "-E", "-B",
                                                "/speed_client.py"], stdout=stdout, stderr=stderr,
                                               timeout=4 * 120 + 30)
            finally:
                matrix_sample.stop.set()
                matrix_sample.thread.join(timeout=6)
                record("matrix_observation", {"vram_samples": matrix_sample.samples,
                                              "sampling_errors": matrix_sample.errors,
                                              "host_memory": memory_available()})
            rows = [json.loads(line) for line in (output / "matrix.jsonl").read_text().splitlines()
                    if line]
            warmups = [row for row in rows if row.get("kind") == "warmup"]
            cases = [row for row in rows if row.get("kind") == "case"]
            if (completed.returncode or len(warmups) != 1
                    or [row.get("case_id") for row in cases] != list(MEASURED_CASES)
                    or any(row.get("status") != "PASS" for row in warmups + cases)):
                raise RuntimeError("finite_speed_matrix_failed_or_incomplete")
            if args.mode == "baseline":
                # Separate diagnostic request AFTER timings; never included in speed totals.
                run(["docker","exec",MODE_NAMES[args.mode],"touch","/cache/profile-request"])
                try:
                    with (output/"profile-request.jsonl").open("wb") as stdout, (output/"profile-request.stderr").open("wb") as stderr:
                        profiled = subprocess.run(["docker","exec",MODE_NAMES[args.mode],
                            "/opt/freetoken013/bin/python","-E","-B","/speed_client.py","--profile-only"],
                            stdout=stdout,stderr=stderr,timeout=180)
                    record("profile_request",{"returncode":profiled.returncode})
                    for filename in ["profile-summary.txt","profile-trace.json"]:
                        size=run(["docker","exec",MODE_NAMES[args.mode],"stat","-c","%s","/cache/"+filename]).stdout
                        if int(size)>128*2**20: raise RuntimeError("profile_output_too_large")
                        (output/filename).write_bytes(run(["docker","exec",MODE_NAMES[args.mode],"cat","/cache/"+filename],timeout=30).stdout)
                except Exception as exc:
                    record("profile_capture_error",str(exc))
            final_free = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
            growth = initial_free - final_free
            record("runtime_disk_growth", {"initial_free": initial_free, "final_free": final_free,
                                           "growth_bytes": growth, "bound_bytes": 512 * 2**20})
            if final_free < 32 * 2**30 or growth > 512 * 2**20:
                raise RuntimeError("runtime_storage_bound_exceeded")
            record("after_matrix", {"stats": request_json("/v1/stats"),
                                    "cache": request_json("/v1/cache/status")})
        outcome.update(status="PASS", phase="complete", terminal_class="valid_result",
                       restoration_verified=True)
    except BaseException as exc:
        outcome["terminal_class"] = ("startup_error" if outcome["phase"] == "startup"
                                     else "request_or_matrix_error" if outcome["phase"] == "matrix"
                                     else "preflight_error")
        outcome["error"] = type(exc).__name__ + ": " + str(exc)
        try:
            production_safe()
            gpu_idle()
            if owned["id"] is not None:
                final = inspect(owned["id"])
                if (not candidate_matches(final, args.mode, args.image_id, args.bundle_sha256,
                                          args.experiment_sha256)
                        or final["State"]["Running"]
                        or final["State"]["Status"] not in ("exited", "created")):
                    raise RuntimeError("candidate_restoration_not_proven")
            elif owned["create_attempted"] and existing_candidate(args.mode) is not None:
                raise RuntimeError("created_candidate_restoration_not_proven")
            from admission_window import Admission, checked_client, require_empty
            require_empty(Admission(checked_client(), record).request(), "public")
            outcome["restoration_verified"] = True
        except Exception as restore_error:
            outcome["terminal_class"] = "restoration_error"
            outcome["restoration_error"] = str(restore_error)
    finally:
        if owned["id"]:
            try:
                logs = run(["docker", "logs", "--tail=4000", owned["id"]], timeout=15)
                (output / "candidate.stdout.log").write_bytes(logs.stdout)
                (output / "candidate.stderr.log").write_bytes(logs.stderr)
            except Exception as exc:
                outcome["log_error"] = type(exc).__name__
        (output / "result.json").write_text(json.dumps(outcome, indent=2) + "\n")
        print(json.dumps({"output": str(output), **outcome}), flush=True)
    return 0 if outcome["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
