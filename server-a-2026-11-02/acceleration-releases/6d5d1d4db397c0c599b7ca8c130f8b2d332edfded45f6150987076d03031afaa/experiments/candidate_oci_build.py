#!/usr/bin/env python3
"""Build sealed candidates once with Docker and emit Docker-save plus OCI transports."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tarfile
from typing import Callable, Mapping, Sequence

try:
    from . import candidate_build_contract as contract
except ImportError:  # pragma: no cover
    import candidate_build_contract as contract  # type: ignore[no-redef]

HEX64 = re.compile(r"[0-9a-f]{64}\Z")
IMMUTABLE_REFERENCE = re.compile(r"[^\s@]+@sha256:([0-9a-f]{64})\Z")
OCI_INDEX = "application/vnd.oci.image.index.v1+json"
OCI_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
OCI_CONFIG = "application/vnd.oci.image.config.v1+json"
OCI_LAYER_GZIP = "application/vnd.oci.image.layer.v1.tar+gzip"
OCI_LAYERS = {OCI_LAYER_GZIP, "application/vnd.oci.image.layer.v1.tar+zstd"}
FREETOKEN_IMAGE_NAME = "local/ai-server-freetoken"
LLAMA_IMAGE_NAME = "local/ai-server-llama-dflash2-official"
IMAGE_NAME = FREETOKEN_IMAGE_NAME
FREETOKEN_EXPECTED_LABELS = {
    "org.opencontainers.image.source": "https://github.com/FlashML-org/FreeToken",
    "org.opencontainers.image.revision": contract.FREETOKEN_REVISION,
    "ai.server.freetoken.version": "0.1.2",
    "ai.server.freetoken.wheel.sha256": contract.FREETOKEN_WHEEL_SHA256,
    "ai.server.freetoken.requirements.sha256": contract.FREETOKEN_REQUIREMENTS_LOCK_SHA256,
    "ai.server.freetoken.requirements.offline.sha256": contract.FREETOKEN_REQUIREMENTS_OFFLINE_LOCK_SHA256,
    "ai.server.freetoken.artifacts.sha256": contract.FREETOKEN_ARTIFACT_MANIFEST_SHA256,
    "ai.server.freetoken.resolver-receipt.sha256": contract.FREETOKEN_RESOLVER_RECEIPT_SHA256,
    "ai.server.python-standalone.sha256": contract.PYTHON_STANDALONE_SHA256,
    "ai.server.healthcheck.runtime": "python-stdlib-urllib",
}
LLAMA_EXPECTED_LABELS = {
    "org.opencontainers.image.source": "https://github.com/ggml-org/llama.cpp",
    "org.opencontainers.image.revision": contract.LLAMA_REVISION,
    "ai.server.cuda.arch": "86",
}
EXPECTED_LABELS = FREETOKEN_EXPECTED_LABELS


class OciBuildError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _canonical_directory(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise OciBuildError(f"{label}_must_be_absolute_real_directory")
    resolved = path.resolve(strict=True)
    if resolved != path.absolute():
        raise OciBuildError(f"{label}_must_be_canonical")
    return resolved


def _new_output(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise OciBuildError(f"{label}_must_be_absolute_and_not_a_symlink")
    result = _canonical_directory(path.parent, f"{label}_parent") / path.name
    if result.exists():
        raise OciBuildError(f"{label}_already_exists")
    return result


def _unlink_private(path: Path) -> None:
    if path.exists():
        try:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
        path.unlink()


def _write_once(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o400)
    except BaseException:
        _unlink_private(path)
        raise


def _remove_private_tree(path: Path) -> None:
    def repair(function, target, _error):  # noqa: ANN001
        os.chmod(target, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        function(target)
    if path.exists():
        shutil.rmtree(path, onexc=repair)


def _copy_exact_tree(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise OciBuildError("sealed_input_contains_symlink")
        target = destination / path.relative_to(source)
        if path.is_dir():
            target.mkdir(mode=0o700)
        elif path.is_file():
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with path.open("rb", buffering=0) as reader, target.open("xb", buffering=0) as writer:
                shutil.copyfileobj(reader, writer, 8 * 1024 * 1024)
            os.chmod(target, 0o400)
        else:
            raise OciBuildError("sealed_input_contains_special_file")


def _safe_tail(value: str, limit: int = 4096) -> str:
    cleaned = "".join(c if c in "\n\t" or ord(c) >= 32 else "?" for c in value)
    cleaned = re.sub(r"(?i)\b(token|password|secret|authorization|api[_-]?key)\s*([:=])\s*([^\s]+)", r"\1\2[REDACTED]", cleaned)
    cleaned = re.sub(r"(?i)(https?://)[^/@\s]+:[^/@\s]+@", r"\1[REDACTED]@", cleaned)
    return cleaned[-limit:]


def _raise_failure(label: str, result: subprocess.CompletedProcess[str]) -> None:
    stdout = result.stdout if isinstance(result.stdout, str) else ""
    stderr = result.stderr if isinstance(result.stderr, str) else ""
    evidence = {
        "returncode": result.returncode,
        "stdout_sha256": hashlib.sha256(stdout.encode(errors="replace")).hexdigest(),
        "stdout_tail": _safe_tail(stdout),
        "stderr_sha256": hashlib.sha256(stderr.encode(errors="replace")).hexdigest(),
        "stderr_tail": _safe_tail(stderr),
    }
    raise OciBuildError(label + ":" + _canonical_json(evidence).decode().strip())


def _run(argv: Sequence[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(argv), check=False, text=True, capture_output=True, **kwargs)


def _load_json(payload: bytes, label: str) -> dict:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OciBuildError(f"invalid_{label}_json") from exc
    if not isinstance(value, dict):
        raise OciBuildError(f"invalid_{label}_json")
    return value


def _safe_members(archive: tarfile.TarFile, label: str) -> dict[str, tarfile.TarInfo]:
    result = {}
    for member in archive.getmembers():
        pure = PurePosixPath(member.name)
        if pure.is_absolute() or not pure.parts or any(p in {"", ".", ".."} for p in pure.parts) or member.name in result or not (member.isfile() or member.isdir()):
            raise OciBuildError(f"unsafe_{label}_member")
        result[member.name] = member
    return result


def _read_member(archive: tarfile.TarFile, members: Mapping[str, tarfile.TarInfo], name: str) -> bytes:
    member = members.get(name)
    if member is None or not member.isfile():
        raise OciBuildError(f"missing_archive_member:{name}")
    handle = archive.extractfile(member)
    if handle is None:
        raise OciBuildError(f"unreadable_archive_member:{name}")
    return handle.read()


def _descriptor(payload: bytes, media_type: str) -> dict[str, object]:
    return {"media_type": media_type, "digest": "sha256:" + hashlib.sha256(payload).hexdigest(), "size": len(payload)}


def _parse_descriptor(value: object, media_types: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise OciBuildError(f"missing_{label}_descriptor")
    media_type, digest, size = value.get("mediaType"), value.get("digest"), value.get("size")
    if media_type not in media_types or not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest) or type(size) is not int or size <= 0:
        raise OciBuildError(f"invalid_{label}_descriptor")
    return {"media_type": media_type, "digest": digest, "size": size}


def _inspect_oci_archive_details(
    archive_path: Path,
    image_name: str = IMAGE_NAME,
    image_tag: str | None = None,
    expected_labels: Mapping[str, str] = EXPECTED_LABELS,
) -> tuple[dict[str, object], dict[str, object]]:
    with tarfile.open(archive_path.resolve(strict=True), "r:*") as archive:
        members = _safe_members(archive, "oci_archive")
        files = {name for name, item in members.items() if item.isfile()}
        if any(name not in {"oci-layout", "index.json"} and not name.startswith("blobs/sha256/") for name in files):
            raise OciBuildError("unexpected_oci_archive_member")
        if _load_json(_read_member(archive, members, "oci-layout"), "oci_layout") != {"imageLayoutVersion": "1.0.0"}:
            raise OciBuildError("unsupported_oci_layout")
        index_payload = _read_member(archive, members, "index.json")
        index = _load_json(index_payload, "oci_index")
        manifests = index.get("manifests")
        if index.get("schemaVersion") != 2 or index.get("mediaType") != OCI_INDEX or not isinstance(manifests, list) or len(manifests) != 1:
            raise OciBuildError("oci_index_must_select_exactly_one_image")
        selected = manifests[0]
        manifest_desc = _parse_descriptor(selected, {OCI_MANIFEST}, "manifest")
        annotations = selected.get("annotations") if isinstance(selected, dict) else None
        if image_tag and (not isinstance(annotations, dict) or annotations.get("org.opencontainers.image.ref.name") != image_tag):
            raise OciBuildError("oci_index_tag_annotation_mismatch")

        def blob(descriptor: Mapping[str, object], label: str) -> bytes:
            payload = _read_member(archive, members, "blobs/sha256/" + str(descriptor["digest"])[7:])
            if len(payload) != descriptor["size"] or hashlib.sha256(payload).hexdigest() != str(descriptor["digest"])[7:]:
                raise OciBuildError(f"{label}_blob_identity_mismatch")
            return payload

        manifest_payload = blob(manifest_desc, "manifest")
        manifest = _load_json(manifest_payload, "oci_manifest")
        config_desc = _parse_descriptor(manifest.get("config"), {OCI_CONFIG}, "config")
        raw_layers = manifest.get("layers")
        if manifest.get("schemaVersion") != 2 or manifest.get("mediaType") != OCI_MANIFEST or not isinstance(raw_layers, list) or not raw_layers:
            raise OciBuildError("oci_manifest_layers_missing")
        layers = [_parse_descriptor(item, OCI_LAYERS, "layer") for item in raw_layers]
        config = _load_json(blob(config_desc, "config"), "oci_config")
        for layer in layers:
            blob(layer, "layer")
        labels = config.get("config", {}).get("Labels") if isinstance(config.get("config"), dict) else None
        if not isinstance(labels, dict) or any(labels.get(k) != v for k, v in expected_labels.items()):
            raise OciBuildError("oci_config_labels_mismatch")
        diff_ids = config.get("rootfs", {}).get("diff_ids") if isinstance(config.get("rootfs"), dict) else None
        if config.get("architecture") != "amd64" or config.get("os") != "linux" or not isinstance(diff_ids, list) or len(diff_ids) != len(layers) or any(not re.fullmatch(r"sha256:[0-9a-f]{64}", str(x)) for x in diff_ids):
            raise OciBuildError("oci_config_runtime_identity_incomplete")
        required = {"oci-layout", "index.json", "blobs/sha256/" + str(manifest_desc["digest"])[7:], "blobs/sha256/" + str(config_desc["digest"])[7:]}
        required.update("blobs/sha256/" + str(layer["digest"])[7:] for layer in layers)
        if files != required:
            raise OciBuildError("unexpected_oci_blob_set")
    index_desc = _descriptor(index_payload, OCI_INDEX)
    output = {"reference": f"{image_name}@{index_desc['digest']}", "digest": index_desc["digest"], "index": index_desc, "manifest": manifest_desc, "config": config_desc, "layers": layers, "labels": dict(expected_labels)}
    runtime = {"architecture": "amd64", "os": "linux", "config_digest": config_desc["digest"], "rootfs_diff_ids": diff_ids, "manifest_digest": manifest_desc["digest"]}
    return output, runtime


def inspect_oci_archive(archive_path: Path, image_name: str = IMAGE_NAME, expected_labels: Mapping[str, str] = EXPECTED_LABELS) -> dict[str, object]:
    return _inspect_oci_archive_details(archive_path, image_name, expected_labels=expected_labels)[0]


def _local_base_identity(runner: Callable[..., subprocess.CompletedProcess[str]], reference: str | None = None) -> dict[str, object]:
    reference = reference or contract.CUDA_BASE
    match = IMMUTABLE_REFERENCE.fullmatch(reference)
    if match is None:
        raise OciBuildError("base_reference_is_mutable")
    result = runner(("docker", "image", "inspect", "--platform", "linux/amd64", reference))
    if result.returncode != 0:
        raise OciBuildError("pinned_base_image_not_local")
    try:
        values = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise OciBuildError("base_image_inspect_invalid") from exc
    value = values[0] if isinstance(values, list) and len(values) == 1 and isinstance(values[0], dict) else None
    if value is None:
        raise OciBuildError("base_image_inspect_invalid")
    rootfs, repo_digests, config_id = value.get("RootFS"), value.get("RepoDigests"), value.get("Id")
    if not isinstance(repo_digests, list) or not any(isinstance(x, str) and x.endswith("@sha256:" + match.group(1)) for x in repo_digests) or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(config_id)) or not isinstance(rootfs, dict) or not isinstance(rootfs.get("Layers"), list) or not rootfs["Layers"] or value.get("Architecture") != "amd64" or value.get("Os") != "linux" or type(value.get("Size")) is not int or value["Size"] <= 0:
        raise OciBuildError("local_base_identity_incomplete")
    return {"reference": reference, "manifest_digest": "sha256:" + match.group(1), "local_config_digest": config_id, "local_rootfs_diff_ids": list(rootfs["Layers"]), "architecture": "amd64", "os": "linux", "unpacked_size": value["Size"]}


def _require_docker_driver(runner: Callable[..., subprocess.CompletedProcess[str]]) -> None:
    result = runner(("docker", "buildx", "inspect"))
    if result.returncode != 0:
        _raise_failure("buildx_inspect_failed", result)
    if re.findall(r"(?im)^Driver:\s*(\S+)\s*$", result.stdout) != ["docker"]:
        raise OciBuildError("active_buildx_driver_must_be_docker")


def _validate_dockerfile_offline(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if re.search(r"(?im)^\s*#\s*syntax\s*=", text):
        raise OciBuildError("external_dockerfile_syntax_forbidden")
    logical = re.sub(r"\\\r?\n\s*", " ", text)
    aliases: set[str] = set()
    for line in logical.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        instruction = stripped.split(None, 1)[0].upper()
        if instruction == "ADD":
            raise OciBuildError("dockerfile_add_forbidden")
        if re.search(r"(?i)\b(git\s+(clone|fetch|pull|submodule|remote)|curl|wget|invoke-webrequest)\b", stripped):
            raise OciBuildError("dockerfile_remote_source_forbidden")
        if instruction == "FROM":
            match = re.search(r"(?i)\s+AS\s+([A-Za-z0-9_.-]+)\s*$", stripped)
            if match:
                aliases.add(match.group(1))
        elif instruction == "RUN":
            match = re.match(r"(?i)^RUN\s+--network=(\S+)\s+", stripped)
            if match is None or match.group(1) != "none":
                raise OciBuildError("dockerfile_run_network_must_be_none")
        elif instruction == "COPY":
            match = re.search(r"(?i)--from=([^\s]+)", stripped)
            if match and match.group(1) not in aliases:
                raise OciBuildError("dockerfile_external_copy_source_forbidden")


def _build_argv(context: Path, reference: str, metadata: Path, build_args: Mapping[str, str]) -> tuple[str, ...]:
    argv = ["docker", "buildx", "build", "--platform", "linux/amd64", "--network=none", "--pull=false", "--no-cache", "--provenance=false", "--sbom=false", "--progress=plain"]
    for name, value in sorted(build_args.items()):
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name) or HEX64.fullmatch(value) is None:
            raise OciBuildError("invalid_exact_build_argument")
        argv.extend(("--build-arg", f"{name}={value}"))
    argv.extend(("--file", str(context / "Dockerfile"), "--tag", reference, "--metadata-file", str(metadata), str(context)))
    return tuple(argv)


def _inspect_engine_image(runner: Callable[..., subprocess.CompletedProcess[str]], reference: str, expected_labels: Mapping[str, str]) -> dict[str, object]:
    result = runner(("docker", "image", "inspect", "--platform", "linux/amd64", reference))
    if result.returncode != 0:
        raise OciBuildError("built_image_inspect_failed")
    try:
        values = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise OciBuildError("built_image_inspect_invalid") from exc
    value = values[0] if isinstance(values, list) and len(values) == 1 and isinstance(values[0], dict) else None
    if value is None:
        raise OciBuildError("built_image_inspect_invalid")
    labels = value.get("Config", {}).get("Labels") if isinstance(value.get("Config"), dict) else None
    rootfs = value.get("RootFS")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(value.get("Id"))) or value.get("Architecture") != "amd64" or value.get("Os") != "linux" or not isinstance(labels, dict) or any(labels.get(k) != v for k, v in expected_labels.items()) or not isinstance(rootfs, dict) or not isinstance(rootfs.get("Layers"), list) or not rootfs["Layers"]:
        raise OciBuildError("built_image_runtime_identity_mismatch")
    return {"architecture": "amd64", "os": "linux", "config_digest": value["Id"], "rootfs_diff_ids": list(rootfs["Layers"]), "labels": dict(expected_labels)}


def _parse_docker_save(raw: Path, reference: str, expected: Mapping[str, object], *, require_native: bool = True) -> tuple[dict[str, bytes], dict[str, object]]:
    with tarfile.open(raw, "r:*") as archive:
        members = _safe_members(archive, "docker_save")
        files = {name for name, item in members.items() if item.isfile()}
        try:
            manifests = json.loads(_read_member(archive, members, "manifest.json"))
        except json.JSONDecodeError as exc:
            raise OciBuildError("docker_save_manifest_invalid") from exc
        if not isinstance(manifests, list) or len(manifests) != 1 or not isinstance(manifests[0], dict):
            raise OciBuildError("docker_save_must_contain_one_manifest")
        entry = manifests[0]
        if set(entry) - {"Config", "RepoTags", "Layers", "LayerSources"} or entry.get("RepoTags") != [reference] or not isinstance(entry.get("Config"), str) or not isinstance(entry.get("Layers"), list) or not entry["Layers"]:
            raise OciBuildError("docker_save_manifest_identity_invalid")
        config_name = entry["Config"]
        layer_names = entry["Layers"]
        if any(not isinstance(name, str) or not re.fullmatch(r"blobs/sha256/[0-9a-f]{64}", name) for name in [config_name, *layer_names]):
            raise OciBuildError("docker_save_blob_path_invalid")
        payloads = {name: _read_member(archive, members, name) for name in [config_name, *layer_names]}
        config_hash = hashlib.sha256(payloads[config_name]).hexdigest()
        if config_hash != config_name.rsplit("/", 1)[1]:
            raise OciBuildError("docker_save_config_digest_mismatch")
        config = _load_json(payloads[config_name], "docker_save_config")
        diff_ids = config.get("rootfs", {}).get("diff_ids") if isinstance(config.get("rootfs"), dict) else None
        calculated = ["sha256:" + hashlib.sha256(payloads[name]).hexdigest() for name in layer_names]
        if calculated != diff_ids or expected["config_digest"] != "sha256:" + config_hash or expected["rootfs_diff_ids"] != diff_ids:
            raise OciBuildError("docker_save_rootfs_identity_mismatch")
        repositories_payload = _read_member(archive, members, "repositories")
        try:
            repositories = json.loads(repositories_payload)
        except json.JSONDecodeError as exc:
            raise OciBuildError("docker_save_repositories_invalid") from exc
        repository, tag = reference.rsplit(":", 1)
        if repositories != {repository: {tag: layer_names[-1].rsplit("/", 1)[1]}}:
            raise OciBuildError("docker_save_repositories_identity_invalid")
        required = {"manifest.json", "repositories", config_name, *layer_names}
        if require_native:
            required.update({"oci-layout", "index.json"})
            layout = _load_json(_read_member(archive, members, "oci-layout"), "docker_save_oci_layout")
            index = _load_json(_read_member(archive, members, "index.json"), "docker_save_oci_index")
            manifests_value = index.get("manifests")
            if layout != {"imageLayoutVersion": "1.0.0"} or not isinstance(manifests_value, list) or len(manifests_value) != 1:
                raise OciBuildError("docker_save_native_oci_layout_invalid")
            native_manifest = _parse_descriptor(manifests_value[0], {OCI_MANIFEST}, "docker_save_native_manifest")
            native_manifest_name = "blobs/sha256/" + str(native_manifest["digest"])[7:]
            native_manifest_payload = _read_member(archive, members, native_manifest_name)
            if len(native_manifest_payload) != native_manifest["size"] or hashlib.sha256(native_manifest_payload).hexdigest() != str(native_manifest["digest"])[7:]:
                raise OciBuildError("docker_save_native_manifest_identity_mismatch")
            native = _load_json(native_manifest_payload, "docker_save_native_manifest")
            native_config = _parse_descriptor(native.get("config"), {OCI_CONFIG}, "docker_save_native_config")
            native_layers_value = native.get("layers")
            if not isinstance(native_layers_value, list) or len(native_layers_value) != len(layer_names):
                raise OciBuildError("docker_save_native_layers_invalid")
            native_layers = [_parse_descriptor(item, {"application/vnd.oci.image.layer.v1.tar"}, "docker_save_native_layer") for item in native_layers_value]
            if "blobs/sha256/" + str(native_config["digest"])[7:] != config_name or ["blobs/sha256/" + str(item["digest"])[7:] for item in native_layers] != layer_names:
                raise OciBuildError("docker_save_native_descriptor_mismatch")
            required.add(native_manifest_name)
        extras = files - required
        if not required <= files or any(not re.fullmatch(r"blobs/sha256/[0-9a-f]{64}", name) for name in extras):
            raise OciBuildError("unexpected_docker_save_layout")
        for name in extras:
            if hashlib.sha256(_read_member(archive, members, name)).hexdigest() != name.rsplit("/", 1)[1]:
                raise OciBuildError("unexpected_docker_save_layout")
    entries = {
        "manifest.json": _canonical_json([{"Config": config_name, "RepoTags": [reference], "Layers": layer_names}]).rstrip(b"\n"),
        "repositories": _canonical_json(repositories).rstrip(b"\n"),
        **payloads,
    }
    return entries, {"config_name": config_name, "layer_names": layer_names, "diff_ids": diff_ids}


def _tar_add(output: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size, info.mtime, info.uid, info.gid = len(payload), 0, 0, 0
    info.uname, info.gname, info.mode = "", "", 0o444
    output.addfile(info, io.BytesIO(payload))


def _write_tar_once(path: Path, entries: Mapping[str, bytes]) -> None:
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise OciBuildError("partial_archive_already_exists")
    try:
        with partial.open("xb") as handle, tarfile.open(fileobj=handle, mode="w") as archive:
            for name in sorted(entries):
                _tar_add(archive, name, entries[name])
        os.link(partial, path)
        partial.unlink()
        os.chmod(path, 0o400)
    except BaseException:
        _unlink_private(partial)
        _unlink_private(path)
        raise


def _make_transports(raw: Path, docker_archive: Path, oci_archive: Path, reference: str, image_name: str, labels: Mapping[str, str], expected: Mapping[str, object]) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    docker_entries, details = _parse_docker_save(raw, reference, expected)
    _write_tar_once(docker_archive, docker_entries)
    config_name = str(details["config_name"])
    config_payload = docker_entries[config_name]
    config_desc = _descriptor(config_payload, OCI_CONFIG)
    oci_entries = {"oci-layout": _canonical_json({"imageLayoutVersion": "1.0.0"}).rstrip(b"\n"), "blobs/sha256/" + str(config_desc["digest"])[7:]: config_payload}
    layers = []
    docker_layers = []
    for name, diff_id in zip(details["layer_names"], details["diff_ids"], strict=True):
        raw_layer = docker_entries[str(name)]
        compressed = gzip.compress(raw_layer, compresslevel=9, mtime=0)
        descriptor = _descriptor(compressed, OCI_LAYER_GZIP)
        layers.append(descriptor)
        docker_layers.append({"path": name, "digest": diff_id, "size": len(raw_layer)})
        oci_entries["blobs/sha256/" + str(descriptor["digest"])[7:]] = compressed
    manifest_payload = _canonical_json({"schemaVersion": 2, "mediaType": OCI_MANIFEST, "config": {"mediaType": OCI_CONFIG, "digest": config_desc["digest"], "size": config_desc["size"]}, "layers": [{"mediaType": x["media_type"], "digest": x["digest"], "size": x["size"]} for x in layers]}).rstrip(b"\n")
    manifest_desc = _descriptor(manifest_payload, OCI_MANIFEST)
    oci_entries["blobs/sha256/" + str(manifest_desc["digest"])[7:]] = manifest_payload
    index_payload = _canonical_json({"schemaVersion": 2, "mediaType": OCI_INDEX, "manifests": [{"mediaType": OCI_MANIFEST, "digest": manifest_desc["digest"], "size": manifest_desc["size"], "annotations": {"org.opencontainers.image.ref.name": reference}}]}).rstrip(b"\n")
    oci_entries["index.json"] = index_payload
    _write_tar_once(oci_archive, oci_entries)
    output, runtime = _inspect_oci_archive_details(oci_archive, image_name=image_name, image_tag=reference, expected_labels=labels)
    if runtime["config_digest"] != expected["config_digest"] or runtime["rootfs_diff_ids"] != expected["rootfs_diff_ids"]:
        raise OciBuildError("oci_transport_runtime_identity_mismatch")
    docker_identity = {"config": {"digest": runtime["config_digest"], "size": len(config_payload)}, "layers": docker_layers, "rootfs_diff_ids": runtime["rootfs_diff_ids"]}
    return output, runtime, docker_identity


def _remove_owned_tag(runner: Callable[..., subprocess.CompletedProcess[str]], reference: str) -> dict[str, object]:
    result = runner(("docker", "image", "rm", reference))
    if result.returncode != 0:
        _raise_failure("temporary_image_cleanup_failed", result)
    check = runner(("docker", "image", "inspect", "--platform", "linux/amd64", reference))
    if check.returncode == 0:
        raise OciBuildError("temporary_image_cleanup_unverified")
    return {"argv": ["docker", "image", "rm", reference], "verified_absent": True}


def _validate_llama_receipt(receipt: object, manifest_sha256: str, dockerfile_sha256: str, helper_hashes: Mapping[str, str], bases: Mapping[str, object], build_args: Mapping[str, str], output: Mapping[str, object]) -> None:
    expected = {
        "format": "ai-server-llama-dflash2-official-oci-build-receipt-v1",
        "candidate_id": "llama-dflash2-official",
        "revision": contract.LLAMA_REVISION,
        "bases": dict(bases),
        "sealed_input_manifest_sha256": manifest_sha256,
        "source": {"dockerfile_sha256": dockerfile_sha256, "healthcheck_c_sha256": helper_hashes["healthcheck.c"], "seal_elf_runtime_sha256": helper_hashes["seal-elf-runtime.sh"]},
        "build": {"platform": "linux/amd64", "network": "none", "pull": False, "no_cache": True, "provenance": False, "sbom": False, "args": dict(build_args)},
        "output": dict(output),
    }
    if receipt != expected or output.get("labels") != LLAMA_EXPECTED_LABELS:
        raise OciBuildError("llama_oci_build_receipt_mismatch")


def _build_candidate(
    *, candidate_id: str, source: Path, sealed: Path, manifest_sha256: str,
    dockerfile: Path, helpers: Mapping[str, Path], build_args: Mapping[str, str],
    bases: Mapping[str, object], image_name: str, revision: str,
    labels: Mapping[str, str], docker_archive: Path, oci_archive: Path,
    receipt_path: Path, lock_path: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, object]:
    outputs = [docker_archive, oci_archive, receipt_path, lock_path]
    if len(set(outputs)) != 4 or len({p.parent for p in outputs}) != 1:
        raise OciBuildError("transport_files_must_be_distinct_and_share_one_directory")
    if any(p.is_relative_to(source) or p.is_relative_to(sealed) for p in outputs):
        raise OciBuildError("outputs_must_not_modify_source_or_sealed_input")
    _validate_dockerfile_offline(dockerfile)
    _require_docker_driver(runner)
    reference = f"{image_name}:{revision}"
    if runner(("docker", "image", "inspect", "--platform", "linux/amd64", reference)).returncode == 0:
        raise OciBuildError("build_would_overwrite_existing_tag")
    context = docker_archive.with_name(docker_archive.name + ".context")
    metadata = docker_archive.with_name(docker_archive.name + ".metadata.partial")
    raw = docker_archive.with_name(docker_archive.name + ".docker-save.partial")
    reservation = docker_archive.with_name(docker_archive.name + ".building")
    if any(path.exists() for path in (context, metadata, raw, reservation)):
        raise OciBuildError("temporary_build_path_already_exists")
    _write_once(reservation, b"offline candidate build reserved\n")
    built = False
    cleaned = False
    try:
        context.mkdir(mode=0o700)
        for name, path in {"Dockerfile": dockerfile, **helpers}.items():
            shutil.copyfile(path, context / name)
            os.chmod(context / name, 0o400)
        sealed_target = context / "sealed-input"
        sealed_target.mkdir(mode=0o700)
        _copy_exact_tree(sealed, sealed_target)
        argv = _build_argv(context, reference, metadata, build_args)
        result = runner(argv)
        if result.returncode != 0:
            built = runner(("docker", "image", "inspect", "--platform", "linux/amd64", reference)).returncode == 0
            _raise_failure("offline_build_failed", result)
        built = True
        runtime = _inspect_engine_image(runner, reference, labels)
        if not metadata.is_file() or metadata.is_symlink():
            raise OciBuildError("build_metadata_missing")
        metadata_payload = metadata.read_bytes()
        metadata_json = _load_json(metadata_payload, "build_metadata")
        if metadata_json.get("containerimage.config.digest") not in {None, runtime["config_digest"]}:
            raise OciBuildError("build_metadata_config_identity_mismatch")
        save = runner(("docker", "image", "save", "--output", str(raw), reference))
        if save.returncode != 0:
            _raise_failure("docker_image_save_failed", save)
        if not raw.is_file() or raw.is_symlink() or raw.stat().st_size <= 0:
            raise OciBuildError("docker_image_save_output_missing")
        output, oci_runtime, docker_identity = _make_transports(raw, docker_archive, oci_archive, reference, image_name, labels, runtime)
        dockerfile_hash = _sha256(dockerfile)
        build_contract = {"platform": "linux/amd64", "network": "none", "pull": False, "no_cache": True, "provenance": False, "sbom": False, "args": dict(build_args)}
        if candidate_id == "freetoken":
            receipt = {"format": "ai-server-freetoken-oci-build-receipt-v1", "candidate_id": candidate_id, "base": {"reference": contract.CUDA_BASE, "digest": contract.FREETOKEN_BASE_DIGEST}, "sealed_input_manifest_sha256": manifest_sha256, "dockerfile_sha256": dockerfile_hash, "build": {"platform": "linux/amd64", "network": "none", "pull": False, "no_cache": True, "args": dict(build_args)}, "output": output}
            try:
                contract.validate_freetoken_oci_receipt(receipt, expected_sealed_input_manifest_sha256=manifest_sha256)
            except contract.BuildContractError as exc:
                raise OciBuildError("build_receipt_contract_mismatch") from exc
        else:
            helper_hashes = {name: _sha256(path) for name, path in helpers.items()}
            receipt = {"format": "ai-server-llama-dflash2-official-oci-build-receipt-v1", "candidate_id": candidate_id, "revision": revision, "bases": dict(bases), "sealed_input_manifest_sha256": manifest_sha256, "source": {"dockerfile_sha256": dockerfile_hash, "healthcheck_c_sha256": helper_hashes["healthcheck.c"], "seal_elf_runtime_sha256": helper_hashes["seal-elf-runtime.sh"]}, "build": build_contract, "output": output}
            _validate_llama_receipt(receipt, manifest_sha256, dockerfile_hash, helper_hashes, bases, build_args, output)
        receipt_payload = _canonical_json(receipt)
        _write_once(receipt_path, receipt_payload)
        cleanup = _remove_owned_tag(runner, reference)
        cleaned = True
        source_identity = {"dockerfile": str(dockerfile.relative_to(source)).replace("\\", "/"), "dockerfile_sha256": dockerfile_hash, "sealed_input_manifest_sha256": manifest_sha256}
        if helpers:
            source_identity.update({"healthcheck_c_sha256": _sha256(helpers["healthcheck.c"]), "seal_elf_runtime_sha256": _sha256(helpers["seal-elf-runtime.sh"])})
        lock = {
            "format": "ai-server-candidate-image-lock-v2", "candidate_id": candidate_id,
            "revision": revision,
            "image": {"reference": reference, "digest": runtime["config_digest"], "digest_kind": "docker-image-config", "docker_image_id": runtime["config_digest"], "manifest_digest": output["manifest"]["digest"]},
            "base": dict(bases), "source": source_identity,
            "build": {"argv": list(argv), "platform": "linux/amd64", "network": "none", "pull": False, "no_cache": True, "provenance": False, "sbom": False, "build_args": dict(build_args), "metadata": {"sha256": hashlib.sha256(metadata_payload).hexdigest(), "size": len(metadata_payload), "values": metadata_json}},
            "archives": {"docker": {"filename": docker_archive.name, "format": "docker-image-save-tar", "sha256": _sha256(docker_archive), "size": docker_archive.stat().st_size}, "oci": {"filename": oci_archive.name, "format": "oci-image-layout-tar", "sha256": _sha256(oci_archive), "size": oci_archive.stat().st_size}},
            "docker": docker_identity, "oci": output,
            "receipt": {"filename": receipt_path.name, "sha256": hashlib.sha256(receipt_payload).hexdigest(), "size": len(receipt_payload)},
            "temporary_tag_cleanup": cleanup,
            "docker_load_probe": {"argv": ["docker", "image", "load", "--input", docker_archive.name], "network_required": False, "expected_runtime_identity": oci_runtime},
        }
        lock_payload = _canonical_json(lock)
        _write_once(lock_path, lock_payload)
        return {"docker_archive": str(docker_archive), "docker_archive_sha256": _sha256(docker_archive), "oci_archive": str(oci_archive), "oci_archive_sha256": _sha256(oci_archive), "digest": runtime["config_digest"], "receipt": str(receipt_path), "lock": str(lock_path), "lock_sha256": hashlib.sha256(lock_payload).hexdigest()}
    except BaseException:
        for path in outputs:
            _unlink_private(path)
        raise
    finally:
        cleanup_error = None
        if built and not cleaned:
            try:
                _remove_owned_tag(runner, reference)
            except BaseException as exc:  # preserve evidence cleanup failure
                cleanup_error = exc
        _remove_private_tree(context)
        for path in (metadata, raw, reservation):
            _unlink_private(path)
        if cleanup_error is not None:
            for path in outputs:
                _unlink_private(path)
            raise cleanup_error


def build_freetoken(source_root: Path, sealed_input: Path, manifest_sha256: str, docker_archive: Path, oci_archive: Path, receipt_path: Path, lock_path: Path, *, runner: Callable[..., subprocess.CompletedProcess[str]] = _run) -> dict[str, object]:
    source = _canonical_directory(source_root, "source_root")
    sealed = _canonical_directory(sealed_input, "sealed_input")
    if HEX64.fullmatch(manifest_sha256) is None:
        raise OciBuildError("manifest_sha256_invalid")
    try:
        contract.validate_freetoken_assets(source)
        contract.verify_sealed_input(sealed, manifest_sha256, candidate="freetoken")
    except contract.BuildContractError as exc:
        raise OciBuildError(str(exc)) from exc
    docker_archive, oci_archive, receipt_path, lock_path = [_new_output(path, label) for path, label in ((docker_archive, "docker_archive"), (oci_archive, "oci_archive"), (receipt_path, "receipt"), (lock_path, "lock"))]
    dockerfile = source / "freetoken" / "Dockerfile"
    if _sha256(dockerfile) != contract.FREETOKEN_DOCKERFILE_SHA256:
        raise OciBuildError("dockerfile_identity_mismatch")
    base = _local_base_identity(runner)
    return _build_candidate(candidate_id="freetoken", source=source, sealed=sealed, manifest_sha256=manifest_sha256, dockerfile=dockerfile, helpers={}, build_args={"SEALED_INPUT_MANIFEST_SHA256": manifest_sha256}, bases=base, image_name=FREETOKEN_IMAGE_NAME, revision=contract.FREETOKEN_REVISION, labels=FREETOKEN_EXPECTED_LABELS, docker_archive=docker_archive, oci_archive=oci_archive, receipt_path=receipt_path, lock_path=lock_path, runner=runner)


def build_llama(source_root: Path, sealed_input: Path, manifest_sha256: str, docker_archive: Path, oci_archive: Path, receipt_path: Path, lock_path: Path, *, runner: Callable[..., subprocess.CompletedProcess[str]] = _run) -> dict[str, object]:
    source = _canonical_directory(source_root, "source_root")
    sealed = _canonical_directory(sealed_input, "sealed_input")
    if HEX64.fullmatch(manifest_sha256) is None:
        raise OciBuildError("manifest_sha256_invalid")
    try:
        contract.validate_llama_assets(source)
        contract.verify_sealed_input(sealed, manifest_sha256, candidate="llama-dflash2-official", source_archive_sha256=contract.LLAMA_SOURCE_ARCHIVE_SHA256)
    except contract.BuildContractError as exc:
        raise OciBuildError(str(exc)) from exc
    directory = source / "llama_dflash2_official"
    dockerfile = directory / "Dockerfile"
    helpers = {"healthcheck.c": directory / "healthcheck.c", "seal-elf-runtime.sh": directory / "seal-elf-runtime.sh"}
    docker_archive, oci_archive, receipt_path, lock_path = [_new_output(path, label) for path, label in ((docker_archive, "docker_archive"), (oci_archive, "oci_archive"), (receipt_path, "receipt"), (lock_path, "lock"))]
    build_args = {"RUNTIME_CLOSURE_LOCK_SHA256": _sha256(sealed / "runtime-closure-lock.json"), "SEALED_INPUT_MANIFEST_SHA256": manifest_sha256, "TOOLCHAIN_LOCK_SHA256": _sha256(sealed / "toolchain-lock.json")}
    bases = {"builder": _local_base_identity(runner, contract.LLAMA_CUDA_BUILDER_BASE), "runtime": _local_base_identity(runner, contract.LLAMA_CUDA_RUNTIME_BASE)}
    return _build_candidate(candidate_id="llama-dflash2-official", source=source, sealed=sealed, manifest_sha256=manifest_sha256, dockerfile=dockerfile, helpers=helpers, build_args=build_args, bases=bases, image_name=LLAMA_IMAGE_NAME, revision=contract.LLAMA_REVISION, labels=LLAMA_EXPECTED_LABELS, docker_archive=docker_archive, oci_archive=oci_archive, receipt_path=receipt_path, lock_path=lock_path, runner=runner)


def _load_transport_lock(docker_archive: Path, oci_archive: Path, lock_path: Path) -> tuple[dict, dict[str, object], dict[str, str]]:
    payload = lock_path.read_bytes()
    lock = _load_json(payload, "image_lock")
    if _canonical_json(lock) != payload or lock.get("format") != "ai-server-candidate-image-lock-v2":
        raise OciBuildError("image_lock_is_not_canonical")
    expected_archives = {"docker": {"filename": docker_archive.name, "format": "docker-image-save-tar", "sha256": _sha256(docker_archive), "size": docker_archive.stat().st_size}, "oci": {"filename": oci_archive.name, "format": "oci-image-layout-tar", "sha256": _sha256(oci_archive), "size": oci_archive.stat().st_size}}
    if lock.get("archives") != expected_archives:
        raise OciBuildError("archive_lock_identity_mismatch")
    candidate = lock.get("candidate_id")
    if candidate == "freetoken":
        image_name, revision, labels = FREETOKEN_IMAGE_NAME, contract.FREETOKEN_REVISION, FREETOKEN_EXPECTED_LABELS
    elif candidate == "llama-dflash2-official":
        image_name, revision, labels = LLAMA_IMAGE_NAME, contract.LLAMA_REVISION, LLAMA_EXPECTED_LABELS
    else:
        raise OciBuildError("image_lock_candidate_not_allowlisted")
    reference = f"{image_name}:{revision}"
    output, runtime = _inspect_oci_archive_details(oci_archive, image_name=image_name, image_tag=reference, expected_labels=labels)
    image = lock.get("image")
    if not isinstance(image, dict) or image.get("reference") != reference or image.get("digest") != runtime["config_digest"] or image.get("docker_image_id") != runtime["config_digest"] or image.get("manifest_digest") != output["manifest"]["digest"] or lock.get("oci") != output:
        raise OciBuildError("locked_image_identity_invalid")
    docker_entries, details = _parse_docker_save(docker_archive, reference, runtime, require_native=False)
    if set(docker_entries) != {"manifest.json", "repositories", details["config_name"], *details["layer_names"]}:
        raise OciBuildError("docker_transport_identity_invalid")
    receipt = lock.get("receipt")
    if not isinstance(receipt, dict):
        raise OciBuildError("receipt_lock_identity_missing")
    receipt_path = lock_path.parent / str(receipt.get("filename"))
    if not receipt_path.is_file() or receipt.get("sha256") != _sha256(receipt_path) or receipt.get("size") != receipt_path.stat().st_size:
        raise OciBuildError("receipt_lock_identity_mismatch")
    receipt_payload = receipt_path.read_bytes()
    receipt_value = _load_json(receipt_payload, "build_receipt")
    if _canonical_json(receipt_value) != receipt_payload or receipt_value.get("output") != output:
        raise OciBuildError("build_receipt_output_mismatch")
    source = lock.get("source")
    build = lock.get("build")
    if not isinstance(source, dict) or not isinstance(build, dict):
        raise OciBuildError("build_receipt_binding_missing")
    if candidate == "freetoken":
        try:
            contract.validate_freetoken_oci_receipt(receipt_value, expected_sealed_input_manifest_sha256=str(source.get("sealed_input_manifest_sha256")))
        except contract.BuildContractError as exc:
            raise OciBuildError("build_receipt_contract_mismatch") from exc
    else:
        _validate_llama_receipt(
            receipt_value,
            str(source.get("sealed_input_manifest_sha256")),
            str(source.get("dockerfile_sha256")),
            {"healthcheck.c": str(source.get("healthcheck_c_sha256")), "seal-elf-runtime.sh": str(source.get("seal_elf_runtime_sha256"))},
            lock.get("base") if isinstance(lock.get("base"), dict) else {},
            build.get("build_args") if isinstance(build.get("build_args"), dict) else {},
            output,
        )
    if lock.get("temporary_tag_cleanup") != {"argv": ["docker", "image", "rm", reference], "verified_absent": True}:
        raise OciBuildError("temporary_tag_cleanup_evidence_missing")
    return lock, runtime, labels


def probe_docker_load(docker_archive: Path, oci_archive: Path, lock_path: Path, *, runner: Callable[..., subprocess.CompletedProcess[str]] = _run) -> dict[str, object]:
    docker_archive, oci_archive, lock_path = docker_archive.resolve(strict=True), oci_archive.resolve(strict=True), lock_path.resolve(strict=True)
    lock, expected, labels = _load_transport_lock(docker_archive, oci_archive, lock_path)
    reference = str(lock["image"]["reference"])
    if runner(("docker", "image", "inspect", "--platform", "linux/amd64", reference)).returncode == 0:
        raise OciBuildError("docker_load_would_overwrite_existing_tag")
    daemon = runner(("docker", "version", "--format", "{{.Server.Version}}"))
    if daemon.returncode != 0 or not daemon.stdout.strip():
        raise OciBuildError("docker_daemon_unavailable")
    loaded = False
    argv = ("docker", "image", "load", "--input", str(docker_archive))
    try:
        result = runner(argv)
        if result.returncode != 0:
            _raise_failure("stock_docker_load_failed", result)
        loaded = True
        actual = _inspect_engine_image(runner, reference, labels)
        if actual["config_digest"] != expected["config_digest"] or actual["rootfs_diff_ids"] != expected["rootfs_diff_ids"]:
            raise OciBuildError("loaded_image_runtime_identity_mismatch")
        return {"argv": list(argv), "image": lock["image"], "status": "docker_load_identity_verified_and_cleaned"}
    finally:
        if loaded:
            _remove_owned_tag(runner, reference)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--candidate", choices=("freetoken", "llama-dflash2-official"), required=True)
    build.add_argument("--source-root", type=Path, required=True)
    build.add_argument("--sealed-input", type=Path, required=True)
    build.add_argument("--manifest-sha256", required=True)
    build.add_argument("--docker-archive", type=Path, required=True)
    build.add_argument("--oci-archive", type=Path, required=True)
    build.add_argument("--receipt", type=Path, required=True)
    build.add_argument("--lock", type=Path, required=True)
    probe = commands.add_parser("probe-load")
    probe.add_argument("--docker-archive", type=Path, required=True)
    probe.add_argument("--oci-archive", type=Path, required=True)
    probe.add_argument("--lock", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "build":
            builder = build_freetoken if arguments.candidate == "freetoken" else build_llama
            result = builder(arguments.source_root, arguments.sealed_input, arguments.manifest_sha256, arguments.docker_archive, arguments.oci_archive, arguments.receipt, arguments.lock)
        else:
            result = probe_docker_load(arguments.docker_archive, arguments.oci_archive, arguments.lock)
    except (OciBuildError, OSError, tarfile.TarError) as exc:
        print(json.dumps({"error": str(exc), "status": "blocked"}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
