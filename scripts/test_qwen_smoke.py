"""Short, opt-in local vision inference check for the bundled Qwen model."""

from __future__ import annotations

import json
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from auto_mode import stage5_qwen_scene_worker as worker


def check_result(raw: str) -> dict:
    assert "<think>" not in raw and "</think>" not in raw, raw
    data = json.loads(raw)
    assert set(data) == set(worker.SEMANTIC_SCHEMA["required"]), data
    for key in worker.NUMERIC_KEYS:
        assert type(data[key]) in (int, float) and 0 <= data[key] <= 1, (key, data)
    assert data["emotion"] in worker.ALLOWED_EMOTIONS, data
    assert data["recommended_use"] in worker.ALLOWED_USES, data
    assert type(data["visible_text"]) is bool, data
    assert isinstance(data["description"], str) and data["description"].strip(), data
    assert worker._normalize_semantic(data), data
    return data


def main(report_dir=None) -> None:
    report_dir = Path(report_dir) if report_dir is not None else worker.ROOT_DIR / "output" / "qwen_smoke_test"
    report_dir.mkdir(parents=True, exist_ok=True)
    paths = worker.LlamaPaths(None)
    paths.validate()
    device, device_label = worker._select_vulkan_device(paths)
    plain = Image.new("RGB", (512, 320), "white")
    ImageDraw.Draw(plain).rectangle((150, 70, 360, 270), fill="red")
    title = Image.new("RGB", (512, 320), "white")
    font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 48)
    ImageDraw.Draw(title).text((35, 110), "BEATSYNC TEST", fill="black", font=font)
    prompt = worker._build_prompt({})
    responses = []
    http_json = worker._http_json

    def capture(method, url, payload=None, timeout=30.0):
        data = http_json(method, url, payload, timeout)
        if url.endswith("/v1/chat/completions"):
            message = data["choices"][0]["message"]
            assert not message.get("reasoning_content"), message
            assert data["choices"][0]["finish_reason"] == "stop", data
            responses.append(data)
        return data

    worker._http_json = capture
    server = None
    results = []
    started = time.perf_counter()
    try:
        server = worker.LlamaServerClient(paths, device, 8192, str(report_dir / "server.json"), 1)
        for name, image, expected_text in [("plain_shape", plain, False), ("readable_title", title, True)]:
            inference_started = time.perf_counter()
            data = check_result(server.generate(image, prompt))
            assert data["visible_text"] is expected_text, (name, data)
            results.append({"test": name, "backend": "server", "seconds": round(time.perf_counter() - inference_started, 2), "semantic": data})
            print("PASS", name, json.dumps(data), flush=True)
        load_seconds = server.load_seconds
    finally:
        if server:
            server.close()
        worker._http_json = http_json

    cli = worker.LlamaMtmdClient(paths, device, 8192, str(report_dir / "cli.json"))
    inference_started = time.perf_counter()
    raw = cli.generate(title, prompt, "readable_title")
    data = check_result(raw)
    assert data["visible_text"] is True, data
    results.append({"test": "cli_readable_title", "backend": "cli", "seconds": round(time.perf_counter() - inference_started, 2), "semantic": data})
    print("PASS CLI fallback", json.dumps(data), flush=True)
    report = {"model": paths.model.name, "projector": paths.mmproj.name, "thinking": False, "device": device_label, "server_load_seconds": round(load_seconds, 2), "total_seconds": round(time.perf_counter() - started, 2), "results": results, "server_responses": responses}
    report_path = report_dir / "results.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Report:", report_path, flush=True)


if __name__ == "__main__":
    main()
