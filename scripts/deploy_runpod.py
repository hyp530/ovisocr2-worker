#!/usr/bin/env python3
"""用 RunPod REST API 创建 serverless template + endpoint(仅标准库)。

前提:镜像**已经**推到了某个 registry(走 GitHub 集成构建时不适用 —— 那条路只能在
console 里操作,因为 REST API 没有 GitHub 源字段)。见 README「方式 B」。

用法:
  set RUNPOD_API_KEY=rpa_xxx
  python deploy_runpod.py --image myuser/ovisocr2-worker:0.22.1 --name ovisocr2-ocr
  python deploy_runpod.py --image ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

API = "https://rest.runpod.io/v1"

DEFAULT_GPUS = [
    "NVIDIA L4",
    "NVIDIA RTX A5000",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA A40",
]

DEFAULT_ENV = {
    "MODEL_NAME": "ATH-MaaS/OvisOCR2",
    "GDN_PREFILL_BACKEND": "triton",
    "GPU_MEMORY_UTILIZATION": "0.85",
    "RENDER_DPI": "200",
    "MAX_PAGES_PER_JOB": "40",
    "FILTER_IMG_TAGS": "true",
}


def call(path: str, api_key: str, payload=None, method: str = "POST"):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(f"{API}{path}", data=data, method=method)
    req.add_header("authorization", f"Bearer {api_key}")
    req.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise SystemExit(f"HTTP {exc.code} {method} {path}\n{body}") from exc


def main() -> int:
    ap = argparse.ArgumentParser(description="Create RunPod template + serverless endpoint")
    ap.add_argument("--image", required=True, help="已推送的镜像名,如 user/repo:tag")
    ap.add_argument("--name", default="ovisocr2-ocr", help="端点/模板名前缀")
    ap.add_argument("--gpus", nargs="*", default=DEFAULT_GPUS, help="可选 GPU 型号(按优先级)")
    ap.add_argument("--max-workers", type=int, default=3)
    ap.add_argument("--idle-timeout", type=int, default=120, help="秒")
    ap.add_argument("--execution-timeout", type=int, default=1200, help="秒")
    ap.add_argument("--container-disk", type=int, default=50, help="GB")
    ap.add_argument("--dry-run", action="store_true", help="只打印将要提交的 payload")
    args = ap.parse_args()

    template_body = {
        "name": f"{args.name}-template",
        "imageName": args.image,
        "isServerless": True,
        "containerDiskInGb": args.container_disk,
        "category": "NVIDIA",
        "env": DEFAULT_ENV,
    }
    endpoint_body = {
        "name": args.name,
        "computeType": "GPU",
        "gpuTypeIds": args.gpus,
        "gpuCount": 1,
        "workersMin": 0,
        "workersMax": args.max_workers,
        "idleTimeout": args.idle_timeout,
        "executionTimeout": args.execution_timeout,
        "scalerType": "QUEUE_DELAY",
        "scalerValue": 4,
    }

    if args.dry_run:
        print("POST /templates")
        print(json.dumps(template_body, ensure_ascii=False, indent=2))
        print("\nPOST /endpoints  (templateId 来自上一步)")
        print(json.dumps(endpoint_body, ensure_ascii=False, indent=2))
        return 0

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("缺少环境变量 RUNPOD_API_KEY", file=sys.stderr)
        return 1

    print(f"[1/2] 创建 template ...")
    tpl = call("/templates", api_key, template_body)
    tpl_id = tpl.get("id")
    print(f"      templateId={tpl_id}")
    if not tpl_id:
        print(json.dumps(tpl, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1

    endpoint_body["templateId"] = tpl_id
    print(f"[2/2] 创建 endpoint ...")
    ep = call("/endpoints", api_key, endpoint_body)
    ep_id = ep.get("id")
    print(f"\n完成。Endpoint ID = {ep_id}")
    print("把下面两行配到 DSH 侧(API key 走环境变量,不要写进文件):")
    print(f'  $env:RUNPOD_OCR_ENDPOINT_ID = "{ep_id}"')
    print('  $env:RUNPOD_API_KEY = "rpa_..."')
    print(f"\n冒烟测试: python smoke_test.py --endpoint {ep_id} --health")
    return 0 if ep_id else 1


if __name__ == "__main__":
    raise SystemExit(main())
