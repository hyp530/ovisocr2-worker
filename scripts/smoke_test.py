#!/usr/bin/env python3
"""端点冒烟测试 —— 只依赖标准库。

用法:
  set RUNPOD_API_KEY=rpa_xxx
  python smoke_test.py --endpoint abc123 --health
  python smoke_test.py --endpoint abc123 --file report.pdf --pages 12-15 --dpi 200
  python smoke_test.py --endpoint abc123 --url https://example.com/a.pdf --page 3

退出码: 0 成功 / 1 失败。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

API_ROOT = os.environ.get("RUNPOD_API_ROOT", "https://api.runpod.ai/v2")
POLL_INTERVAL = 3.0


def _request(url: str, api_key: str, payload=None, method: str = "GET"):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("authorization", f"Bearer {api_key}")
    req.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise SystemExit(f"HTTP {exc.code} {url}\n{body}") from exc


def parse_pages(spec: str | None):
    if not spec:
        return None, None
    if "-" in spec:
        a, _, b = spec.partition("-")
        return int(a), int(b or a)
    return int(spec), int(spec)


def main() -> int:
    ap = argparse.ArgumentParser(description="RunPod OvisOCR2 endpoint smoke test")
    ap.add_argument("--endpoint", required=True, help="Endpoint ID")
    ap.add_argument("--file", help="本地文件路径(PDF 或图片)")
    ap.add_argument("--url", help="远端文件 URL(与 --file 二选一)")
    ap.add_argument("--pages", help='页码区间,如 "12-15" 或 "7"')
    ap.add_argument("--dpi", type=int, default=None)
    ap.add_argument("--health", action="store_true", help="只查端点/worker 状态")
    ap.add_argument("--timeout", type=float, default=1800.0, help="总轮询上限(秒)")
    ap.add_argument("--out", help="把 markdown 写到该文件")
    args = ap.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("缺少环境变量 RUNPOD_API_KEY", file=sys.stderr)
        return 1

    base = f"{API_ROOT}/{args.endpoint}"

    health = _request(f"{base}/health", api_key)
    print(f"[health] {json.dumps(health, ensure_ascii=False)}")
    if args.health:
        return 0

    if not args.file and not args.url:
        print("需要 --file 或 --url(或只加 --health)", file=sys.stderr)
        return 1

    payload: dict = {"input": {}}
    if args.file:
        with open(args.file, "rb") as fh:
            raw = fh.read()
        print(f"[input] {args.file}  {len(raw) / 1048576:.2f} MiB")
        payload["input"]["file_b64"] = base64.b64encode(raw).decode("ascii")
        payload["input"]["filename"] = os.path.basename(args.file)
    else:
        payload["input"]["file_url"] = args.url

    start_page, end_page = parse_pages(args.pages)
    if start_page:
        payload["input"]["page_start"] = start_page
    if end_page:
        payload["input"]["page_end"] = end_page
    if args.dpi:
        payload["input"]["dpi"] = args.dpi

    t0 = time.time()
    job = _request(f"{base}/run", api_key, payload, method="POST")
    job_id = job.get("id")
    print(f"[run] job={job_id} status={job.get('status')}")
    if not job_id:
        print(json.dumps(job, ensure_ascii=False), file=sys.stderr)
        return 1

    deadline = time.time() + args.timeout
    last = None
    while time.time() < deadline:
        time.sleep(POLL_INTERVAL)
        state = _request(f"{base}/status/{job_id}", api_key)
        status = state.get("status")
        if status != last:
            print(f"[status] {status}  ({time.time() - t0:.0f}s)")
            last = status
        if status == "COMPLETED":
            out = state.get("output") or {}
            print(f"[done] {time.time() - t0:.0f}s  pages={out.get('pages_returned')}"
                  f"/{out.get('page_count')}  chars={out.get('chars')}"
                  f"  infer_ms={out.get('infer_ms')}")
            md = out.get("markdown", "")
            print("-" * 70)
            print(md[:4000])
            if len(md) > 4000:
                print(f"... (共 {len(md)} 字符,已截断显示)")
            if args.out:
                with open(args.out, "w", encoding="utf-8") as fh:
                    fh.write(md)
                print(f"[saved] {args.out}")
            return 0
        if status in {"FAILED", "CANCELLED", "TIMED_OUT"}:
            print(json.dumps(state, ensure_ascii=False, indent=2), file=sys.stderr)
            return 1

    print("[timeout] 轮询超时", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
