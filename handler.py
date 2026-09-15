"""RunPod Serverless worker —— OvisOCR2 文档解析(页面图像 -> Markdown)。

模型: https://huggingface.co/ATH-MaaS/OvisOCR2  (Qwen3.5-0.8B 后训练,0.8B bf16)
依赖: vllm==0.22.1  —— 模型卡指定的版本;Qwen3_5 架构 + gdn_prefill_backend="triton"

设计要点
  1. vLLM 引擎在模块导入时构建一次,worker 存活期间复用(冷启动付费一次,之后每个 job 只付推理);
  2. **PDF -> 页面图像在 worker 侧完成**(pypdfium2),调用方只需把 PDF 原样发过来;
  3. 一个 job 可带多页,内部按 batch 送 vLLM,摊薄单页开销;
  4. 采样参数固定 greedy(temperature=0),与模型卡一致。

请求(input,JSON 对象)
  file_b64        str   文件内容 base64(PDF 或 png/jpg/webp)。与 file_url 二选一。
  file_url        str   由 worker 侧下载的文件 URL。与 file_b64 二选一。
  filename        str   可选;用于推断类型与回填响应。
  page_start      int   可选,PDF 起始页(1-based,含),默认 1。
  page_end        int   可选,PDF 结束页(含),默认到最后一页。
  dpi             int   可选,PDF 渲染 DPI,默认 RENDER_DPI(200)。
  filter_img_tags bool  可选,是否去掉 <img src="images/bbox_..."> 占位块,默认 True。
  min_pixels      int   可选,覆盖视觉编码下限像素。
  max_pixels      int   可选,覆盖视觉编码上限像素。
  max_tokens      int   可选,覆盖单页最大输出 token。

响应
  {
    "model": "ATH-MaaS/OvisOCR2",
    "filename": "...", "page_count": 312, "pages_returned": 8,
    "pages": [{"page": 12, "markdown": "..."}],
    "markdown": "<!-- page 12 -->\n...\n\n<!-- page 13 -->\n...",
    "truncated": false,
    "elapsed_ms": 12345
  }
"""

from __future__ import annotations

import base64
import binascii
import io
import os
import re
import time
import urllib.request

import runpod

# ── 配置(全部可用环境变量覆盖) ────────────────────────────────────────────
MODEL_NAME = os.environ.get("MODEL_NAME", "ATH-MaaS/OvisOCR2")
TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.85"))
MAX_MODEL_LEN = os.environ.get("MAX_MODEL_LEN")
GDN_PREFILL_BACKEND = os.environ.get("GDN_PREFILL_BACKEND", "triton")

MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "16384"))
MIN_PIXELS = int(os.environ.get("MIN_PIXELS", str(448 * 448)))
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", str(2880 * 2880)))
RENDER_DPI = int(os.environ.get("RENDER_DPI", "200"))

# 单个 job 最多解析多少页 —— 防止一次请求把 execution timeout 打满。
MAX_PAGES_PER_JOB = int(os.environ.get("MAX_PAGES_PER_JOB", "40"))
FILTER_IMG_TAGS_DEFAULT = os.environ.get("FILTER_IMG_TAGS", "true").lower() in ("1", "true", "yes", "on")
URL_TIMEOUT_SEC = int(os.environ.get("URL_TIMEOUT_SEC", "120"))

# 与模型卡逐字一致;不要改写 prompt,模型是在这个指令上后训练的。
PROMPT = (
    "\nExtract all readable content from the image in natural human reading order and "
    "output the result as a single Markdown document. For charts or images, represent "
    "them using an HTML image tag: <"
    'img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />'
    ", where left, top, right, bottom are bounding box coordinates scaled to [0, 1000). "
    "Format formulas as LaTeX. Format tables as HTML: <table>...</table>. Transcribe all "
    "other text as standard Markdown. Preserve the original text without translation or "
    "paraphrasing."
)

BBOX_IMG_PREFIX = '<img src="images/bbox_'

_engine = None
_sampling_params = None
_prompt_text = None


def _log(msg: str) -> None:
    print(f"[ovisocr2-worker] {msg}", flush=True)


# ── 引擎 ───────────────────────────────────────────────────────────────────
def _build_engine():
    """构建 vLLM 引擎与固定 prompt / 采样参数。"""
    global _engine, _sampling_params, _prompt_text
    from vllm import LLM, SamplingParams

    kwargs = {
        "model": MODEL_NAME,
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
    }
    if MAX_MODEL_LEN:
        kwargs["max_model_len"] = int(MAX_MODEL_LEN)
    if GDN_PREFILL_BACKEND:
        # Qwen3.5 的线性注意力(Gated Delta Net)prefill 后端。模型卡显式指定 triton。
        kwargs["gdn_prefill_backend"] = GDN_PREFILL_BACKEND

    started = time.time()
    try:
        llm = LLM(**kwargs)
    except TypeError as exc:
        # 不同 vLLM 版本可能没这个入参;退一步用默认后端,不因此让 worker 起不来。
        _log(f"gdn_prefill_backend 不被当前 vLLM 接受({exc});改用默认后端重试")
        kwargs.pop("gdn_prefill_backend", None)
        llm = LLM(**kwargs)
    _log(f"vLLM 引擎就绪,耗时 {time.time() - started:.1f}s (model={MODEL_NAME})")

    tokenizer = llm.get_tokenizer()
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    try:
        _prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        # 老模板不认识 enable_thinking;模型卡用 False,缺失时按模板默认走。
        _log("chat template 不支持 enable_thinking,使用默认")
        _prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    _sampling_params = SamplingParams(max_tokens=MAX_OUTPUT_TOKENS, temperature=0.0)
    _engine = llm
    return _engine


def _get_engine():
    if _engine is None:
        _build_engine()
    return _engine


# ── 输出清理(逐字移植模型卡实现) ─────────────────────────────────────────
def _clean_truncated_repeats(
    text: str,
    min_text_len: int = 8000,
    max_period: int = 200,
    min_period: int = 1,
    min_repeat_chars: int = 100,
    min_repeat_times: int = 5,
) -> str:
    """去掉被 max_tokens 截断后常见的长尾重复。"""
    n = len(text)
    if n < min_text_len:
        return text

    max_period = min(max_period, n - 1)
    for unit_len in range(min_period, max_period + 1):
        if text[n - 1] != text[n - 1 - unit_len]:
            continue

        match_len = 1
        idx = n - 2
        while idx >= unit_len and text[idx] == text[idx - unit_len]:
            match_len += 1
            idx -= 1

        total_len = match_len + unit_len
        repeat_times = total_len // unit_len
        tail_len = total_len % unit_len

        if repeat_times >= min_repeat_times and total_len >= min_repeat_chars:
            return text[: n - total_len + unit_len] + text[n - tail_len :]

    return text


def _filter_img_tags(text: str) -> str:
    return "\n\n".join(
        block for block in text.split("\n\n") if not block.strip().startswith(BBOX_IMG_PREFIX)
    )


# ── 输入解析 ───────────────────────────────────────────────────────────────
def _decode_b64(value: str) -> bytes:
    raw = value.strip()
    if raw.startswith("data:"):
        comma = raw.find(",")
        if comma >= 0:
            raw = raw[comma + 1 :]
    raw = re.sub(r"\s+", "", raw)
    try:
        return base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        # 容忍 URL-safe / 缺 padding 的变体。
        return base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))


def _fetch_url(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "dsh-ocr-worker/1.0"})
    with urllib.request.urlopen(req, timeout=URL_TIMEOUT_SEC) as resp:
        return resp.read()


def _is_pdf(data: bytes, filename: str) -> bool:
    if data[:5] == b"%PDF-":
        return True
    return filename.lower().endswith(".pdf")


def _render_pdf(data: bytes, page_start: int, page_end: int, dpi: int):
    """PDF -> [(页码, PIL.Image)]。页码 1-based。"""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(data)
    try:
        total = len(pdf)
        start = max(1, int(page_start or 1))
        end = total if not page_end else min(total, int(page_end))
        if end < start:
            raise ValueError(f"页码范围非法: page_start={start} > page_end={end}(共 {total} 页)")
        if end - start + 1 > MAX_PAGES_PER_JOB:
            end = start + MAX_PAGES_PER_JOB - 1

        scale = float(dpi) / 72.0
        out = []
        for index in range(start - 1, end):
            page = pdf[index]
            try:
                bitmap = page.render(scale=scale)
                try:
                    # convert() 会复制像素,之后关掉 bitmap 才安全。
                    image = bitmap.to_pil().convert("RGB")
                finally:
                    bitmap.close()
            finally:
                page.close()
            out.append((index + 1, image))
        return out, total
    finally:
        pdf.close()


def _load_page_images(inp: dict):
    """返回 ([(页码, PIL.Image)], pdf_total_pages 或 None, filename)。"""
    from PIL import Image

    filename = str(inp.get("filename") or "")

    if inp.get("file_b64"):
        data = _decode_b64(str(inp["file_b64"]))
    elif inp.get("file_url"):
        data = _fetch_url(str(inp["file_url"]))
        if not filename:
            filename = os.path.basename(str(inp["file_url"]).split("?")[0])
    elif inp.get("image_b64"):
        data = _decode_b64(str(inp["image_b64"]))
    else:
        raise ValueError("必须提供 file_b64 / file_url / image_b64 之一")

    if not data:
        raise ValueError("文件内容为空")

    if _is_pdf(data, filename):
        pages, total = _render_pdf(
            data,
            int(inp.get("page_start") or 1),
            int(inp.get("page_end") or 0),
            int(inp.get("dpi") or RENDER_DPI),
        )
        return pages, total, filename or "input.pdf"

    image = Image.open(io.BytesIO(data))
    return [(1, image.convert("RGB"))], None, filename or "input.image"


# ── handler ────────────────────────────────────────────────────────────────
def handler(job: dict) -> dict:
    started = time.time()
    inp = job.get("input") or {}

    # 健康/就绪探测:不带业务入参时直接回引擎状态,避免误触发推理。
    if not any(k in inp for k in ("file_b64", "file_url", "image_b64")):
        return {
            "ok": True,
            "model": MODEL_NAME,
            "engine_ready": _engine is not None,
            "max_pages_per_job": MAX_PAGES_PER_JOB,
            "render_dpi": RENDER_DPI,
            "note": "提供 file_b64 / file_url / image_b64 以执行 OCR",
        }

    engine = _get_engine()

    pages, pdf_total, filename = _load_page_images(inp)
    if not pages:
        raise ValueError("没有可解析的页面")

    page_numbers = [p for p, _ in pages]
    images = [im for _, im in pages]

    min_pixels = int(inp.get("min_pixels") or MIN_PIXELS)
    max_pixels = int(inp.get("max_pixels") or MAX_PIXELS)
    filter_img_tags = bool(inp.get("filter_img_tags", FILTER_IMG_TAGS_DEFAULT))
    max_tokens = int(inp.get("max_tokens") or MAX_OUTPUT_TOKENS)

    sampling = _sampling_params
    if max_tokens != MAX_OUTPUT_TOKENS:
        from vllm import SamplingParams

        sampling = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    vllm_inputs = [
        {
            "prompt": _prompt_text,
            "multi_modal_data": {"image": image},
            "mm_processor_kwargs": {
                "images_kwargs": {"min_pixels": min_pixels, "max_pixels": max_pixels}
            },
        }
        for image in images
    ]

    infer_started = time.time()
    outputs = engine.generate(vllm_inputs, sampling)
    infer_sec = time.time() - infer_started

    rendered = []
    for page_no, output in zip(page_numbers, outputs):
        text = output.outputs[0].text.strip()
        if filter_img_tags:
            text = _filter_img_tags(text)
        rendered.append({"page": page_no, "markdown": _clean_truncated_repeats(text)})

    markdown = "\n\n".join(f"<!-- page {item['page']} -->\n{item['markdown']}" for item in rendered)

    return {
        "model": MODEL_NAME,
        "filename": filename,
        "page_count": pdf_total,
        "pages_returned": len(rendered),
        "page_numbers": page_numbers,
        "truncated": bool(pdf_total and page_numbers[-1] < pdf_total),
        "pages": rendered,
        "markdown": markdown,
        "chars": len(markdown),
        "infer_ms": int(infer_sec * 1000),
        "elapsed_ms": int((time.time() - started) * 1000),
    }


if __name__ == "__main__":
    # 预热:把引擎构建放在 runpod 启动之前,冷启动时间计入 worker 初始化而不是首个 job。
    if os.environ.get("SKIP_ENGINE_WARMUP", "").lower() not in ("1", "true", "yes"):
        _get_engine()
    runpod.serverless.start({"handler": handler})
