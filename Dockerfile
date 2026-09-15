# OvisOCR2 —— RunPod Serverless worker
#
# 基础镜像用 vLLM 官方镜像并**固定 0.22.1**:模型卡指定的版本,且带 cu129 运行时的
# Qwen3_5(Gated Delta Net 线性注意力)支持。不要随手升级,先按 README 的两步验证跑通。
FROM vllm/vllm-openai:v0.22.1-cu129-ubuntu2404

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MODEL_NAME=ATH-MaaS/OvisOCR2 \
    GDN_PREFILL_BACKEND=triton \
    RENDER_DPI=200 \
    MAX_PAGES_PER_JOB=40 \
    FILTER_IMG_TAGS=true

# worker 侧依赖:runpod SDK + PDF 渲染(pypdfium2,自带 pdfium 二进制,无需 apt)+ PIL
COPY requirements.txt /requirements.txt
RUN python3 -m pip install --no-cache-dir -r /requirements.txt

# 把模型权重烤进镜像:冷启动只拉镜像,不再每次去 Hugging Face 下载 1.7GB 权重。
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('ATH-MaaS/OvisOCR2', ignore_patterns=['*.png', '.eval_results/*'])"

COPY handler.py /handler.py

# vLLM 官方镜像的 ENTRYPOINT 是 openai api_server,必须覆盖成我们的 handler。
ENTRYPOINT ["python3", "-u", "/handler.py"]
