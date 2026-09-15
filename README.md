# ocr-worker —— OvisOCR2 RunPod Serverless worker

把 [ATH-MaaS/OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2)(0.8B 文档解析 VLM)
封装成一个 **队列式 Serverless 端点**,输入 PDF/图片,输出 Markdown。
DSH 侧由插件 `ocr-reports` 调用(见 `financial-analyst/packages/ocr-reports/`)。

## 为什么自建 worker,而不是直接用 RunPod 官方 vLLM worker

| 需求 | 官方 `runpod/worker-v1-vllm` | 本 worker |
| --- | --- | --- |
| 固定 vLLM 0.22.1(模型卡指定版本) | ✗ 跟随上游(当前 0.29.0) | ✓ 镜像里锁死 |
| PDF → 页面图像 | ✗ 只收文本/图片 | ✓ 内置 pypdfium2 |
| `gdn_prefill_backend=triton` | 需靠 env 名转 CLI flag,不保证 | ✓ 代码里显式设置 |
| 每页视觉分辨率预算(min/max_pixels) | 只能在启动参数里全局设 | ✓ 每个 job 可覆盖 |
| 冷启动 | 每次去 HF 拉 1.7GB 权重 | ✓ 权重烤进镜像 |

## 架构

```
DSH 插件 ocr-reports
   │  base64(PDF 或图片) / 文件 URL
   ▼
RunPod 队列端点  POST https://api.runpod.ai/v2/<ENDPOINT_ID>/run
   │  轮询 GET .../status/<JOB_ID>
   ▼
worker 容器
   ├─ pypdfium2:  PDF 指定页 → PNG(RENDER_DPI,默认 200)
   ├─ vLLM 0.22.1: Qwen3.5-0.8B + OvisOCR2 权重,贪心解码
   └─ 后处理:去 `<img src="images/bbox_…">` 占位、折叠截断长尾重复
   ▼
{ markdown, pages:[{page, markdown}], page_count, elapsed_ms, … }
```

## 文件

| 文件 | 作用 |
| --- | --- |
| `Dockerfile` | 基于 `vllm/vllm-openai:v0.22.1-cu129-ubuntu2404`,装 runpod/pypdfium2,烤入权重 |
| `rp_handler.py` | RunPod handler;引擎在导入时构建一次并复用(文件名与 RunPod 官方 worker-basic 一致) |
| `requirements.txt` | worker 运行期依赖 |
| `scripts/smoke_test.py` | 端点建好后打一发真实请求(仅用标准库) |
| `scripts/deploy_runpod.py` | 走 REST API 用**已推送到镜像仓库**的镜像建 template + endpoint |

## 请求 / 响应契约

请求(`POST /run` 的 `input`):

```json
{
  "input": {
    "file_b64": "<PDF 或图片的 base64>",
    "filename": "600519_2024_年报.pdf",
    "page_start": 12,
    "page_end": 30,
    "dpi": 200,
    "filter_img_tags": true
  }
}
```

`file_b64` 与 `file_url` 二选一。不带任何文件字段时返回健康信息(不触发推理):

```json
{ "ok": true, "model": "ATH-MaaS/OvisOCR2", "engine_ready": true }
```

响应:

```json
{
  "model": "ATH-MaaS/OvisOCR2",
  "page_count": 312, "pages_returned": 4, "page_numbers": [12,13,14,15],
  "truncated": true,
  "pages": [{ "page": 12, "markdown": "..." }],
  "markdown": "<!-- page 12 -->\n...",
  "chars": 8421, "infer_ms": 9120, "elapsed_ms": 11030
}
```

## 可调环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `MODEL_NAME` | `ATH-MaaS/OvisOCR2` | 权重路径或 HF repo |
| `GDN_PREFILL_BACKEND` | `triton` | Qwen3.5 线性注意力 prefill 后端;不被当前 vLLM 接受时自动降级 |
| `GPU_MEMORY_UTILIZATION` | `0.85` | 显存占用比例;卡小就下调 |
| `MAX_MODEL_LEN` | 不设(用模型默认) | 上下文上限 |
| `MAX_OUTPUT_TOKENS` | `16384` | 单页最大输出 token,与模型卡一致 |
| `MIN_PIXELS` / `MAX_PIXELS` | `448²` / `2880²` | 视觉编码像素预算,与模型卡一致 |
| `RENDER_DPI` | `200` | PDF 渲染 DPI;调高更清晰但更慢更吃显存 |
| `MAX_PAGES_PER_JOB` | `40` | 单个 job 页数上限,防止打满 execution timeout |
| `FILTER_IMG_TAGS` | `true` | 是否丢掉图表占位块(留正文与表格) |

---

## 部署步骤

### 前置

- 一个 RunPod 账号(需要余额:Serverless 按秒计费)
- 一个 GitHub 账号(方式 A)—— **本机不需要装 Docker**

### 方式 A:RunPod GitHub 集成构建(推荐,本机无 Docker 也能用)

RunPod 会把仓库里的代码拉过去、用你的 Dockerfile 构建、存进它自己的镜像仓库,再挂到端点上。

1. 把本目录推到 GitHub(仓库根可以是任意目录,只要 Dockerfile 路径对得上):

   ```powershell
   cd F:\code\dsh\ocr-worker
   git init; git add -A; git commit -m "OvisOCR2 RunPod serverless worker"
   git remote add origin https://github.com/<你>/<repo>.git
   git push -u origin main
   ```

2. 授权 RunPod 访问 GitHub:<https://console.runpod.io/user/settings> → **Connections** → **GitHub** → **Connect**,
   选择刚推的仓库。

3. 打开 <https://console.runpod.io/serverless/new-endpoint> → **Import Git Repository** → 选仓库
   → **Branch** 填 `main` → **Dockerfile Path** 填 `Dockerfile` → Next。

4. 按下面的「端点参数建议」填设置。注意 **Container Disk 要给够**(镜像内已含 1.7GB 权重,
   基础镜像本身就很大),否则构建/启动会因磁盘不足失败。

5. **Deploy Endpoint**。构建 10–25 分钟(基础镜像 ~16GB + 权重下载),在端点的 **Builds** 标签看进度:
   Pending → Building → Uploading → Testing → Completed。

> 构建只跑一次;之后改代码 push 到该分支,端点可以重新构建(见 RunPod 文档 "Update your endpoint")。

### 方式 B:本地 Docker 构建后推送

本机当前**没装 Docker**。要走路 B,先装 Docker Desktop,然后:

```powershell
cd F:\code\dsh\ocr-worker
docker build -t <你的dockerhub用户名>/ovisocr2-worker:0.22.1 .
docker push <你的dockerhub用户名>/ovisocr2-worker:0.22.1
```

推完在 <https://console.runpod.io/serverless/new-endpoint> 选 **Docker Image** 填该镜像名;
或者用脚本自动化:

```powershell
$env:RUNPOD_API_KEY = "rpa_..."
python scripts\deploy_runpod.py --image <你的dockerhub用户名>/ovisocr2-worker:0.22.1
```

### 端点参数建议

| 设置 | 建议值 | 理由 |
| --- | --- | --- |
| Endpoint Type | **Queue** | OCR 是长任务,要 `/run` + 轮询;`/runsync` 只适合秒级 |
| GPU | 24GB 档:L4 / RTX A5000 / RTX 4090 / A40 | 0.8B 权重仅 1.7GB,但单页最长边 2880px 的视觉 token 很吃显存;16GB 可试但要下调 `GPU_MEMORY_UTILIZATION` |
| GPUs per worker | 1 | 用不到张量并行 |
| Container Disk | **50 GB** | 基础镜像 + 权重;给小了会启动失败 |
| Max Workers | 3 起 | 有并发财报解析需求再加大 |
| Active Workers | 0 | 不常跑就 0,只在冷启动付费 |
| Idle Timeout | 120 s | vLLM 冷启动要 60–120s,太短会反复重建引擎 |
| Execution Timeout | 1200 s | 40 页 × 数秒/页 + 渲染,600s 不够 |
| FlashBoot | 开启 | 加速冷启动 |

### 验证

```powershell
$env:RUNPOD_API_KEY = "rpa_..."
python scripts\smoke_test.py --endpoint <ENDPOINT_ID> --health
python scripts\smoke_test.py --endpoint <ENDPOINT_ID> --file "E:\data\cn\financial_reports\<某份年报>.pdf" --pages 12-15
```

看到 Markdown 输出即成功。首次调用会等冷启动(1–3 分钟),之后复用热 worker。

## 接到 DSH

端点建好后,把 `ENDPOINT_ID` 写进环境变量 `RUNPOD_OCR_ENDPOINT_ID`,
`RUNPOD_API_KEY` 走环境变量(不要写进 `agent.cordis.yml`),然后按
`financial-analyst/packages/ocr-reports/README.md` 安装插件。

## 成本与调优

- 计费按 worker 运行秒数;**引擎构建 + 权重加载都算冷启动时间**,所以 Idle Timeout 别设太短
  (设 10s 会导致每个请求都重新加载 1.7GB 权重)。
- 批量场景:一个 job 塞多页(插件默认 8 页/请求)比一页一请求划算得多。
- 想进一步省冷启动:改用 RunPod **Network Volume** 缓存 HF 权重目录,
  把 `HF_HOME` 指到挂载点,镜像里就不用烤权重(镜像能小 1.7GB)。

## 升级 vLLM 的注意

`Dockerfile` 固定 `v0.22.1`,因为模型卡就是在这个版本上验证的。要升级:

1. 先确认新版本 `vllm/vllm-openai:<tag>` 存在;
2. 确认新版本仍然支持 `Qwen3_5ForConditionalGeneration`;
3. 改 tag 重新构建,用 `smoke_test.py` 对**同一页**做前后对比 ——
   这类后训练模型对推理栈版本很敏感,输出格式(表格/公式)可能变。
