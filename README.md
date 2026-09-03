# 🧞 GSV-Genie-Dubbing

**输入字幕文件，用 Genie（genie-tts，GPT-SoVITS 轻量 ONNX 推理引擎）逐句自动配音。**

加载 `.srt` / `.ass` / `.vtt` 字幕后，浏览器中逐句展示台词 —— 文本可编辑、单句可重新生成、试听、下载，全部满意后按字幕时间轴合并导出整轨 WAV。支持断点续跑（关掉再来，已合成的句子自动复用）。

> 双引擎支持：**Genie**（推荐，pip install genie-tts 即可推理，无需部署完整 GPT-SoVITS）与 **GPT-SoVITS api_v2**（备用，适合已有 API 服务的场景），见下方「引擎切换」。

## ✨ 功能特性

- **Genie ONNX 推理**：走 `genie_tts`（GPT-SoVITS → ONNX 转换后轻量部署），进程内直调或 HTTP 两种模式
- **逐句配音**：每句独立合成，单句可反复重生成直到满意
- **前端逐句编辑**：改台词文本后重新生成该句，不影响其他句子
- **单句播放 / 下载**：浏览器里直接试听、下载任意一句的 WAV
- **时间轴自适应**：音频比字幕窗口长时自动时长压缩（上限可设）
- **整轨合并导出**：按字幕时间码铺设静音与语音，重叠句自动交叉淡化防爆音
- **断点续跑**：每句落盘缓存 + 会话状态 JSON，中断后重跑自动跳过已完成句子
- **多说话人**：ASS 的 Name 字段或「名字：」前缀路由到不同 Genie 角色/参考音频（profile JSON 配置）
- **字幕清洗**：自动去除【旁白】、`{\i1}` ASS 覆盖标签、HTML 标签、♪ 音乐行、纯符号行

## 🚀 快速开始（Genie 引擎）

### 1. 安装 Genie

```bash
pip install genie-tts
```

首次使用需准备 **ONNX 模型目录**（角色 = ONNX 模型 + 参考音频）。用已有 GPT-SoVITS 权重转换：

```python
import genie_tts as genie
genie.convert_to_onnx(
    torch_ckpt_path="GPT_weights_v2ProPlus/Yukikaze-e15.ckpt",   # GPT 权重
    torch_pth_path="SoVITS_weights_v2ProPlus/Yukikaze_e8_s704.pth",  # SoVITS 权重
    output_dir="onnx_models",  # 输出的 ONNX 目录（界面里填这个）
)
```

### 2. 启动本工具

```bash
python server_genie.py                # 进程内直调 genie_tts（推荐，最简）
python server_genie.py --port 8766    # 自定义端口
```

浏览器打开 **http://127.0.0.1:8766**

依赖：`Python 3.10+`，`fastapi`、`uvicorn`、`numpy`、`genie-tts`

### 3. 使用流程

1. **检测 Genie** —— 绿点 = 可用（会自动显示当前引擎模式）
2. **Genie 角色** —— 填角色名（任意）+ 语言（中/日/英）+ ONNX 模型目录 + 参考音频 + 参考文本
   > 参考音频文本必须与音频内容完全一致（Genie 音色克隆的依据）
3. **加载字幕** —— 本机路径或上传
4. **开始配音** —— 批量逐句合成，实时进度条
5. **逐句打磨** —— 点击文本直接编辑 → 「⟳ 生成」重出该句 → 「▶ 播放」试听 → 不满意再来
6. **合并导出** —— 「合并导出」后「⬇ 下载整轨」；单句也可单独下载

### HTTP 模式（Genie 服务独立部署时）

若 Genie 服务单独跑（`genie_tts.start_server()` 或本仓库 `tests/mock_genie_api.py` 的真实版）：

```bash
python server_genie.py --engine http --genie-url http://127.0.0.1:8000
```

此时参考音频/ONNX 路径需为 **Genie 服务器** 可访问的本机路径。

## 🔀 引擎切换：GPT-SoVITS api_v2 备用后端

已有 GPT-SoVITS api_v2 服务（`python api_v2.py`）时可用原版后端，界面功能完全相同：

```bash
python server.py    # http://127.0.0.1:8765，填参考音频/GPT/SoVITS 权重
```

| | Genie（server_genie.py） | GPT-SoVITS api_v2（server.py） |
|---|---|---|
| 部署 | `pip install genie-tts` | 完整 GPT-SoVITS 环境 |
| 模型 | ONNX 目录（convert_to_onnx 转换） | 原生 ckpt/pth 权重 |
| 音色 | 参考音频 + 文本 | 参考音频 + 文本 + 权重切换 |
| 语速 | 无原生参数（自动时长压缩） | 原生 speed_factor |
| 语言 | 中文/日语/英语 | api_v2 支持全集 |

## 🎭 多说话人

ASS 字幕按 Name 字段、SRT 按行首「名字：」前缀识别说话人，参考 [`profile_genie.example.json`](profile_genie.example.json)（Genie）或 [`profile.example.json`](profile.example.json)（api_v2）：

```json
{
  "default_speaker": {
    "genie_character": "yukikaze",
    "genie_onnx_dir": "D:/onnx_models/Yukikaze",
    "genie_ref_audio": "D:/ref/yukikaze.wav",
    "genie_ref_text": "参考音频的完整文本"
  },
  "speakers": {
    "旁白": { "genie_character": "narrator", "genie_ref_audio": "D:/ref/narrator.wav", "genie_ref_text": "……" }
  }
}
```

说话人名支持 `fnmatch` 通配。朗读时自动去掉「名字：」前缀。

## 🔧 工作原理

```
字幕文件 ──解析/清洗──> 句子列表 ──逐句──> Genie load_character → set_reference_audio → tts
                                   │            ↑ 超窗自动时长压缩
                                   ▼
                        单句 WAV 缓存 (workspace/<会话>/clips/)
                                   │
                                   ▼
                    按时间轴铺设 + 交叉淡化 → 整轨 WAV
```

- Genie 三步调用：**load_character**（ONNX 目录）→ **set_reference_audio**（音色克隆）→ **tts**（`split_sentence=False` 保证整句一次出）
- 双后端统一接口：`gsv_dubbing/genie_client.py`（`GenieLocalClient` 进程内 / `GenieHTTPClient` HTTP）
- 重叠句在重叠区做线性交叉淡化，避免叠加爆音

## 📁 项目结构

```
server_genie.py    Web 服务端 — Genie 引擎（主入口）
server.py          Web 服务端 — GPT-SoVITS api_v2 引擎（备用）
web/               前端（原生 HTML/CSS/JS，无构建；Genie 版与 api_v2 版各一套）
gsv_dubbing/       核心库
  ├ genie_client.py       Genie 双后端客户端（local / http）
  ├ subtitle_parser.py   SRT/ASS/VTT 解析与清洗
  ├ speaker_router.py    多说话人路由（兼容两种引擎的说话人配置）
  ├ gsv_client.py        GPT-SoVITS api_v2 客户端（备用引擎）
  ├ engine.py            api_v2 版配音引擎（批量/断点/语速自适应）
  ├ audio_utils.py       重采样/时间轴合成/导出
  └ session_state.py     会话状态（断点续跑）
dubbing_cli.py     命令行界面（api_v2 后端）
dubbing_gui.py     Tkinter 图形界面（api_v2 后端）
tests/             单元测试 + E2E 测试 + mock（Genie / api_v2 各一）
```

## 🧪 测试

```bash
python -m unittest tests.test_core          # 单元测试（解析/路由/音频管线）
python tests/mock_genie_api.py --port 8000   # 模拟 Genie HTTP 服务（无模型也能测）
python server_genie.py --port 8766 --engine http &
python tests/e2e_genie_test.py               # Genie 后端端到端（8 步全链路）

# api_v2 备用后端
python tests/mock_gsv_api.py --port 9880 &
python tests/e2e_api_test.py
```

已验证链路：字幕解析过滤 → 单句编辑重生成 → 批量合成 → 断点续跑 → 时间轴合并导出 → 浏览器逐句播放/下载。

## ❓ 常见问题

**Q: 「Genie 不可用」？**
`python -c "import genie_tts"` 试一下。未装则 `pip install genie-tts`；或改用 `--engine http` 对接已运行的 Genie 服务。首次运行 genie_tts 需下载 GenieData 资源（会自动提示）。

**Q: ONNX 模型从哪来？**
`genie.convert_to_onnx(torch_ckpt_path=..., torch_pth_path=..., output_dir=...)` 转换现有 GPT-SoVITS V2/v2ProPlus 权重；或用 `genie.load_predefined_character()` 的内置角色。

**Q: 参考音频文本为什么要"完全准确"？**
它是 Genie（GPT-SoVITS 系）音色克隆的提示依据，文本与音频对不上会明显劣化效果。

**Q: 中断后如何继续？**
同一字幕重新「开始配音」，已完成句子自动复用缓存。

**Q: 想要原生语速控制/更多语言？**
用备用后端 `python server.py`（api_v2 有 speed_factor 与更全的语言支持）。

---

基于 [Genie (genie-tts)](https://pypi.org/project/genie-tts/) —— GPT-SoVITS 轻量级 ONNX 推理引擎构建。
