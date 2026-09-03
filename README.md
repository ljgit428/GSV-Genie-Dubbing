# 🎙️ GSV-Genie-Dubbing

**输入字幕文件，用 GPT-SoVITS 逐句自动配音。**

加载 `.srt` / `.ass` / `.vtt` 字幕后，浏览器中逐句展示台词 —— 文本可编辑、单句可重新生成、试听、下载，全部满意后按字幕时间轴合并导出整轨 WAV。支持断点续跑（关掉再来，已合成的句子自动复用）。

## ✨ 功能特性

- **逐句配音**：每句独立调用 GPT-SoVITS（api_v2），单句可反复重生成直到满意
- **前端逐句编辑**：改台词文本后重新生成该句，不影响其他句子
- **单句播放 / 下载**：浏览器里直接试听、下载任意一句的 WAV
- **时间轴自适应**：合成的音频比字幕窗口长时自动按比例提速重试（上限可设）
- **整轨合并导出**：按字幕时间码铺设静音与语音，重叠句自动交叉淡化防爆音
- **断点续跑**：每句落盘缓存 + 会话状态 JSON，中断后重跑自动跳过已完成句子
- **多说话人**：ASS 的 Name 字段或「名字：」前缀路由到不同参考音频/权重（profile.json 配置）
- **字幕清洗**：自动去除【旁白】、`{\i1}` ASS 覆盖标签、HTML 标签、♪ 音乐行、纯符号行

## 🚀 快速开始

### 1. 启动 GPT-SoVITS API（v2 版接口）

在 GPT-SoVITS 项目目录（api_v2.py 所在处）：

```bash
python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
```

### 2. 启动本工具

```bash
python server.py            # 默认 http://127.0.0.1:8765
python server.py --port 9000 # 自定义端口
```

浏览器打开 **http://127.0.0.1:8765**

依赖：`Python 3.10+`，`fastapi`、`uvicorn`、`numpy`（`pip install fastapi uvicorn numpy`）

### 3. 使用流程

1. **测试连接** —— 确认 GPT-SoVITS API 在线（绿点）
2. **音色设置** —— 填参考音频 / 参考文本 /（可选）GPT & SoVITS 权重路径
   > ⚠️ 路径必须是 **GPT-SoVITS 服务器可访问的本机路径**（如 `D:/modelscope/...`），不是本项目里的路径
3. **加载字幕** —— 填服务器上的字幕路径，或点「上传文件…」
4. **开始配音** —— 批量逐句合成，实时进度条
5. **逐句打磨** —— 点击任意句文本直接编辑 → 「⟳ 生成」重出该句 → 「▶ 播放」试听 → 不满意再来
6. **合并导出** —— 「合并导出」后「⬇ 下载整轨」；单句也可单独下载

## 🖥️ CLI 用法（可选）

不想开浏览器时可用命令行，同样支持断点续跑：

```bash
# 预览解析结果（不合成）
python dubbing_cli.py sub.srt --list

# 一键配音导出
python dubbing_cli.py sub.srt -o output \
    --ref "D:/参考音频/雪风.wav" --prompt "雪风的台词" \
    --gpt "GPT_weights_v2ProPlus/Yukikaze-e15.ckpt" \
    --sovits "SoVITS_weights_v2ProPlus/Yukikaze_e8_s704.pth"
```

另有 Tkinter 图形界面 `python dubbing_gui.py`（零第三方依赖）。

## 🎭 多说话人

ASS 字幕按 Name 字段、SRT 按行首「名字：」前缀识别说话人，参考 [`profile.example.json`](profile.example.json)：

```json
{
  "default_speaker": { "ref_audio": "…", "prompt_text": "…", "speed": 1.0 },
  "speakers": {
    "雪风": { "ref_audio": "D:/ref/yukikaze.wav", "prompt_text": "雪风的台词", "speed": 1.0 },
    "旁白": { "speed": 0.9 }
  }
}
```

speaker 名支持 `fnmatch` 通配（如 `yukikaze*`）。朗读时自动去掉「名字：」前缀。

## 🔧 工作原理

```
字幕文件 ──解析/清洗──> 句子列表 ──逐句──> GPT-SoVITS /tts (cut0 不切分)
                                   │            ↑ 超窗自动提速重试
                                   ▼
                        单句 WAV 缓存 (workspace/<会话>/clips/)
                                   │
                                   ▼
                    按时间轴铺设 + 交叉淡化 → 整轨 WAV
```

- 每句以 `cut0` 完整送入，保证句间独立可控
- 语速自适应：`新语速 = min(原语速 × 音频时长/字幕窗口, 上限)`
- 重叠句在重叠区做线性交叉淡化，避免叠加爆音

## 📁 项目结构

```
server.py          Web 服务端（FastAPI，REST API + 静态页）
web/               前端（原生 HTML/CSS/JS，无构建）
gsv_dubbing/       核心库
  ├ subtitle_parser.py   SRT/ASS/VTT 解析与清洗
  ├ gsv_client.py       GPT-SoVITS api_v2 HTTP 客户端
  ├ engine.py           配音引擎（批量/断点/语速自适应）
  ├ speaker_router.py   多说话人路由
  ├ audio_utils.py      重采样/时间轴合成/导出
  └ session_state.py    会话状态（断点续跑）
dubbing_cli.py     命令行界面
dubbing_gui.py     Tkinter 图形界面
tests/             单元测试 + E2E 测试 + mock API
```

## 🧪 测试

```bash
python -m unittest tests.test_core        # 单元测试（解析/路由/音频管线）
python tests/mock_gsv_api.py --port 9880  # 模拟 GPT-SoVITS（无 GPU 也能测）
python tests/e2e_api_test.py              # 端到端（需先启动 server + mock）
```

## ❓ 常见问题

**Q: 提示「ref_audio_path is required」？**
必须填参考音频。它是 GPT-SoVITS 推理的必需输入（音色来源）。

**Q: 路径填了还是失败？**
参考音频/权重/字幕路径都是 **GPT-SoVITS API 服务器视角** 的本机路径。若 API 部署在别机，填那台机器上的路径；本工具的 Web 页面随便哪台机开浏览器都行。

**Q: 想换一句的音色？**
改全局设置后点该句「⟳ 生成」即可 —— 单句重生成用当前面板参数。

**Q: 中断后如何继续？**
同一字幕重新「开始配音」，已完成的句子自动复用缓存，不会重复合成。

**Q: V3/V4 模型？**
`sample_steps` 参数已透传（默认 32）。V3/V4 权重路径填到对应输入框即可。

---

基于 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 的 api_v2 接口构建。
