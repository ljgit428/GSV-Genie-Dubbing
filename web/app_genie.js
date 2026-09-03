/* GSV-Genie-Dubbing 前端逻辑（Genie 引擎版）
 * 与 app.js 同构，参数面板换成 Genie 的角色/ONNX/参考音频。
 */

const $ = (id) => document.getElementById(id);

const state = {
  sessionId: null,
  lines: [],
  running: false,
  pollTimer: null,
  edits: {},
  audioObj: null,
  playingIdx: null,
};

function toast(msg, isError = false) {
  let el = $("toast");
  if (!el) {
    el = document.createElement("div");
    el.id = "toast";
    document.body.appendChild(el);
  }
  el.textContent = msg;
  el.className = "show" + (isError ? " error" : "");
  clearTimeout(el._t);
  el._t = setTimeout(() => (el.className = ""), isError ? 5000 : 2500);
}

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch {}
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  const ct = res.headers.get("content-type") || "";
  return ct.includes("json") ? res.json() : res;
}

function collectParams() {
  return {
    character_name: $("character-name").value.trim() || "my_character",
    onnx_dir: $("onnx-dir").value.trim(),
    language: $("language").value,
    ref_audio: $("ref-audio").value.trim(),
    ref_text: $("ref-text").value.trim(),
    speaker_profile: $("speaker-profile").value.trim(),
    max_speed: parseFloat($("max-speed").value) || 1.4,
    retry: parseInt($("retry").value) || 2,
    fit_timeline: $("fit-timeline").checked,
    strip_brackets: $("strip-brackets").checked,
    only_cjk: $("only-cjk").checked,
  };
}

/* ───────────────────────── Genie 检测 ───────────────────────── */

$("btn-ping").onclick = async () => {
  const dot = $("ping-dot");
  dot.className = "dot off";
  try {
    const r = await api("/api/ping");
    dot.className = "dot " + (r.ok ? "on" : "bad");
    dot.title = r.ok ? `Genie 可用（${r.engine === "http" ? "HTTP 服务" : "进程内"}）` : "不可用";
    $("engine-tag").textContent = "引擎: " + (r.engine === "http" ? "HTTP 服务" : "进程内");
    toast(r.ok ? `Genie 可用 ✓（${r.engine}）` : "Genie 不可用（未装 genie-tts 或服务未启动）", !r.ok);
  } catch (e) {
    dot.className = "dot bad";
    toast(e.message, true);
  }
};

/* ───────────────────────── 字幕加载 ───────────────────────── */

$("btn-load").onclick = loadFromPath;
$("subtitle-path").addEventListener("keydown", (e) => e.key === "Enter" && loadFromPath());
$("btn-upload").onclick = () => $("file-input").click();
$("file-input").onchange = async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  // Genie 服务器与本工具通常同机，直接读文件名提示用户填路径；简单起见仍走上传到 workspace
  const buf = await f.arrayBuffer();
  const name = encodeURIComponent(f.name);
  const r = await api(`/api/subtitle?name=${name}`, {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: buf,
  });
  $("subtitle-path").value = r.path;
  await openSession(r.path);
};

async function loadFromPath() {
  const p = $("subtitle-path").value.trim();
  if (!p) return toast("请填写字幕路径或上传文件", true);
  await openSession(p);
}

async function openSession(path) {
  try {
    const r = await api("/api/subtitle/file", {
      method: "POST",
      body: JSON.stringify({ path, params: collectParams() }),
    });
    state.sessionId = r.session_id;
    state.lines = r.lines;
    state.edits = {};
    $("workspace").hidden = false;
    $("session-tag").hidden = false;
    $("session-tag").textContent = `会话 ${r.session_id} · ${r.total} 句 · 缓存 ${r.done} 句`;
    $("sub-info").hidden = false;
    $("sub-info").textContent = `已加载 ${path}：${r.total} 句可朗读，其中 ${r.done} 句已有缓存可复用`;
    renderLines();
    updateProgress(r.done, r.total);
    toast(`字幕加载完成：${r.total} 句`);
    $("workspace").scrollIntoView({ behavior: "smooth" });
  } catch (e) {
    toast(e.message, true);
  }
}

/* ───────────────────────── 渲染 ───────────────────────── */

function renderLines() {
  const wrap = $("lines");
  wrap.innerHTML = "";
  const kw = $("filter-input").value.trim().toLowerCase();
  const st = $("filter-status").value;

  for (const l of state.lines) {
    if (kw && !l.text.toLowerCase().includes(kw)) continue;
    if (st && l.status !== st) continue;

    const div = document.createElement("div");
    div.className = `line ${l.status}`;
    div.id = `line-${l.index}`;

    const idx = document.createElement("span");
    idx.className = "idx";
    idx.textContent = `#${l.index}`;

    const time = document.createElement("span");
    time.className = "time";
    time.textContent = l.time;

    const mid = document.createElement("div");
    mid.className = "text";

    if (l.speaker) {
      const spk = document.createElement("span");
      spk.className = "speaker";
      spk.textContent = l.speaker;
      mid.appendChild(spk);
    }

    const ta = document.createElement("textarea");
    ta.value = l.text;
    ta.rows = 1;
    ta.addEventListener("input", () => {
      ta.classList.add("dirty");
      state.edits[l.index] = { ...(state.edits[l.index] || {}), text: ta.value };
    });
    mid.appendChild(ta);

    const actions = document.createElement("div");
    actions.className = "actions";
    const btnGen = mkBtn("⟳ 生成", () => regenLine(l.index, ta.value));
    const btnPlay = mkBtn("▶ 播放", () => playClip(l.index));
    const btnDl = mkBtn("⬇", () => downloadClip(l.index));
    const btnSkip = mkBtn("跳过", function () { toggleSkip(l.index, this); });
    if (l.status !== "done") { btnPlay.disabled = true; btnDl.disabled = true; }
    if (state.edits[l.index]?.skip) btnSkip.classList.add("active");
    actions.append(btnGen, btnPlay, btnDl, btnSkip);
    mid.appendChild(actions);

    const status = document.createElement("span");
    status.className = `status ${l.status}`;
    status.id = `status-${l.index}`;
    status.textContent = l.status === "done" ? "完成" : l.status === "failed" ? "失败" : "待合成";

    div.append(idx, time, mid, status);
    wrap.appendChild(div);
  }
}

function mkBtn(text, onclick) {
  const b = document.createElement("button");
  b.className = "mini";
  b.textContent = text;
  b.onclick = onclick;
  return b;
}

function toggleSkip(index, btn) {
  const ed = (state.edits[index] = state.edits[index] || {});
  ed.skip = !ed.skip;
  btn.classList.toggle("active", ed.skip);
  $(`line-${index}`).style.opacity = ed.skip ? 0.4 : 1;
}

/* ───────────────────────── 单句操作 ───────────────────────── */

async function regenLine(index, text) {
  const statusEl = $(`status-${index}`);
  const lineEl = $(`line-${index}`);
  const btns = lineEl.querySelectorAll(".mini");
  btns.forEach((b) => (b.disabled = true));
  statusEl.textContent = "合成中";
  statusEl.className = "status generating";
  lineEl.classList.add("generating");
  try {
    const r = await api(`/api/regen/${state.sessionId}`, {
      method: "POST",
      body: JSON.stringify({ index, text: text || null, params: collectParams() }),
    });
    const l = state.lines.find((x) => x.index === index);
    l.status = "done";
    statusEl.textContent = "完成";
    statusEl.className = "status done";
    lineEl.classList.remove("generating", "pending", "failed");
    lineEl.classList.add("done");
    const ta = lineEl.querySelector("textarea");
    ta.classList.remove("dirty");
    ta.value = l.text = text || l.text;
    btns[1].disabled = false;  // 播放
    btns[2].disabled = false;  // 下载
    delete state.edits[index];
    toast(`#${index} 生成完成 (${r.duration}s)`);
  } catch (e) {
    statusEl.textContent = "失败";
    statusEl.className = "status failed";
    lineEl.classList.remove("generating");
    lineEl.classList.add("failed");
    state.lines.find((x) => x.index === index).status = "failed";
    toast(`#${index}: ${e.message}`, true);
  } finally {
    btns.forEach((b) => {
      const isPlay = b.textContent.includes("播放");
      const isDl = b.textContent === "⬇";
      if (!isPlay && !isDl) b.disabled = false;
      if ((isPlay || isDl) && lineEl.classList.contains("done")) b.disabled = false;
    });
  }
}

function playClip(index) {
  if (state.audioObj) {
    state.audioObj.pause();
    if (state.playingIdx === index) {
      state.playingIdx = null;
      return;
    }
  }
  const a = new Audio(`/api/clip/${state.sessionId}/${index}?t=${Date.now()}`);
  a.onended = () => { state.playingIdx = null; };
  a.play();
  state.audioObj = a;
  state.playingIdx = index;
}

function downloadClip(index) {
  const a = document.createElement("a");
  a.href = `/api/clip/${state.sessionId}/${index}`;
  a.download = `${String(index).padStart(5, "0")}.wav`;
  a.click();
}

/* ───────────────────────── 批量 ───────────────────────── */

$("btn-start").onclick = async () => {
  if (!state.sessionId || state.running) return;
  const p = collectParams();
  if (!p.onnx_dir || !p.ref_audio || !p.ref_text) {
    if (!p.speaker_profile) {
      return toast("请先填写 Genie 角色：ONNX 目录 + 参考音频 + 参考文本", true);
    }
  }
  try {
    const r = await api("/api/start", {
      method: "POST",
      body: JSON.stringify({
        session_id: state.sessionId,
        overrides: state.edits,
        params: p,
      }),
    });
    state.running = true;
    $("btn-start").disabled = true;
    $("btn-stop").disabled = false;
    $("status-line").textContent = `开始合成：共 ${r.total} 句（本次需生成 ${r.resume} 句）`;
    startPolling();
  } catch (e) {
    toast(e.message, true);
  }
};

$("btn-stop").onclick = async () => {
  if (!state.sessionId) return;
  await api(`/api/stop/${state.sessionId}`, { method: "POST" });
  $("status-line").textContent = "正在停止（等当前句完成）…";
};

function startPolling() {
  clearInterval(state.pollTimer);
  state.pollTimer = setInterval(async () => {
    try {
      const r = await api(`/api/progress/${state.sessionId}`);
      updateProgress(r.done, r.total);
      for (const l of r.lines) {
        const el = state.lines.find((x) => x.index === l.index);
        if (!el || el.status === l.status) continue;
        el.status = l.status;
        const lineEl = $(`line-${l.index}`);
        const statusEl = $(`status-${l.index}`);
        if (lineEl && statusEl) {
          statusEl.textContent = l.status === "done" ? "完成" : "失败";
          statusEl.className = `status ${l.status}`;
          lineEl.classList.remove("pending", "failed", "done", "generating");
          lineEl.classList.add(l.status);
          if (l.status === "done") {
            const miniBtns = lineEl.querySelectorAll(".mini");
            miniBtns[1].disabled = false;
            miniBtns[2].disabled = false;
          }
        }
      }
      if (r.running) {
        $("status-line").textContent = `合成中… ${r.done}/${r.total}`;
      } else {
        state.running = false;
        clearInterval(state.pollTimer);
        $("btn-start").disabled = false;
        $("btn-stop").disabled = true;
        if (r.error) {
          $("status-line").textContent = "出错：" + r.error.split("\n")[0];
          toast(r.error.split("\n")[0], true);
        } else {
          $("status-line").textContent = `完成！${r.done}/${r.total} 句` + (r.merged_available ? "，整轨已生成" : "");
          toast("配音完成 ✓");
          if (r.merged_available) $("btn-download").disabled = false;
        }
      }
    } catch (e) { console.error(e); }
  }, 1200);
}

function updateProgress(done, total) {
  $("progress-bar").style.width = total ? `${(done / total) * 100}%` : "0%";
  $("progress-text").textContent = `${done} / ${total}`;
}

/* ───────────────────────── 合并 / 下载 ───────────────────────── */

$("btn-merge").onclick = async () => {
  if (!state.sessionId) return;
  try {
    await api(`/api/merge/${state.sessionId}`, { method: "POST" });
    $("btn-download").disabled = false;
    toast("合并完成，可下载整轨");
  } catch (e) {
    toast(e.message, true);
  }
};

$("btn-download").onclick = () => {
  if (!state.sessionId) return;
  const a = document.createElement("a");
  a.href = `/api/download/${state.sessionId}`;
  a.download = "dubbed.wav";
  a.click();
};

/* ───────────────────────── 筛选 ───────────────────────── */

$("filter-input").oninput = renderLines;
$("filter-status").onchange = renderLines;

window.addEventListener("load", () => $("btn-ping").click());
