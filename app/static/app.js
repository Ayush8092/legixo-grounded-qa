// Legixo Grounded Q&A — frontend.
//
// This file only ever calls this API's own endpoints (/health, /ready, /ask).
// It never talks to Gemini, Groq, or Pinecone directly, and never sees an
// API key — everything is proxied through FastAPI.

(() => {
  "use strict";

  const form = document.getElementById("ask-form");
  const questionInput = document.getElementById("question");
  const askButton = document.getElementById("ask-button");
  const errorBanner = document.getElementById("error-banner");
  const pipelineCard = document.getElementById("pipeline-card");
  const pipelineSteps = Array.from(document.querySelectorAll(".pipeline-step"));
  const answerCard = document.getElementById("answer-card");
  const answerQuestion = document.getElementById("answer-question");
  const answerText = document.getElementById("answer-text");
  const stampEl = document.getElementById("stamp");
  const citationsEl = document.getElementById("citations");
  const citationsHeading = document.getElementById("citations-heading");
  const copyButton = document.getElementById("copy-button");
  const copyButtonLabel = document.getElementById("copy-button-label");
  const askAnotherButton = document.getElementById("ask-another-button");
  const scenesToggle = document.getElementById("scenes-toggle");
  const scenesPanel = document.getElementById("scenes-panel");
  const expansionBlock = document.getElementById("expansion-block");
  const queryList = document.getElementById("query-list");
  const retrievalBlock = document.getElementById("retrieval-block");
  const retrievalStats = document.getElementById("retrieval-stats");
  const rawToggle = document.getElementById("raw-toggle");
  const rawTrace = document.getElementById("raw-trace");
  const statusDot = document.getElementById("status-dot");
  const statusText = document.getElementById("status-text");
  const themeToggle = document.getElementById("theme-toggle");

  // ---------- Theme ----------
  //
  // The actual theme attribute is set synchronously in <head> (see
  // index.html) to avoid a flash of the wrong theme. This just wires up
  // the toggle button and persists the choice.

  const THEME_KEY = "legixo-theme";

  function currentTheme() {
    return document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
  }

  function setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch {
      // Storage unavailable (private mode, etc.) — theme still applies for this session.
    }
    themeToggle.setAttribute("aria-label", theme === "light" ? "Switch to dark theme" : "Switch to light theme");
  }

  themeToggle.addEventListener("click", () => {
    setTheme(currentTheme() === "light" ? "dark" : "light");
  });
  // Ensure the aria-label matches whatever theme the inline bootstrap script picked.
  setTheme(currentTheme());

  // ---------- Example chips ----------

  document.getElementById("example-chips").addEventListener("click", (event) => {
    const chip = event.target.closest(".chip");
    if (!chip) return;
    questionInput.value = chip.dataset.q;
    form.requestSubmit();
  });

  // Enter submits, Shift+Enter inserts a newline.
  questionInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  // ---------- Loading / thinking pipeline ----------
  //
  // Mirrors the real backend pipeline stages (retrieve -> grade -> generate
  // -> validate). This is a UI-only progression through those four labels
  // while the single /ask request is in flight — it does not represent
  // model reasoning or invented intermediate output, just which stage of
  // the actual application pipeline is likely running.

  const PIPELINE_STEP_MS = 850;
  let pipelineTimer = null;

  function startPipeline() {
    pipelineSteps.forEach((el) => el.classList.remove("active", "done"));
    pipelineCard.hidden = false;
    let index = 0;
    pipelineSteps[0].classList.add("active");

    pipelineTimer = window.setInterval(() => {
      if (index >= pipelineSteps.length - 1) {
        window.clearInterval(pipelineTimer);
        pipelineTimer = null;
        return;
      }
      pipelineSteps[index].classList.remove("active");
      pipelineSteps[index].classList.add("done");
      index += 1;
      pipelineSteps[index].classList.add("active");
    }, PIPELINE_STEP_MS);
  }

  function stopPipeline() {
    if (pipelineTimer) {
      window.clearInterval(pipelineTimer);
      pipelineTimer = null;
    }
    pipelineCard.hidden = true;
    pipelineSteps.forEach((el) => el.classList.remove("active", "done"));
  }

  // ---------- Ask ----------

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = questionInput.value.trim();
    if (!question) return;

    setLoading(true);
    hideError();
    answerCard.hidden = true;
    startPipeline();

    try {
      const response = await fetch("/ask?trace=true", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (!response.ok) {
        const detail = await safeDetail(response);
        throw new Error(detail || `Request failed (HTTP ${response.status}).`);
      }

      const data = await response.json();
      renderAnswer(question, data);
    } catch (err) {
      showError(err.message || "Something went wrong. Please try again.");
    } finally {
      setLoading(false);
      stopPipeline();
    }
  });

  async function safeDetail(response) {
    try {
      const body = await response.json();
      return body.detail;
    } catch {
      return null;
    }
  }

  function setLoading(isLoading) {
    askButton.disabled = isLoading;
    askButton.classList.toggle("loading", isLoading);
  }

  function showError(message) {
    errorBanner.textContent = message;
    errorBanner.hidden = false;
  }

  function hideError() {
    errorBanner.hidden = true;
  }

  // ---------- Render answer + citations ----------

  function renderAnswer(question, data) {
    answerQuestion.textContent = question;
    answerText.textContent = data.answer;
    answerText.dataset.rawAnswer = data.answer;

    stampEl.textContent = data.found ? "Grounded" : "Not found";
    stampEl.className = "stamp " + (data.found ? "found" : "not-found");

    resetCopyButton();

    citationsEl.innerHTML = "";
    const citations = data.citations || [];
    citationsHeading.hidden = citations.length === 0;
    citations.forEach((c, i) => {
      citationsEl.appendChild(renderExhibit(c, i + 1));
    });

    renderScenes(data.trace || []);
    answerCard.hidden = false;
    answerCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function renderExhibit(citation, index) {
    // Compact by default: header (label, source, section, score) plus a
    // "View evidence" toggle — the full retrieved text only renders once
    // expanded. No citation data is removed, just not shown until asked for.
    const el = document.createElement("div");
    el.className = "exhibit";
    el.innerHTML = `
      <div class="exhibit-head">
        <span class="exhibit-tag">EVIDENCE ${index}</span>
        <span class="exhibit-source">${escapeHtml(citation.source_file)}</span>
        <span class="exhibit-section">${escapeHtml(citation.section)}</span>
        <span class="exhibit-score">score ${citation.score.toFixed(4)}</span>
      </div>
      <button type="button" class="exhibit-toggle" aria-expanded="false">View evidence</button>
      <p class="exhibit-snippet" hidden>"${escapeHtml(citation.snippet)}"</p>
    `;

    const toggle = el.querySelector(".exhibit-toggle");
    const snippet = el.querySelector(".exhibit-snippet");
    toggle.addEventListener("click", () => {
      const expanded = toggle.getAttribute("aria-expanded") === "true";
      toggle.setAttribute("aria-expanded", String(!expanded));
      toggle.textContent = expanded ? "View evidence" : "Hide evidence";
      snippet.hidden = expanded;
    });

    return el;
  }

  // ---------- Copy answer ----------

  let copyResetTimer = null;

  function resetCopyButton() {
    if (copyResetTimer) {
      window.clearTimeout(copyResetTimer);
      copyResetTimer = null;
    }
    copyButton.classList.remove("copied");
    copyButtonLabel.textContent = "Copy answer";
  }

  copyButton.addEventListener("click", async () => {
    const text = answerText.dataset.rawAnswer || answerText.textContent || "";
    const ok = await copyToClipboard(text);
    if (ok) {
      copyButton.classList.add("copied");
      copyButtonLabel.textContent = "✓ Copied";
    } else {
      copyButtonLabel.textContent = "Copy failed";
    }
    if (copyResetTimer) window.clearTimeout(copyResetTimer);
    copyResetTimer = window.setTimeout(resetCopyButton, 1800);
  });

  async function copyToClipboard(text) {
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
        return true;
      }
    } catch {
      // fall through to the legacy fallback below
    }
    try {
      const textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      const ok = document.execCommand("copy");
      document.body.removeChild(textarea);
      return ok;
    } catch {
      return false;
    }
  }

  // ---------- Ask another question ----------

  askAnotherButton.addEventListener("click", () => {
    answerCard.hidden = true;
    hideError();
    questionInput.value = "";
    questionInput.focus();
    form.scrollIntoView({ behavior: "smooth", block: "center" });
  });

  // ---------- Behind the scenes: parse the real trace strings ----------

  const NODE_LABELS = {
    retrieve: "Retrieve (Pinecone)",
    rerank: "BM25 rerank",
    grade_chunks: "Grounding grader",
    rewrite_query: "Rewrite query",
    generate_answer: "Generate answer",
    validate_citations: "Validate citations",
    refuse: "Refuse",
  };

  // ---------- Trace stages: collapsible presentation ----------
  //
  // The 5 stages recruiters/reviewers see are a presentation grouping over
  // the same raw trace lines the app already parses above — no new data,
  // no backend change. Each raw trace node key maps to exactly one of the
  // 5 stage lists below; the existing per-line technical detail (already
  // built by renderScenes) is simply appended into that stage's own list
  // instead of one combined list. `rewrite_query` lines are part of the
  // retrieval loop (a retry re-runs retrieve), so they join the Retrieve
  // stage; `refuse` explains why Generate produced no answer, so it joins
  // the Generate stage. Every stage stays visible even with zero lines —
  // its static default description (in index.html) still shows.
  const STAGE_KEYS = ["retrieve", "rerank", "grade", "generate", "validate"];
  const STAGE_FOR_NODE_KEY = {
    retrieve: "retrieve",
    rewrite_query: "retrieve",
    rerank: "rerank",
    grade_chunks: "grade",
    generate_answer: "generate",
    refuse: "generate",
    validate_citations: "validate",
  };

  const stageLists = {};
  const stageHeaders = {};
  const stageChevrons = {};
  STAGE_KEYS.forEach((stage) => {
    stageLists[stage] = document.getElementById(`docket-log-${stage}`);
    stageHeaders[stage] = document.getElementById(`trace-stage-header-${stage}`);
    stageChevrons[stage] = stageHeaders[stage].querySelector(".trace-stage-chevron");
  });

  function collapseAllStages() {
    STAGE_KEYS.forEach((stage) => setStageExpanded(stage, false));
  }

  function setStageExpanded(stage, expanded) {
    const header = stageHeaders[stage];
    const body = document.getElementById(`trace-stage-body-${stage}`);
    header.setAttribute("aria-expanded", String(expanded));
    body.hidden = !expanded;
    stageChevrons[stage].textContent = expanded ? "⌄" : "›";
  }

  // Each stage expands/collapses independently; multiple may be open at once.
  STAGE_KEYS.forEach((stage) => {
    stageHeaders[stage].addEventListener("click", () => {
      const expanded = stageHeaders[stage].getAttribute("aria-expanded") === "true";
      setStageExpanded(stage, !expanded);
    });
  });

  // All five stages start collapsed — both on initial page load (already
  // reflected in index.html's static `hidden`/aria-expanded markup) and
  // again before every new render, so a previous answer's expanded stages
  // never leak into the next question's result.
  collapseAllStages();

  function nodeKeyFromLine(line) {
    const head = line.split(":")[0].trim();
    return head.split(" (")[0].trim();
  }

  /** Pull single/double-quoted substrings out of a Python-repr-ish list, e.g. "['a', 'b']". */
  function extractQuoted(str) {
    const out = [];
    const re = /'([^']*)'|"([^"]*)"/g;
    let m;
    while ((m = re.exec(str)) !== null) {
      out.push(m[1] !== undefined ? m[1] : m[2]);
    }
    return out;
  }

  function renderScenes(trace) {
    STAGE_KEYS.forEach((stage) => {
      stageLists[stage].innerHTML = "";
    });
    collapseAllStages();
    queryList.innerHTML = "";
    retrievalStats.innerHTML = "";
    expansionBlock.hidden = true;
    retrievalBlock.hidden = true;
    rawTrace.textContent = trace.join("\n");

    if (!trace.length) {
      return;
    }

    let lastRetrieveQueries = null;
    let lastChunkCount = null;
    let lastThreshold = null;
    let relevantCount = null;
    let retryCount = 0;

    trace.forEach((line, i) => {
      const key = nodeKeyFromLine(line);
      const label = NODE_LABELS[key] || key || "step";
      const detail = line.includes(":") ? line.slice(line.indexOf(":") + 1).trim() : line;

      const li = document.createElement("li");
      li.className = "node-" + key;
      li.dataset.n = String(i + 1);
      li.innerHTML = `<span class="node-name">${escapeHtml(label)}</span><span class="node-detail">${escapeHtml(detail)}</span>`;
      const targetStage = STAGE_FOR_NODE_KEY[key] || "retrieve";
      stageLists[targetStage].appendChild(li);

      if (key === "retrieve") {
        const queriesMatch = line.match(/queries=(\[.*?\])/);
        if (queriesMatch) lastRetrieveQueries = extractQuoted(queriesMatch[1]);
        const countMatch = line.match(/->\s*(\d+)\s*chunks/);
        if (countMatch) lastChunkCount = countMatch[1];
        const thresholdMatch = line.match(/score>=([\d.]+)/);
        if (thresholdMatch) lastThreshold = thresholdMatch[1];
      }
      if (key === "grade_chunks") {
        const relevantMatch = line.match(/relevant=(\[.*?\])/);
        if (relevantMatch) relevantCount = extractQuoted(relevantMatch[1]).length;
      }
      if (key === "rewrite_query") {
        retryCount += 1;
      }
    });

    if (lastRetrieveQueries && lastRetrieveQueries.length) {
      expansionBlock.hidden = false;
      lastRetrieveQueries.forEach((q, i) => {
        const li = document.createElement("li");
        li.textContent = q;
        if (i === 0) li.classList.add("original");
        queryList.appendChild(li);
      });
    }

    if (lastChunkCount !== null) {
      retrievalBlock.hidden = false;
      addStat("Chunks retrieved (final attempt)", lastChunkCount);
      if (lastThreshold !== null) addStat("Score threshold", lastThreshold);
      if (relevantCount !== null) addStat("Graded relevant", String(relevantCount));
      addStat("Retry attempts used", String(retryCount));
    }
  }

  function addStat(label, value) {
    const div = document.createElement("div");
    div.innerHTML = `<dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd>`;
    retrievalStats.appendChild(div);
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = String(str);
    return div.innerHTML;
  }

  // ---------- Toggles ----------

  scenesToggle.addEventListener("click", () => {
    const expanded = scenesToggle.getAttribute("aria-expanded") === "true";
    scenesToggle.setAttribute("aria-expanded", String(!expanded));
    scenesPanel.hidden = expanded;
  });

  rawToggle.addEventListener("click", () => {
    const expanded = rawToggle.getAttribute("aria-expanded") === "true";
    rawToggle.setAttribute("aria-expanded", String(!expanded));
    rawTrace.hidden = expanded;
    rawToggle.textContent = expanded ? "Show raw trace" : "Hide raw trace";
  });

  // ---------- Footer status ----------

  (async () => {
    try {
      const response = await fetch("/ready");
      const data = await response.json();
      statusDot.className = "status-dot " + (data.ready ? "ok" : "down");
      statusText.textContent = data.ready
        ? "Corpus loaded and ready"
        : "Not ready — run ingestion / check API keys";
      statusText.title = data.detail || "";
    } catch {
      statusDot.className = "status-dot down";
      statusText.textContent = "Service unreachable";
    }
  })();

  // ---------- Upload documents ----------
  //
  // Only ever calls this API's own POST /upload — same single-origin
  // pattern as /ask. The upload button stays disabled until at least one
  // file is selected/dropped; there is no auto-upload on selection, so a
  // person can review the file list first.

  const dropzone = document.getElementById("dropzone");
  const uploadInput = document.getElementById("upload-input");
  const uploadFileList = document.getElementById("upload-file-list");
  const uploadButton = document.getElementById("upload-button");
  const uploadStatus = document.getElementById("upload-status");

  const ACCEPTED_EXTENSIONS = [".md", ".txt", ".pdf", ".docx"];
  let selectedFiles = [];

  function extensionOf(filename) {
    const i = filename.lastIndexOf(".");
    return i === -1 ? "" : filename.slice(i).toLowerCase();
  }

  function setSelectedFiles(fileList) {
    // Client-side extension filtering is a UX nicety only — the server
    // re-validates every file regardless (see app/main.py `/upload`), so
    // this is never the actual security boundary.
    selectedFiles = Array.from(fileList).filter((f) => ACCEPTED_EXTENSIONS.includes(extensionOf(f.name)));
    renderFileList();
    uploadButton.disabled = selectedFiles.length === 0;
  }

  function renderFileList() {
    uploadFileList.innerHTML = "";
    uploadFileList.hidden = selectedFiles.length === 0;
    selectedFiles.forEach((file, index) => {
      const li = document.createElement("li");
      li.className = "upload-file-item";

      const name = document.createElement("span");
      name.className = "upload-file-name";
      name.textContent = file.name;

      const size = document.createElement("span");
      size.className = "upload-file-size";
      size.textContent = formatBytes(file.size);

      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "upload-file-remove";
      remove.setAttribute("aria-label", `Remove ${file.name}`);
      remove.textContent = "×";
      remove.addEventListener("click", () => {
        selectedFiles.splice(index, 1);
        renderFileList();
        uploadButton.disabled = selectedFiles.length === 0;
      });

      li.append(name, size, remove);
      uploadFileList.appendChild(li);
    });
  }

  function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  uploadInput.addEventListener("change", () => setSelectedFiles(uploadInput.files));

  ["dragover", "dragenter"].forEach((evt) =>
    dropzone.addEventListener(evt, (event) => {
      event.preventDefault();
      dropzone.classList.add("dropzone-active");
    })
  );
  ["dragleave", "dragend"].forEach((evt) =>
    dropzone.addEventListener(evt, () => dropzone.classList.remove("dropzone-active"))
  );
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("dropzone-active");
    if (event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files.length) {
      setSelectedFiles(event.dataTransfer.files);
    }
  });

  function setUploadStatus(message, tone) {
    if (!message) {
      uploadStatus.hidden = true;
      uploadStatus.innerHTML = "";
      return;
    }
    setUploadStatusLines([{ text: message, tone }]);
  }

  /** Render each status update as its own line, e.g.:
   *    ✓ contract.pdf uploaded
   *    12 chunks created · 10 vectors indexed · 2 unchanged
   *    ✓ Ready to ask questions
   * Every value here comes straight from the /upload response — nothing is invented. */
  function setUploadStatusLines(lines) {
    uploadStatus.hidden = lines.length === 0;
    uploadStatus.innerHTML = lines
      .map((line) => {
        const text = typeof line === "string" ? line : line.text;
        const tone = typeof line === "string" ? "" : line.tone || "";
        const cls = "upload-status-line" + (tone ? ` upload-status-${tone}` : "");
        return `<div class="${cls}">${escapeHtml(text)}</div>`;
      })
      .join("");
  }

  function setUploadLoading(isLoading) {
    uploadButton.disabled = isLoading || selectedFiles.length === 0;
    uploadButton.classList.toggle("loading", isLoading);
    uploadInput.disabled = isLoading;
  }

  uploadButton.addEventListener("click", async () => {
    if (!selectedFiles.length) return;

    setUploadLoading(true);
    // A simple, honest request lifecycle: this project's ingestion is a
    // single request/response, not a streaming job, so "Uploading..." ->
    // "Processing..." is shown optimistically rather than tracking real
    // server-side sub-steps (see final_improvement_lexido.docx: "if the
    // architecture doesn't support streaming progress, do not over-engineer it").
    setUploadStatus("Uploading & indexing…", "pending");

    const formData = new FormData();
    selectedFiles.forEach((file) => formData.append("files", file));

    try {
      const response = await fetch("/upload", { method: "POST", body: formData });
      const data = await response.json().catch(() => null);

      if (!response.ok) {
        const detail = data && data.detail;
        throw new Error(detail || `Upload failed (HTTP ${response.status}).`);
      }

      renderUploadResult(data);
    } catch (err) {
      setUploadStatus(err.message || "Upload failed. Please try again.", "error");
    } finally {
      setUploadLoading(false);
    }
  });

  function renderUploadResult(data) {
    const lines = [];
    const files = data.files || [];
    const rejected = data.rejected || [];

    files.forEach((name) => lines.push({ text: `✓ ${name} uploaded`, tone: "success" }));

    if (files.length) {
      const chunkWord = data.chunks_created === 1 ? "chunk" : "chunks";
      const vectorWord = data.vectors_upserted === 1 ? "vector" : "vectors";
      lines.push({ text: `${data.chunks_created} ${chunkWord} created · ${data.vectors_upserted} ${vectorWord} indexed` });
      if (data.unchanged_chunks) {
        lines.push({ text: `${data.unchanged_chunks} unchanged` });
      }
      if (data.stale_vectors_deleted) {
        const staleWord = data.stale_vectors_deleted === 1 ? "vector" : "vectors";
        lines.push({ text: `${data.stale_vectors_deleted} stale ${staleWord} cleaned up` });
      }
    }

    const errorWord = rejected.length === 1 ? "error" : "errors";
    lines.push({ text: `${rejected.length} ${errorWord}`, tone: rejected.length ? "error" : "" });
    rejected.forEach((r) => lines.push({ text: `✗ ${r.filename}: ${r.error}`, tone: "error" }));

    if (files.length) {
      lines.push({
        text: rejected.length ? "Ready to ask questions (some files were skipped)" : "✓ Ready to ask questions",
        tone: "success",
      });
    }

    setUploadStatusLines(lines.length ? lines : [{ text: "Upload complete." }]);

    if (data.status !== "error") {
      selectedFiles = [];
      uploadInput.value = "";
      renderFileList();
      uploadButton.disabled = true;
    }
  }
})();