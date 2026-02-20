/* AURORA Dashboard — Frontend Logic */

(function () {
  "use strict";

  // ── DOM references ─────────────────────────────────────────────────────────
  const form           = document.getElementById("predict-form");
  const runBtn         = document.getElementById("run-btn");
  const btnText        = document.getElementById("btn-text");
  const spinner        = document.getElementById("spinner");
  const errorBanner    = document.getElementById("error-banner");
  const errorMsg       = document.getElementById("error-msg");
  const resultsSection = document.getElementById("results-section");
  const dropZone       = document.getElementById("drop-zone");
  const fileInput      = document.getElementById("dicom_file");
  const dropText       = document.getElementById("drop-text");
  const fileName       = document.getElementById("file-name");

  // Results widgets
  const riskBox      = document.getElementById("risk-box");
  const riskLabel    = document.getElementById("risk-label");
  const riskPct      = document.getElementById("risk-pct");
  const statCnnProb  = document.getElementById("stat-cnn-prob");
  const statEnsProb  = document.getElementById("stat-ens-prob");
  const statFrames   = document.getElementById("stat-frames");
  const statCS       = document.getElementById("stat-cs");
  const statPrevia   = document.getElementById("stat-previa");
  const statPid      = document.getElementById("stat-pid");

  // GradCAM widgets
  const gradcamSection       = document.getElementById("gradcam-section");
  const gradcamLoading       = document.getElementById("gradcam-loading");
  const gradcamLoadText      = document.getElementById("gradcam-loading-text");
  const gradcamProgressWrap  = document.getElementById("gradcam-progress-wrap");
  const gradcamProgressBar   = document.getElementById("gradcam-progress-bar");
  const gradcamProgressLabel = document.getElementById("gradcam-progress-label");
  const gradcamViewer        = document.getElementById("gradcam-viewer");
  const gradcamCanvas   = document.getElementById("gradcam-canvas");
  const gradcamCtx      = gradcamCanvas ? gradcamCanvas.getContext("2d") : null;
  const gradcamSlider   = document.getElementById("gradcam-slider");
  const gcFrameNum      = document.getElementById("gc-frame-num");
  const gcTotalFrames   = document.getElementById("gc-total-frames");
  const gcProb          = document.getElementById("gc-prob");
  const gcSliderEnd     = document.getElementById("gc-slider-end");
  const overlayToggle   = document.getElementById("overlay-toggle");
  const opacitySlider   = document.getElementById("opacity-slider");
  const opacityValue    = document.getElementById("opacity-value");

  // State
  let frameChart    = null;
  let perFrameProbs = [];
  let gcOverlays    = [];   // [{frame_idx, rawImg, hmImg}, ...]
  let gcTotalCount  = 0;
  let overlayOn     = true;
  let overlayOpacity = 0.6;

  // ── Drag-and-drop ──────────────────────────────────────────────────────────
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const files = e.dataTransfer.files;
    if (files.length > 0) setFile(files[0]);
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) setFile(fileInput.files[0]);
  });

  function setFile(file) {
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;
    dropText.classList.add("hidden");
    fileName.textContent = file.name;
    fileName.classList.remove("hidden");
  }

  // ── Form submit ────────────────────────────────────────────────────────────
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideError();

    if (!fileInput.files || fileInput.files.length === 0) {
      showError("Please select a .dcm file before running analysis.");
      return;
    }

    setLoading(true);
    resetGradcam();

    try {
      const formData = new FormData(form);
      const response = await fetch("/predict", { method: "POST", body: formData });
      const data = await response.json();

      if (!response.ok) {
        showError(data.detail || `HTTP ${response.status}`);
        return;
      }

      renderResults(data);

      // Auto-trigger GradCAM after predict completes
      triggerGradcam(formData);

    } catch (err) {
      showError("Network error: " + err.message);
    } finally {
      setLoading(false);
    }
  });

  // ── GradCAM trigger ────────────────────────────────────────────────────────
  async function triggerGradcam(formData) {
    gradcamSection.classList.remove("hidden");
    gradcamLoading.classList.remove("hidden");
    gradcamViewer.classList.add("hidden");
    gradcamLoadText.textContent = "Generating GradCAM overlays…";
    gradcamProgressWrap.classList.add("hidden");
    gradcamProgressBar.style.width = "0%";
    gradcamProgressLabel.textContent = "";

    try {
      const response = await fetch("/gradcam", { method: "POST", body: formData });

      if (!response.ok) {
        // Non-streaming error from the server (e.g. 503, 413)
        const data = await response.json();
        gradcamLoadText.textContent = "GradCAM failed: " + (data.detail || "unknown error");
        return;
      }

      // Read the NDJSON stream line-by-line
      // Each line is either an overlay+progress or the final done signal.
      const reader    = response.body.getReader();
      const decoder   = new TextDecoder();
      let buffer      = "";
      const overlays  = [];   // collect overlays as they arrive

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();           // keep any incomplete last line

        for (const line of lines) {
          if (!line.trim()) continue;
          const msg = JSON.parse(line);
          if (msg.error) {
            gradcamLoadText.textContent = "GradCAM error: " + msg.error;
            return;
          } else if (msg.overlay) {
            overlays.push(msg.overlay);
            updateGradcamProgress(msg.progress, msg.total);
          } else if (msg.done) {
            // Attach the collected overlays and render
            renderGradcam({ ...msg, overlays });
          }
        }
      }

      // Flush any remaining data in buffer
      if (buffer.trim()) {
        const msg = JSON.parse(buffer);
        if (msg.done) renderGradcam({ ...msg, overlays });
      }

    } catch (err) {
      gradcamLoadText.textContent = "GradCAM error: " + err.message;
    }
  }

  function updateGradcamProgress(progress, total) {
    gradcamProgressWrap.classList.remove("hidden");
    const pct = Math.round((progress / total) * 100);
    gradcamProgressBar.style.width = pct + "%";
    gradcamProgressLabel.textContent = `${progress} / ${total} frames`;
  }

  // ── Render GradCAM viewer ──────────────────────────────────────────────────
  function renderGradcam(data) {
    const rawOverlays = data.overlays || [];
    gcTotalCount = data.num_frames || 0;

    if (rawOverlays.length === 0) {
      gradcamLoadText.textContent = "No GradCAM frames returned.";
      return;
    }

    gcOverlays = [];
    let loadCount = 0;
    const total = rawOverlays.length;

    function onPairLoaded() {
      loadCount++;
      if (loadCount === total) showViewer();
    }

    rawOverlays.forEach((ov) => {
      const rawImg = new Image();
      const hmImg  = new Image();
      let rawDone = false, hmDone = false;

      function checkPair() {
        if (rawDone && hmDone) onPairLoaded();
      }

      rawImg.onload = () => { rawDone = true; checkPair(); };
      hmImg.onload  = () => { hmDone  = true; checkPair(); };

      rawImg.src = "data:image/jpeg;base64," + ov.raw_b64;
      hmImg.src  = "data:image/png;base64,"  + ov.heatmap_b64;

      // Data URIs may already be complete in some browsers before onload fires
      if (rawImg.complete && !rawDone) { rawDone = true; checkPair(); }
      if (hmImg.complete && !hmDone)   { hmDone  = true; checkPair(); }

      gcOverlays.push({ frame_idx: ov.frame_idx, rawImg, hmImg });
    });
  }

  function showViewer() {
    gradcamSlider.min   = 0;
    gradcamSlider.max   = gcTotalCount - 1;
    gradcamSlider.value = 0;
    gcTotalFrames.textContent = gcTotalCount;
    gcSliderEnd.textContent   = `Frame ${gcTotalCount - 1}`;

    // Reset overlay controls to defaults
    overlayOn      = true;
    overlayOpacity = 0.6;
    overlayToggle.textContent = "Hide Overlay";
    overlayToggle.classList.add("active");
    opacitySlider.value       = 60;
    opacityValue.textContent  = "60%";

    showGradcamFrame(0);

    gradcamLoading.classList.add("hidden");
    gradcamViewer.classList.remove("hidden");
    gradcamSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // Find the overlay entry whose frame_idx is closest to the requested index
  function nearestOverlay(frameIdx) {
    let best = gcOverlays[0];
    let bestDist = Math.abs(gcOverlays[0].frame_idx - frameIdx);
    for (const ov of gcOverlays) {
      const d = Math.abs(ov.frame_idx - frameIdx);
      if (d < bestDist) { bestDist = d; best = ov; }
    }
    return best;
  }

  function showGradcamFrame(frameIdx) {
    if (!gradcamCtx || gcOverlays.length === 0) return;

    const ov = nearestOverlay(frameIdx);
    const W  = gradcamCanvas.width;
    const H  = gradcamCanvas.height;

    // Draw raw ultrasound frame
    gradcamCtx.globalAlpha = 1.0;
    gradcamCtx.drawImage(ov.rawImg, 0, 0, W, H);

    // Composite heatmap on top if overlay is enabled
    if (overlayOn) {
      gradcamCtx.globalAlpha = overlayOpacity;
      gradcamCtx.drawImage(ov.hmImg, 0, 0, W, H);
      gradcamCtx.globalAlpha = 1.0;
    }

    gcFrameNum.textContent = frameIdx;
    const prob = perFrameProbs[frameIdx];
    gcProb.textContent = prob !== undefined ? Number(prob).toFixed(3) : "—";
  }

  gradcamSlider.addEventListener("input", () => {
    showGradcamFrame(parseInt(gradcamSlider.value, 10));
  });

  overlayToggle.addEventListener("click", () => {
    overlayOn = !overlayOn;
    overlayToggle.textContent = overlayOn ? "Hide Overlay" : "Show Overlay";
    overlayToggle.classList.toggle("active", overlayOn);
    showGradcamFrame(parseInt(gradcamSlider.value, 10));
  });

  opacitySlider.addEventListener("input", () => {
    overlayOpacity = parseInt(opacitySlider.value, 10) / 100;
    opacityValue.textContent = opacitySlider.value + "%";
    if (overlayOn) showGradcamFrame(parseInt(gradcamSlider.value, 10));
  });

  function resetGradcam() {
    gcOverlays   = [];
    gcTotalCount = 0;
    if (gradcamCtx) gradcamCtx.clearRect(0, 0, gradcamCanvas.width, gradcamCanvas.height);
    gradcamProgressWrap.classList.add("hidden");
    gradcamProgressBar.style.width = "0%";
    gradcamProgressLabel.textContent = "";
    gradcamSection.classList.add("hidden");
    gradcamViewer.classList.add("hidden");
    gradcamLoading.classList.add("hidden");
  }

  // ── Render results ─────────────────────────────────────────────────────────
  function renderResults(data) {
    const cnn       = data.cnn || {};
    const ensemble  = data.ensemble || {};
    const dicomInfo = data.dicom_info || {};
    const isHigh    = data.risk_level === "HIGH";

    // Store per-frame probs for the slider label
    perFrameProbs = cnn.per_frame_prob || [];

    const ensProb = ensemble.ensemble_prob !== undefined
      ? ensemble.ensemble_prob
      : cnn.cnn_prob;

    riskBox.className     = "risk-box " + (isHigh ? "high" : "low");
    riskLabel.textContent = isHigh ? "HIGH RISK" : "LOW RISK";
    riskPct.textContent   = (ensProb * 100).toFixed(1) + "%";

    statCnnProb.textContent = fmt(cnn.cnn_prob);
    statEnsProb.textContent = ensemble.ensemble_prob !== undefined
      ? fmt(ensemble.ensemble_prob) : "N/A";
    statFrames.textContent  = dicomInfo.num_frames ?? "—";
    statCS.textContent      = data.num_prior_cs ?? "—";
    statPrevia.textContent  = data.previa
      ? data.previa.charAt(0).toUpperCase() + data.previa.slice(1) : "—";
    statPid.textContent     = data.patient_id || "—";

    renderChart(perFrameProbs);

    resultsSection.classList.remove("hidden");
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function fmt(val) {
    return val !== undefined && val !== null ? Number(val).toFixed(3) : "—";
  }

  // ── Chart.js ───────────────────────────────────────────────────────────────
  function renderChart(probs) {
    const ctx = document.getElementById("frame-chart").getContext("2d");

    let labels, values;
    if (probs.length > 400) {
      const step = Math.ceil(probs.length / 400);
      values = probs.filter((_, i) => i % step === 0);
      labels = values.map((_, i) => i * step);
    } else {
      values = probs;
      labels = probs.map((_, i) => i);
    }

    if (frameChart) frameChart.destroy();

    const pointColors = values.map((v) =>
      v >= 0.5 ? "rgba(192, 57, 43, 0.7)" : "rgba(30, 77, 145, 0.5)"
    );

    frameChart = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "CNN Probability",
            data: values,
            borderColor: "rgba(30, 77, 145, 0.8)",
            borderWidth: 1.5,
            pointRadius: probs.length > 100 ? 0 : 3,
            pointBackgroundColor: pointColors,
            fill: { target: "origin", above: "rgba(192, 57, 43, 0.06)" },
            tension: 0.3,
          },
          {
            label: "Threshold (0.5)",
            data: labels.map(() => 0.5),
            borderColor: "rgba(192, 57, 43, 0.5)",
            borderWidth: 1,
            borderDash: [6, 4],
            pointRadius: 0,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { position: "top", labels: { font: { size: 12 } } },
          tooltip: {
            callbacks: {
              label: (ctx) =>
                ctx.datasetIndex === 0
                  ? `  Prob: ${ctx.parsed.y.toFixed(3)}`
                  : null,
            },
          },
        },
        scales: {
          x: {
            title: { display: true, text: "Frame", font: { size: 11 } },
            ticks: { maxTicksLimit: 10 },
          },
          y: {
            title: { display: true, text: "Probability", font: { size: 11 } },
            min: 0,
            max: 1,
            ticks: { stepSize: 0.25 },
          },
        },
      },
    });
  }

  // ── UI helpers ─────────────────────────────────────────────────────────────
  function setLoading(on) {
    runBtn.disabled    = on;
    spinner.classList.toggle("hidden", !on);
    btnText.textContent = on ? "Analyzing…" : "Run Analysis";
  }

  function showError(msg) {
    errorMsg.textContent = msg;
    errorBanner.classList.remove("hidden");
  }

  function hideError() {
    errorBanner.classList.add("hidden");
    errorMsg.textContent = "";
  }
})();
