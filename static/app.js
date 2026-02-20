/* AURORA Dashboard — Frontend Logic */

(function () {
  "use strict";

  // ── DOM references ─────────────────────────────────────────────────────────
  const form          = document.getElementById("predict-form");
  const runBtn        = document.getElementById("run-btn");
  const btnText       = document.getElementById("btn-text");
  const spinner       = document.getElementById("spinner");
  const errorBanner   = document.getElementById("error-banner");
  const errorMsg      = document.getElementById("error-msg");
  const resultsSection = document.getElementById("results-section");
  const dropZone      = document.getElementById("drop-zone");
  const fileInput     = document.getElementById("dicom_file");
  const dropText      = document.getElementById("drop-text");
  const fileName      = document.getElementById("file-name");

  // Results widgets
  const riskBox       = document.getElementById("risk-box");
  const riskLabel     = document.getElementById("risk-label");
  const riskPct       = document.getElementById("risk-pct");
  const statCnnProb   = document.getElementById("stat-cnn-prob");
  const statEnsProb   = document.getElementById("stat-ens-prob");
  const statFrames    = document.getElementById("stat-frames");
  const statCS        = document.getElementById("stat-cs");
  const statPrevia    = document.getElementById("stat-previa");
  const statPid       = document.getElementById("stat-pid");

  // Chart instance
  let frameChart = null;

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
    if (files.length > 0) {
      setFile(files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      setFile(fileInput.files[0]);
    }
  });

  function setFile(file) {
    // Transfer to the hidden file input so FormData picks it up
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

    // Basic validation
    if (!fileInput.files || fileInput.files.length === 0) {
      showError("Please select a .dcm file before running analysis.");
      return;
    }

    setLoading(true);

    try {
      const formData = new FormData(form);
      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        const detail = data.detail || `HTTP ${response.status}`;
        showError(detail);
        return;
      }

      renderResults(data);
    } catch (err) {
      showError("Network error: " + err.message);
    } finally {
      setLoading(false);
    }
  });

  // ── Render results ─────────────────────────────────────────────────────────
  function renderResults(data) {
    const cnn         = data.cnn || {};
    const ensemble    = data.ensemble || {};
    const dicomInfo   = data.dicom_info || {};
    const isHigh      = data.risk_level === "HIGH";

    const ensProb     = ensemble.ensemble_prob !== undefined
                          ? ensemble.ensemble_prob
                          : cnn.cnn_prob;
    const pct         = (ensProb * 100).toFixed(1) + "%";

    // Risk badge
    riskBox.className = "risk-box " + (isHigh ? "high" : "low");
    riskLabel.textContent = isHigh ? "HIGH RISK" : "LOW RISK";
    riskPct.textContent   = pct;

    // Stats
    statCnnProb.textContent  = fmt(cnn.cnn_prob);
    statEnsProb.textContent  = ensemble.ensemble_prob !== undefined
                                 ? fmt(ensemble.ensemble_prob)
                                 : "N/A";
    statFrames.textContent   = dicomInfo.num_frames ?? "—";
    statCS.textContent       = data.num_prior_cs ?? "—";
    statPrevia.textContent   = data.previa
                                 ? data.previa.charAt(0).toUpperCase() + data.previa.slice(1)
                                 : "—";
    statPid.textContent      = data.patient_id || "—";

    // Per-frame chart
    renderChart(cnn.per_frame_prob || []);

    // Show section
    resultsSection.classList.remove("hidden");
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function fmt(val) {
    return val !== undefined && val !== null ? Number(val).toFixed(3) : "—";
  }

  // ── Chart.js chart ─────────────────────────────────────────────────────────
  function renderChart(probs) {
    const ctx = document.getElementById("frame-chart").getContext("2d");

    // Down-sample if too many frames (> 400) for performance
    let labels, values;
    if (probs.length > 400) {
      const step = Math.ceil(probs.length / 400);
      values = probs.filter((_, i) => i % step === 0);
      labels = values.map((_, i) => i * step);
    } else {
      values = probs;
      labels = probs.map((_, i) => i);
    }

    if (frameChart) {
      frameChart.destroy();
    }

    // Colour each point: red if ≥ 0.5, steelblue if < 0.5
    const pointColors = values.map((v) =>
      v >= 0.5 ? "rgba(192, 57, 43, 0.7)" : "rgba(30, 77, 145, 0.5)"
    );

    frameChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "CNN Probability",
            data: values,
            borderColor: "rgba(30, 77, 145, 0.8)",
            borderWidth: 1.5,
            pointRadius: probs.length > 100 ? 0 : 3,
            pointBackgroundColor: pointColors,
            fill: {
              target: "origin",
              above: "rgba(192, 57, 43, 0.06)",
            },
            tension: 0.3,
          },
          {
            // Decision threshold line
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
    runBtn.disabled = on;
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
