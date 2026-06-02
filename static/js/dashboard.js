let logitsChart = null;

function getBackend() {
    return document.getElementById("backend").value;
}

function fmt(x, digits = 3) {
    if (x === null || x === undefined || x === "") return "-";
    const n = Number(x);
    if (Number.isNaN(n)) return String(x);
    return n.toFixed(digits);
}

function setText(id, value) {
    document.getElementById(id).textContent = value;
}

function drawLogitsChart(logits) {
    const ctx = document.getElementById("logits-chart");

    if (logitsChart) {
        logitsChart.destroy();
    }

    logitsChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            datasets: [{
                label: "Output logits",
                data: logits,
                borderWidth: 1,
                borderColor: "rgba(34, 211, 238, 0.9)",
                backgroundColor: "rgba(56, 189, 248, 0.48)"
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: "#eef4ff" }
                }
            },
            scales: {
                x: {
                    ticks: { color: "#eef4ff" },
                    grid: { color: "rgba(169, 182, 200, 0.18)" }
                },
                y: {
                    ticks: { color: "#eef4ff" },
                    grid: { color: "rgba(169, 182, 200, 0.18)" }
                }
            }
        }
    });
}

function renderResult(data) {
    if (!data.ok) {
        alert(data.error || "Có lỗi xảy ra.");
        return;
    }

    setText("predicted-label", data.predicted_label);
    setText("backend-used", data.backend);
    setText("true-label", data.true_label ?? "-");

    if (data.correct === null || data.correct === undefined) {
        setText("correct", "-");
    } else {
        setText("correct", data.correct ? "Đúng" : "Sai");
    }

    setText("latency", `${fmt(data.latency_ms, 3)} ms`);
    setText("throughput", `${fmt(data.throughput_fps, 2)} fps`);

    if (data.input_summary) {
        setText("input-range", `${data.input_summary.min} → ${data.input_summary.max}`);
    } else {
        setText("input-range", "-");
    }

    setText("top1-class", data.top1_class ?? "-");
    setText("top1-logit", data.top1_logit !== undefined ? `logit: ${fmt(data.top1_logit, 3)}` : "-");

    setText("top2-class", data.top2_class ?? "-");
    setText("top2-logit", data.top2_logit !== undefined ? `logit: ${fmt(data.top2_logit, 3)}` : "-");

    setText("margin", data.margin !== undefined ? fmt(data.margin, 3) : "-");

    setText("note", data.note || "");

    drawLogitsChart(data.logits || []);
}

function switchTab(name) {
    document.querySelectorAll(".tab").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.tab === name);
    });

    document.querySelectorAll(".tab-content").forEach(item => {
        item.classList.remove("active");
    });

    document.getElementById(`tab-${name}`).classList.add("active");
}

document.querySelectorAll(".tab").forEach(btn => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
});

document.getElementById("upload-form").addEventListener("submit", async (event) => {
    event.preventDefault();

    const formData = new FormData();
    const fileInput = document.getElementById("upload-file");

    if (!fileInput.files.length) {
        alert("Chưa chọn file test.");
        return;
    }

    formData.append("file", fileInput.files[0]);
    formData.append("true_label", document.getElementById("upload-true-label").value);
    formData.append("backend", getBackend());

    const response = await fetch("/api/predict_upload", {
        method: "POST",
        body: formData
    });

    renderResult(await response.json());
});

const btnRunSample = document.getElementById("btn-run-sample");
if (btnRunSample) {
    btnRunSample.addEventListener("click", async () => {
        const select = document.getElementById("sample-select");

        const response = await fetch("/api/predict_sample", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                backend: getBackend(),
                sample_path: select.value
            })
        });

        renderResult(await response.json());
    });
}

document.getElementById("btn-clear-cache").addEventListener("click", async () => {
    const response = await fetch("/api/clear_cache", { method: "POST" });
    const data = await response.json();
    alert(data.message || "Đã load lại.");
});

// Drawing canvas
const canvas = document.getElementById("draw-canvas");
const ctx = canvas.getContext("2d");
let drawing = false;

function resetCanvas() {
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function getPointerPos(event) {
    const rect = canvas.getBoundingClientRect();

    if (event.touches && event.touches.length > 0) {
        return {
            x: (event.touches[0].clientX - rect.left) * (canvas.width / rect.width),
            y: (event.touches[0].clientY - rect.top) * (canvas.height / rect.height)
        };
    }

    return {
        x: (event.clientX - rect.left) * (canvas.width / rect.width),
        y: (event.clientY - rect.top) * (canvas.height / rect.height)
    };
}

function startDraw(event) {
    event.preventDefault();
    drawing = true;
    const p = getPointerPos(event);

    ctx.beginPath();
    ctx.moveTo(p.x, p.y);
}

function draw(event) {
    if (!drawing) return;

    event.preventDefault();
    const p = getPointerPos(event);

    ctx.lineWidth = 20;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#ffffff";
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
}

function stopDraw(event) {
    if (!drawing) return;
    event.preventDefault();
    drawing = false;
}

canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDraw);
canvas.addEventListener("mouseleave", stopDraw);

canvas.addEventListener("touchstart", startDraw, { passive: false });
canvas.addEventListener("touchmove", draw, { passive: false });
canvas.addEventListener("touchend", stopDraw, { passive: false });

document.getElementById("btn-clear-draw").addEventListener("click", resetCanvas);

document.getElementById("btn-predict-draw").addEventListener("click", async () => {
    const imageData = canvas.toDataURL("image/png");

    const response = await fetch("/api/predict_drawing", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            backend: getBackend(),
            true_label: document.getElementById("draw-true-label").value,
            image_data: imageData
        })
    });

    renderResult(await response.json());
});

document.getElementById("btn-save-draw").addEventListener("click", async () => {
    const label = document.getElementById("draw-true-label").value;

    if (label === "") {
        alert("Bạn cần nhập nhãn chính xác của số đã vẽ trước khi lưu.");
        return;
    }

    const imageData = canvas.toDataURL("image/png");

    const response = await fetch("/api/save_drawing", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            label: label,
            image_data: imageData
        })
    });

    const data = await response.json();

    if (!data.ok) {
        alert(data.error || "Không lưu được ảnh vẽ.");
        return;
    }

    alert(`${data.message}\nẢnh 28x28: ${data.mnist_28x28_path}\nẢnh raw: ${data.raw_path}`);
});

resetCanvas();
drawLogitsChart([0,0,0,0,0,0,0,0,0,0]);
