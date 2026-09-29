/**
 * Pitch Calibration Canvas & Keypoint Controller (calibrate.js)
 */

(function () {
    'use strict';

    function initCalibration() {
        const wrap = document.getElementById("calibration-workbench");
        if (!wrap) return;

        const FRAME_URL = wrap.dataset.frameUrl;
        const SAVE_URL = wrap.dataset.saveUrl;
        const SUGGEST_URL = wrap.dataset.suggestUrl;

        let LANDMARKS = [];
        const landmarksScript = document.getElementById("landmarks-data");
        if (landmarksScript) {
            try {
                LANDMARKS = JSON.parse(landmarksScript.textContent);
            } catch (e) {
                console.error("Failed to parse landmarks data", e);
            }
        }

        const DOT_COLORS = ["#10b981", "#06b6d4", "#f59e0b", "#f43f5e", "#8b5cf6", "#ec4899", "#3b82f6", "#14b8a6"];

        const img = document.getElementById("frame-img");
        const canvas = document.getElementById("frame-canvas");
        if (!img || !canvas) return;

        const ctx = canvas.getContext("2d");
        const frameStatus = document.getElementById("frame-status");
        const saveStatus = document.getElementById("save-status");
        const pointList = document.getElementById("point-list");
        const saveBtn = document.getElementById("save-btn");
        const smartAssistBtn = document.getElementById("smart-assist-btn");
        const frameIdxInput = document.getElementById("frame-idx-input");
        const loadFrameBtn = document.getElementById("load-frame-btn");
        const clearPointsBtn = document.getElementById("clear-points-btn");

        const MIN_POINTS = 4;
        let currentFrameIdx = parseInt(frameIdxInput ? frameIdxInput.value : 0, 10) || 0;
        let points = [];

        function usedLandmarkIds() {
            return new Set(points.map(p => p.landmark_id).filter(Boolean));
        }

        function landmarkOptionsHtml(selectedId) {
            const used = usedLandmarkIds();
            return LANDMARKS.map(l => {
                const disabled = (l.id !== selectedId && used.has(l.id)) ? "disabled" : "";
                const selected = (l.id === selectedId) ? "selected" : "";
                return `<option value="${l.id}" ${disabled} ${selected}>${l.label}</option>`;
            }).join("");
        }

        function resizeCanvasToImage() {
            canvas.width = img.clientWidth;
            canvas.height = img.clientHeight;
            redraw();
        }

        function redraw() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            if (!img.naturalWidth) return;
            const scaleX = canvas.width / img.naturalWidth;
            const scaleY = canvas.height / img.naturalHeight;
            points.forEach((p, i) => {
                const x = p.pixel_x * scaleX;
                const y = p.pixel_y * scaleY;
                const color = DOT_COLORS[i % DOT_COLORS.length];
                ctx.beginPath();
                ctx.arc(x, y, 10, 0, Math.PI * 2);
                ctx.fillStyle = color + '33';
                ctx.fill();

                ctx.beginPath();
                ctx.arc(x, y, 7, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();

                ctx.strokeStyle = 'rgba(0,0,0,0.5)';
                ctx.lineWidth = 1.5;
                ctx.stroke();

                ctx.fillStyle = '#020710';
                ctx.font = 'bold 10px Inter, sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(String(i + 1), x, y);
            });
        }

        function renderPointList() {
            if (!pointList) return;
            pointList.innerHTML = "";
            points.forEach((p, i) => {
                const li = document.createElement("li");
                li.className = "point-row";
                const confidenceTag = (typeof p.confidence === "number")
                    ? `<span class="confidence-tag">${Math.round(p.confidence * 100)}% conf.</span>`
                    : "";
                li.innerHTML = `
                    <span class="dot" style="background:${DOT_COLORS[i % DOT_COLORS.length]}">${i + 1}</span>
                    <select data-index="${i}">
                        <option value="">Select landmark…</option>
                        ${landmarkOptionsHtml(p.landmark_id)}
                    </select>
                    ${confidenceTag}
                    <button type="button" class="remove-btn" data-index="${i}" title="Remove point">&times;</button>
                `;
                pointList.appendChild(li);
            });

            pointList.querySelectorAll("select").forEach(sel => {
                sel.addEventListener("change", (e) => {
                    points[parseInt(e.target.dataset.index, 10)].landmark_id = e.target.value || null;
                    renderPointList();
                    updateSaveButtonState();
                });
            });

            pointList.querySelectorAll(".remove-btn").forEach(btn => {
                btn.addEventListener("click", (e) => {
                    points.splice(parseInt(e.target.dataset.index, 10), 1);
                    renderPointList();
                    redraw();
                    updateSaveButtonState();
                });
            });
        }

        function updateSaveButtonState() {
            if (saveBtn) {
                saveBtn.disabled = !(points.length >= MIN_POINTS && points.every(p => p.landmark_id));
            }
        }

        function loadFrame(frameIdx) {
            if (frameStatus) {
                frameStatus.textContent = "Loading frame…";
                frameStatus.className = "status-line";
            }
            img.onload = () => {
                currentFrameIdx = frameIdx;
                resizeCanvasToImage();
                if (frameStatus) frameStatus.textContent = `Loaded frame ${frameIdx}.`;
            };
            img.onerror = () => {
                if (frameStatus) {
                    frameStatus.textContent = "Could not load that frame.";
                    frameStatus.className = "status-line error";
                }
            };
            img.src = `${FRAME_URL}?frame_idx=${encodeURIComponent(frameIdx)}&_=${Date.now()}`;
        }

        if (loadFrameBtn) {
            loadFrameBtn.addEventListener("click", () => {
                const idx = parseInt(frameIdxInput.value, 10);
                if (Number.isNaN(idx) || idx < 0) {
                    if (frameStatus) {
                        frameStatus.textContent = "Enter a valid frame number.";
                        frameStatus.className = "status-line error";
                    }
                    return;
                }
                loadFrame(idx);
            });
        }

        if (clearPointsBtn) {
            clearPointsBtn.addEventListener("click", () => {
                points = [];
                renderPointList();
                redraw();
                updateSaveButtonState();
                if (saveStatus) saveStatus.textContent = "";
            });
        }

        img.addEventListener("click", (e) => {
            const rect = img.getBoundingClientRect();
            points.push({
                pixel_x: (e.clientX - rect.left) * (img.naturalWidth / img.clientWidth),
                pixel_y: (e.clientY - rect.top) * (img.naturalHeight / img.clientHeight),
                landmark_id: null,
            });
            renderPointList();
            redraw();
            updateSaveButtonState();
        });

        window.addEventListener("resize", resizeCanvasToImage);

        async function runSmartAssist() {
            if (smartAssistBtn) smartAssistBtn.disabled = true;
            if (saveStatus) {
                saveStatus.textContent = "Detecting pitch keypoints…";
                saveStatus.className = "status-line";
            }
            try {
                const resp = await fetch(`${SUGGEST_URL}?frame_idx=${encodeURIComponent(currentFrameIdx)}&_=${Date.now()}`);
                const data = await resp.json();
                if (!resp.ok) {
                    if (saveStatus) {
                        saveStatus.textContent = data.error || "Smart-assist failed.";
                        saveStatus.className = "status-line error";
                    }
                    return;
                }
                const already = usedLandmarkIds();
                let added = 0;
                for (const s of data.suggestions) {
                    if (already.has(s.landmark_id)) continue;
                    points.push({
                        pixel_x: s.pixel_x,
                        pixel_y: s.pixel_y,
                        landmark_id: s.landmark_id,
                        confidence: s.confidence
                    });
                    already.add(s.landmark_id);
                    added++;
                }
                renderPointList();
                redraw();
                updateSaveButtonState();
                if (saveStatus) {
                    saveStatus.textContent = added > 0
                        ? `✨ Smart-assist added ${added} point${added === 1 ? "" : "s"} — review before saving.`
                        : "Smart-assist found no new points.";
                    saveStatus.className = "status-line success";
                }
            } catch (err) {
                if (saveStatus) {
                    saveStatus.textContent = "Network error running smart-assist.";
                    saveStatus.className = "status-line error";
                }
            } finally {
                if (smartAssistBtn) smartAssistBtn.disabled = false;
            }
        }

        if (smartAssistBtn) {
            smartAssistBtn.addEventListener("click", runSmartAssist);
        }

        async function submitCalibration(confirmOverwrite) {
            if (saveBtn) saveBtn.disabled = true;
            if (saveStatus) {
                saveStatus.textContent = "Saving calibration…";
                saveStatus.className = "status-line";
            }

            const csrfToken = (window.Studio && window.Studio.getCsrfToken) ? window.Studio.getCsrfToken() : '';

            try {
                const resp = await fetch(SAVE_URL, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "X-CSRFToken": csrfToken
                    },
                    body: JSON.stringify({
                        calibration_frame: currentFrameIdx,
                        points: points.map(p => ({
                            pixel_x: p.pixel_x,
                            pixel_y: p.pixel_y,
                            landmark_id: p.landmark_id
                        })),
                        confirm_overwrite: !!confirmOverwrite,
                    }),
                });
                const data = await resp.json();
                if (!resp.ok) {
                    if (saveStatus) {
                        saveStatus.textContent = data.error || "Something went wrong.";
                        saveStatus.className = "status-line error";
                    }
                    updateSaveButtonState();
                    return;
                }
                if (data.needs_confirmation) {
                    if (window.confirm("This match already has real stats. Recalibrating will overwrite them. Continue?")) {
                        await submitCalibration(true);
                    } else {
                        if (saveStatus) {
                            saveStatus.textContent = "Calibration not saved.";
                            saveStatus.className = "status-line";
                        }
                        updateSaveButtonState();
                    }
                    return;
                }
                if (saveStatus) {
                    saveStatus.textContent = "✓ Calibration saved — recomputing pitch stats…";
                    saveStatus.className = "status-line success";
                }
                setTimeout(() => { window.location.href = data.redirect_url; }, 1200);
            } catch (err) {
                if (saveStatus) {
                    saveStatus.textContent = "Network error saving calibration.";
                    saveStatus.className = "status-line error";
                }
                updateSaveButtonState();
            }
        }

        if (saveBtn) {
            saveBtn.addEventListener("click", () => submitCalibration(false));
        }

        loadFrame(currentFrameIdx);
    }

    document.addEventListener("DOMContentLoaded", initCalibration);
})();
