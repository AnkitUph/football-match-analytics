/**
 * Real-time Match Processing Pipeline Monitor (processing.js)
 */

(function () {
    'use strict';

    function initProcessingMonitor() {
        const container = document.getElementById('processing-monitor');
        if (!container) return;

        const statusUrl = container.dataset.statusUrl;
        if (!statusUrl) return;

        const ringOuter = document.getElementById('ring-outer');
        const ringInner = document.getElementById('ring-inner');
        const statusText = document.getElementById('status-text');
        const progressFill = document.getElementById('progress-fill');
        const progressPercent = document.getElementById('progress-percent');
        const errorBox = document.getElementById('error-box');
        const hintText = document.getElementById('hint-text');
        const actions = document.getElementById('actions');

        let pollTimer = null;

        function updatePipeline(progress) {
            const steps = 6;
            const perStep = 100 / steps;
            for (let i = 0; i < steps; i++) {
                const el = document.getElementById('ps-' + i);
                if (!el) continue;
                const stepProgress = i * perStep;
                if (progress > stepProgress + perStep) {
                    el.className = 'pipe-step done-step';
                } else if (progress > stepProgress) {
                    el.className = 'pipe-step active';
                } else {
                    el.className = 'pipe-step';
                }
            }
        }

        function poll() {
            fetch(statusUrl)
                .then(r => r.json())
                .then(data => {
                    if (progressFill) progressFill.style.width = data.progress + '%';
                    if (progressPercent) progressPercent.textContent = data.progress + '%';
                    if (statusText) statusText.textContent = data.status_display;
                    updatePipeline(data.progress);

                    if (data.status === 'COMPLETED') {
                        clearInterval(pollTimer);
                        if (ringOuter) ringOuter.className = 'ring outer done';
                        if (ringInner) ringInner.className = 'ring inner done';
                        if (progressFill) progressFill.style.background = 'linear-gradient(90deg, #10b981, #06b6d4)';
                        if (hintText) hintText.textContent = '✓ Done! Taking you to the results…';
                        updatePipeline(100);
                        setTimeout(() => { window.location.href = data.results_url; }, 900);
                    } else if (data.status === 'FAILED') {
                        clearInterval(pollTimer);
                        if (ringOuter) ringOuter.className = 'ring outer failed';
                        if (ringInner) ringInner.className = 'ring inner failed';
                        if (errorBox) errorBox.style.display = 'block';
                        if (hintText) hintText.style.display = 'none';
                        if (actions) actions.style.display = 'block';
                    }
                })
                .catch(() => {});
        }

        poll();
        pollTimer = setInterval(poll, 2000);

        const initialProgress = parseFloat(container.dataset.initialProgress) || 0;
        updatePipeline(initialProgress);
    }

    document.addEventListener('DOMContentLoaded', initProcessingMonitor);
})();
