/**
 * Match Upload & Lineup Setup Controller (upload.js)
 */

(function () {
    'use strict';

    function setMode(mode) {
        const pExisting = document.getElementById('panel-existing');
        const pCustom = document.getElementById('panel-custom');
        const lExisting = document.getElementById('label-existing');
        const lCustom = document.getElementById('label-custom');

        if (pExisting) pExisting.classList.toggle('active', mode === 'existing');
        if (pCustom) pCustom.classList.toggle('active', mode === 'custom');
        if (lExisting) lExisting.classList.toggle('active', mode === 'existing');
        if (lCustom) lCustom.classList.toggle('active', mode === 'custom');

        const homeName = document.getElementById('home_team_name');
        const homeShort = document.getElementById('home_team_short');
        const awayName = document.getElementById('away_team_name');
        const awayShort = document.getElementById('away_team_short');

        if (homeName) homeName.required = (mode === 'custom');
        if (homeShort) homeShort.required = (mode === 'custom');
        if (awayName) awayName.required = (mode === 'custom');
        if (awayShort) awayShort.required = (mode === 'custom');
    }

    function toggleNewTeam(side, value) {
        const fields = document.getElementById('new-team-' + side);
        if (!fields) return;
        const isNew = (value === '__new__');
        fields.classList.toggle('active', isNew);

        const nameField = document.getElementById('new_' + side + '_team_name');
        const shortField = document.getElementById('new_' + side + '_team_short');
        const colorField = document.getElementById('new_' + side + '_team_primary_color');
        if (nameField) nameField.required = isNew;
        if (shortField) shortField.required = isNew;
        if (colorField) colorField.required = isNew;
    }

    function assignSwatch(hex, fieldName) {
        if (!fieldName) return;
        document.querySelectorAll('input[name="' + fieldName + '"]').forEach(input => {
            input.value = hex;
        });
        if (window.Studio && window.Studio.showToast) {
            window.Studio.showToast(`Assigned ${hex} to ${fieldName.replace(/_/g, ' ')}`, 'success', 2000);
        }
    }

    function renderSwatches(data) {
        const panel = document.getElementById('color-swatch-panel');
        if (!panel) return;

        const allSwatches = [
            ...(data.outfield || []).map(hex => ({ hex, group: 'Outfield' })),
            ...(data.goalkeeper || []).map(hex => ({ hex, group: 'Goalkeeper' })),
        ];

        if (allSwatches.length === 0) {
            panel.innerHTML = '<p style="font-size:13px; color:var(--text-muted);">No distinct kit clusters detected in sampled frames. Select colors manually above.</p>';
            return;
        }

        const targets = [
            ['home_kit_color', 'Team A Outfield'],
            ['away_kit_color', 'Team B Outfield'],
            ['home_gk_kit_color', 'Team A GK'],
            ['away_gk_kit_color', 'Team B GK'],
        ];

        panel.innerHTML = '<p style="font-size:12.5px; color:#34d399; margin-bottom:12px; font-weight:600;">✓ Colors detected from footage. Assign each swatch to its corresponding team strip:</p>' +
            '<div style="display:flex; gap:16px; flex-wrap:wrap;">' +
            allSwatches.map(s => `
                <div style="text-align:center; background:rgba(255,255,255,0.03); border:1px solid var(--border-glass); border-radius:10px; padding:10px;">
                    <div style="width:44px; height:44px; border-radius:50%; background:${s.hex}; border:2px solid rgba(255,255,255,0.25); margin:0 auto 6px; box-shadow:0 0 10px ${s.hex}40;"></div>
                    <div style="font-size:11px; color:var(--text-muted); margin-bottom:6px; font-weight:500;">${s.group}</div>
                    <select onchange="window.assignSwatch('${s.hex}', this.value); this.value='';" style="font-size:11.5px; padding:5px 8px; background:#0f1519; color:#f8fafc; border:1px solid var(--border-glass); border-radius:6px;">
                        <option value="">Assign to…</option>
                        ${targets.map(([fieldName, label]) => `<option value="${fieldName}">${label}</option>`).join('')}
                    </select>
                </div>
            `).join('') +
            '</div>';
    }

    async function detectKitColors(endpointUrl) {
        const videoInput = document.getElementById('video');
        const btn = document.getElementById('detect-colors-btn');
        const panel = document.getElementById('color-swatch-panel');

        const url = endpointUrl || (btn ? btn.dataset.detectUrl : '');

        if (!videoInput || !videoInput.files.length) {
            alert('Please select a video file first.');
            return;
        }

        if (btn) {
            btn.disabled = true;
            btn.innerHTML = '<span>⏳ Sampling video frames for jersey colors…</span>';
        }
        if (panel) {
            panel.style.display = 'block';
            panel.innerHTML = '<p style="font-size:13px; color:var(--text-secondary);">Uploading sample clip to run CV palette clustering…</p>';
        }

        const formData = new FormData();
        formData.append('video', videoInput.files[0]);

        const csrfToken = (window.Studio && window.Studio.getCsrfToken) ? window.Studio.getCsrfToken() : '';

        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: { 'X-CSRFToken': csrfToken },
                body: formData,
            });
            const data = await response.json();

            if (data.error) {
                if (panel) panel.innerHTML = '<p style="font-size:13px; color:#f87171;">' + data.error + '</p>';
                return;
            }

            renderSwatches(data);
        } catch (err) {
            if (panel) panel.innerHTML = '<p style="font-size:13px; color:#f87171;">Detection failed: ' + err + '</p>';
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = '<span>🎨 Detect Kit Colors From Video</span>';
            }
        }
    }

    // Initialize on DOM Ready
    document.addEventListener('DOMContentLoaded', function () {
        setMode('existing');

        // File dropzone
        const videoInput = document.getElementById('video');
        const filePill = document.getElementById('selectedFilePill');
        const fileNameSpan = document.getElementById('selectedFileName');

        if (videoInput && filePill && fileNameSpan) {
            videoInput.addEventListener('change', function () {
                if (this.files && this.files.length > 0) {
                    const file = this.files[0];
                    const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
                    fileNameSpan.textContent = `${file.name} (${sizeMB} MB)`;
                    filePill.style.display = 'inline-flex';
                } else {
                    filePill.style.display = 'none';
                }
            });
        }

        // Form submit spinner
        const form = document.getElementById('upload-form');
        const submitBtn = document.getElementById('submitBtn');
        if (form && submitBtn) {
            form.addEventListener('submit', function () {
                submitBtn.classList.add('is-submitting');
                submitBtn.disabled = true;
            });
        }

        // Color detector button
        const detectBtn = document.getElementById('detect-colors-btn');
        if (detectBtn) {
            detectBtn.addEventListener('click', function () {
                detectKitColors(detectBtn.dataset.detectUrl);
            });
        }
    });

    // Expose to window for inline onchange / onclick handlers
    window.setMode = setMode;
    window.toggleNewTeam = toggleNewTeam;
    window.assignSwatch = assignSwatch;
    window.detectKitColors = detectKitColors;
})();
