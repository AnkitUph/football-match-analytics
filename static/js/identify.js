/**
 * Player Identification and Tagging Controller (identify.js)
 */

(function () {
    'use strict';

    function initPlayerIdentification() {
        const grid = document.getElementById('grid');
        const saveStatus = document.getElementById('save-status');
        const saveBtn = document.getElementById('save-btn');
        const workbench = document.getElementById('identify-workbench');

        const SAVE_URL = workbench ? workbench.dataset.saveUrl : '';

        let tracks = [];
        const tracksScript = document.getElementById('tracks-data');
        if (tracksScript) {
            try {
                tracks = JSON.parse(tracksScript.textContent);
            } catch (e) {
                console.error("Failed to parse tracks data", e);
            }
        }

        function optionsHtml(track) {
            const opts = ['<option value="">Unassigned</option>'];
            const assignedId = track.assigned_lineup_entry_id;
            const topCandidates = track.top_candidates || [];
            const allOptions = track.lineup_options || [];

            if (topCandidates.length > 0) {
                opts.push('<optgroup label="✨ Top AI Suggestions">');
                topCandidates.forEach(cand => {
                    const selected = cand.lineup_entry_id === assignedId ? 'selected' : '';
                    opts.push(`<option value="${cand.lineup_entry_id}" ${selected}>#${cand.jersey_number} ${cand.player_name}</option>`);
                });
                opts.push('</optgroup>');

                opts.push('<optgroup label="📋 Full Squad Lineup">');
                allOptions.forEach(entry => {
                    const alreadyInTop = topCandidates.some(c => c.lineup_entry_id === entry.id);
                    const selected = (!alreadyInTop && entry.id === assignedId) ? 'selected' : '';
                    opts.push(`<option value="${entry.id}" ${selected}>#${entry.jersey_number} ${entry.player_name}</option>`);
                });
                opts.push('</optgroup>');
            } else {
                allOptions.forEach(entry => {
                    const selected = entry.id === assignedId ? 'selected' : '';
                    opts.push(`<option value="${entry.id}" ${selected}>#${entry.jersey_number} ${entry.player_name}</option>`);
                });
            }
            return opts.join('');
        }

        if (grid && tracks.length > 0) {
            grid.innerHTML = '';
            tracks.forEach(track => {
                const card = document.createElement('div');
                const cardClass = track.assigned_lineup_entry_id
                    ? (track.is_auto_assigned ? ' guessed' : ' assigned')
                    : '';
                card.className = 'player-card' + cardClass;
                card.dataset.trackId = track.track_id;

                const teamLabel = track.team === 'team_a' ? 'home' : 'away';
                const cropHtml = track.crop_b64
                    ? `<img src="data:image/jpeg;base64,${track.crop_b64}" alt="Tracked player crop">`
                    : `<span class="no-crop">No usable crop</span>`;

                const badgeHtml = track.assigned_lineup_entry_id
                    ? (track.is_auto_assigned
                        ? '<span class="badge badge-guess">AI Auto</span>'
                        : '<span class="badge badge-confirmed">Verified</span>')
                    : '';

                card.innerHTML = `
                    <div class="crop-wrap">${cropHtml}</div>
                    <div class="card-meta">
                        <span class="team-pill ${teamLabel}">${teamLabel}</span>
                        <span>${track.distance_km} km</span>
                    </div>
                    ${badgeHtml ? `<div>${badgeHtml}</div>` : ''}
                    <select data-track-id="${track.track_id}">
                        ${optionsHtml(track)}
                    </select>
                `;
                grid.appendChild(card);

                const select = card.querySelector('select');
                select.addEventListener('change', () => {
                    select.dataset.touched = 'true';
                    card.classList.remove('guessed');
                    card.classList.add('assigned');
                });
            });
        }

        if (saveBtn) {
            saveBtn.addEventListener('click', async () => {
                saveBtn.disabled = true;
                if (saveStatus) {
                    saveStatus.textContent = "Saving…";
                    saveStatus.className = "save-status";
                }

                const assignments = Array.from(grid.querySelectorAll('select')).map(sel => ({
                    track_id: parseInt(sel.dataset.trackId, 10),
                    lineup_entry_id: sel.value ? parseInt(sel.value, 10) : null,
                    confirmed: sel.dataset.touched === 'true',
                }));

                const csrfToken = (window.Studio && window.Studio.getCsrfToken) ? window.Studio.getCsrfToken() : '';

                try {
                    const resp = await fetch(SAVE_URL, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'X-CSRFToken': csrfToken
                        },
                        body: JSON.stringify({ assignments }),
                    });
                    const data = await resp.json();

                    if (!resp.ok) {
                        if (saveStatus) {
                            saveStatus.textContent = data.error || "Something went wrong saving assignments.";
                            saveStatus.className = "save-status error";
                        }
                        saveBtn.disabled = false;
                        return;
                    }

                    if (saveStatus) {
                        saveStatus.textContent = "✓ Saved — redirecting to results…";
                        saveStatus.className = "save-status success";
                    }
                    setTimeout(() => { window.location.href = data.redirect_url; }, 900);
                } catch (err) {
                    if (saveStatus) {
                        saveStatus.textContent = "Network error saving assignments.";
                        saveStatus.className = "save-status error";
                    }
                    saveBtn.disabled = false;
                }
            });
        }
    }

    document.addEventListener('DOMContentLoaded', initPlayerIdentification);
})();
