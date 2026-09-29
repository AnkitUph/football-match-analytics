/**
 * Match Analytics Studio Interactivity Controller (results.js)
 */

// --- Match Studio Configuration ---
const cfg = window.MATCH_STUDIO_CONFIG || {};
const ANNOTATED_VIDEO_URL = cfg.annotatedVideoUrl || "";
const RAW_VIDEO_URL = cfg.rawVideoUrl || "";
const HOME_COLOR = cfg.homeColor || "#ef4444";
const AWAY_COLOR = cfg.awayColor || "#eab308";
const HOME_TEAM_NAME = cfg.homeTeamName || "Home";
const AWAY_TEAM_NAME = cfg.awayTeamName || "Away";
const HOME_TEAM_SHORT = cfg.homeTeamShort || "HOME";
const AWAY_TEAM_SHORT = cfg.awayTeamShort || "AWAY";
const MATCH_PUBLIC_ID = cfg.matchPublicId || "";
const ALL_PLAYERS = cfg.allPlayers || [];

// --- Match Video Replay Controller ---
let currentVideoMode = ANNOTATED_VIDEO_URL ? 'annotated' : 'raw';
const videoEl = document.getElementById('replay-video');
const hudTime = document.getElementById('hud-time');
const hudFrame = document.getElementById('hud-frame');
const hudModeTag = document.getElementById('hud-mode-tag');

        function formatTime(seconds) {
            if (isNaN(seconds)) return "00:00";
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            const tenths = Math.floor((seconds % 1) * 10);
            return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${tenths}`;
        }

        window.switchVideoMode = function (mode) {
            if (!videoEl) return;
            const targetUrl = mode === 'annotated' ? ANNOTATED_VIDEO_URL : RAW_VIDEO_URL;
            if (!targetUrl) return;

            const prevTime = videoEl.currentTime;
            const wasPlaying = !videoEl.paused;

            videoEl.src = targetUrl;
            videoEl.currentTime = prevTime;
            if (wasPlaying) {
                videoEl.play().catch(() => { });
            }

            currentVideoMode = mode;
            document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
            const activeBtn = document.getElementById(mode === 'annotated' ? 'btn-mode-annotated' : 'btn-mode-raw');
            if (activeBtn) activeBtn.classList.add('active');

            if (hudModeTag) {
                hudModeTag.textContent = mode === 'annotated' ? 'AI Tracking Active' : 'Raw Video';
            }
        };

        window.setVideoSpeed = function (speed) {
            if (!videoEl) return;
            videoEl.playbackRate = speed;
            document.querySelectorAll('.speed-pill').forEach(pill => {
                pill.classList.toggle('active', parseFloat(pill.dataset.speed) === speed);
            });
        };

        window.toggleVideoLoop = function () {
            if (!videoEl) return;
            videoEl.loop = !videoEl.loop;
            const loopBtn = document.getElementById('btn-video-loop');
            if (loopBtn) {
                loopBtn.classList.toggle('active', videoEl.loop);
                loopBtn.innerHTML = videoEl.loop ? '🔁 Loop: On' : '🔁 Loop: Off';
            }
        };

        window.seekVideoToFrame = function (frameIdx) {
            if (!videoEl) return;
            if (frameIdx == null || frameIdx < 0) return;
            // Switch to overview tab so the replay widget is visible
            showTab('overview');
            // 25 fps nominal video rate; lead in by 0.6 seconds so user sees play develop
            const targetTime = Math.max(0, (frameIdx / 25.0) - 0.6);
            videoEl.currentTime = targetTime;
            videoEl.play().catch(() => { });

            // Visual pulse on the video container
            const wrap = document.querySelector('.replay-video-wrap');
            if (wrap) {
                wrap.style.boxShadow = '0 0 0 3px #4FAE79, 0 8px 24px rgba(79, 174, 121, 0.35)';
                setTimeout(() => { wrap.style.boxShadow = ''; }, 1200);
            }
        };

        if (videoEl) {
            videoEl.addEventListener('timeupdate', () => {
                if (hudTime) hudTime.textContent = formatTime(videoEl.currentTime);
                if (hudFrame) {
                    const estFrame = Math.round(videoEl.currentTime * 25);
                    hudFrame.textContent = `Frame: ${estFrame}`;
                }
            });
        }

        const homePlayers = JSON.parse(document.getElementById('home-players-data').textContent);
        const awayPlayers = JSON.parse(document.getElementById('away-players-data').textContent);

        function renderLineup(containerId, players) {
            const container = document.getElementById(containerId);
            if (!container) return;
            container.innerHTML = players.map((p, i) => {
                let badgeHtml = '';
                if (p.distance_is_real) {
                    badgeHtml = p.distance_confirmed
                        ? '<span class="badge-tracked" title="Identified & confirmed tracking data">Tracked</span>'
                        : '<span class="badge-unverified" title="Automatically linked from CV trajectory">Auto</span>';
                }
                const subStats = [];
                if (p.position) subStats.push(p.position);
                if (p.distance_is_real && p.distance_km) subStats.push(p.distance_km + ' km');
                if (p.passes_completed_is_real && p.passes_completed !== undefined) subStats.push(p.passes_completed + ' passes');

                return `
                <div class="lineup-row" onclick="openPlayer('${containerId}', ${i})">
                    <div class="lineup-jersey">${p.jersey_number}</div>
                    <div class="lineup-info">
                        <div class="lineup-name">
                            ${p.name}${badgeHtml}
                        </div>
                        <div class="lineup-position">${subStats.join(' · ')}</div>
                    </div>
                    <div class="lineup-rating">${p.rating}</div>
                </div>
            `}).join('');
        }
        renderLineup('lineup-home', homePlayers);
        renderLineup('lineup-away', awayPlayers);

        // --- FotMob Style Formation Lineup Controller ---
        let currentLineupView = 'pitch';
        let lineupPitchFilter = 'both';

        function detectFormation(players) {
            const gks = [];
            const defs = [];
            const mids = [];
            const fwds = [];

            players.forEach((p, idx) => {
                const item = { ...p, _origIndex: idx };
                const pos = (p.position || 'MID').toUpperCase();
                if (pos === 'GK') gks.push(item);
                else if (pos === 'DEF') defs.push(item);
                else if (pos === 'FWD') fwds.push(item);
                else mids.push(item);
            });

            const dCount = defs.length || 4;
            const mCount = mids.length || 3;
            const fCount = fwds.length || 3;
            const label = `${dCount}-${mCount}-${fCount}`;

            return { label, gks, defs, mids, fwds };
        }

        const homeFormation = detectFormation(homePlayers);
        const awayFormation = detectFormation(awayPlayers);

        function updateFormationHeaderBadges() {
            const bHome = document.getElementById('badge-formation-home');
            const bAway = document.getElementById('badge-formation-away');
            const topBanner = document.getElementById('top-banner-formation');
            const botBanner = document.getElementById('bottom-banner-formation');
            const listHome = document.getElementById('list-formation-badge-home');
            const listAway = document.getElementById('list-formation-badge-away');
            const benchHome = document.getElementById('bench-home-summary');
            const benchAway = document.getElementById('bench-away-summary');

            if (bHome) bHome.textContent = homeFormation.label;
            if (bAway) bAway.textContent = awayFormation.label;
            if (topBanner) topBanner.textContent = `(${awayFormation.label})`;
            if (botBanner) botBanner.textContent = `(${homeFormation.label})`;
            if (listHome) listHome.textContent = homeFormation.label;
            if (listAway) listAway.textContent = awayFormation.label;

            if (benchHome) {
                benchHome.textContent = `${homeFormation.defs.length} Defenders · ${homeFormation.mids.length} Midfielders · ${homeFormation.fwds.length} Forwards`;
            }
            if (benchAway) {
                benchAway.textContent = `${awayFormation.defs.length} Defenders · ${awayFormation.mids.length} Midfielders · ${awayFormation.fwds.length} Forwards`;
            }
        }

        function createPitchNodeHtml(p, xPct, yPct, isHome) {
            const color = isHome ? HOME_COLOR : AWAY_COLOR;
            const isGk = p.position === 'GK';
            const rating = parseFloat(p.rating || 6.0).toFixed(1);
            let ratingClass = 'rating-high';
            if (rating < 6.5) ratingClass = 'rating-low';
            else if (rating < 7.0) ratingClass = 'rating-mid';

            let badges = '';
            if (p.goals > 0) {
                badges += `<span class="node-event-badge" title="${p.goals} Goal(s)">⚽${p.goals > 1 ? p.goals : ''}</span>`;
            }
            if (p.yellow_cards > 0) {
                badges += `<span class="node-event-badge" title="Yellow Card">🟨</span>`;
            }
            if (p.red_cards > 0) {
                badges += `<span class="node-event-badge" title="Red Card">🟥</span>`;
            }

            const containerId = isHome ? 'lineup-home' : 'lineup-away';

            return `
                <div class="pitch-player-node" style="left: ${xPct}%; top: ${yPct}%; --node-glow-color: ${color};" onclick="openPlayer('${containerId}', ${p._origIndex})" title="${p.name} (#${p.jersey_number}) - ${p.position} [Rating: ${rating}]">
                    <div class="node-jersey ${isGk ? 'gk-jersey' : ''}" style="background: ${color};">
                        <span>${p.jersey_number}</span>
                        <span class="node-rating-pill ${ratingClass}">${rating}</span>
                    </div>
                    <div class="node-name-chip">
                        <span>${p.name}</span>
                        ${badges}
                    </div>
                </div>
            `;
        }

        function renderFormationPitch(filter) {
            const layer = document.getElementById('formation-nodes-layer');
            const topBanner = document.getElementById('pitch-banner-top');
            const botBanner = document.getElementById('pitch-banner-bottom');
            if (!layer) return;

            let html = '';

            if (filter === 'both') {
                if (topBanner) topBanner.style.display = 'flex';
                if (botBanner) botBanner.style.display = 'flex';

                // Away Team (top half, facing down)
                awayFormation.gks.forEach((p, i) => {
                    const x = ((i + 1) / (awayFormation.gks.length + 1)) * 100;
                    html += createPitchNodeHtml(p, x, 9, false);
                });
                awayFormation.defs.forEach((p, i) => {
                    const x = ((i + 1) / (awayFormation.defs.length + 1)) * 100;
                    html += createPitchNodeHtml(p, x, 21, false);
                });
                awayFormation.mids.forEach((p, i) => {
                    const x = ((i + 1) / (awayFormation.mids.length + 1)) * 100;
                    html += createPitchNodeHtml(p, x, 32, false);
                });
                awayFormation.fwds.forEach((p, i) => {
                    const x = ((i + 1) / (awayFormation.fwds.length + 1)) * 100;
                    html += createPitchNodeHtml(p, x, 43, false);
                });

                // Home Team (bottom half, facing up)
                homeFormation.fwds.forEach((p, i) => {
                    const x = ((i + 1) / (homeFormation.fwds.length + 1)) * 100;
                    html += createPitchNodeHtml(p, x, 57, true);
                });
                homeFormation.mids.forEach((p, i) => {
                    const x = ((i + 1) / (homeFormation.mids.length + 1)) * 100;
                    html += createPitchNodeHtml(p, x, 68, true);
                });
                homeFormation.defs.forEach((p, i) => {
                    const x = ((i + 1) / (homeFormation.defs.length + 1)) * 100;
                    html += createPitchNodeHtml(p, x, 79, true);
                });
                homeFormation.gks.forEach((p, i) => {
                    const x = ((i + 1) / (homeFormation.gks.length + 1)) * 100;
                    html += createPitchNodeHtml(p, x, 91, true);
                });

            } else if (filter === 'home') {
                if (topBanner) topBanner.style.display = 'none';
                if (botBanner) botBanner.style.display = 'flex';

                homeFormation.gks.forEach((p, i) => {
                    const x = ((i + 1) / (homeFormation.gks.length + 1)) * 100;
                    html += createPitchNodeHtml(p, x, 90, true);
                });
                homeFormation.defs.forEach((p, i) => {
                    const x = ((i + 1) / (homeFormation.defs.length + 1)) * 100;
                    html += createPitchNodeHtml(p, x, 69, true);
                });
                homeFormation.mids.forEach((p, i) => {
                    const x = ((i + 1) / (homeFormation.mids.length + 1)) * 100;
                    html += createPitchNodeHtml(p, x, 45, true);
                });
                homeFormation.fwds.forEach((p, i) => {
                    const x = ((i + 1) / (homeFormation.fwds.length + 1)) * 100;
                    html += createPitchNodeHtml(p, x, 22, true);
                });

            } else if (filter === 'away') {
                if (topBanner) topBanner.style.display = 'flex';
                if (botBanner) botBanner.style.display = 'none';

                awayFormation.gks.forEach((p, i) => {
                    const x = ((i + 1) / (awayFormation.gks.length + 1)) * 100;
                    html += createPitchNodeHtml(p, x, 90, false);
                });
                awayFormation.defs.forEach((p, i) => {
                    const x = ((i + 1) / (awayFormation.defs.length + 1)) * 100;
                    html += createPitchNodeHtml(p, x, 69, false);
                });
                awayFormation.mids.forEach((p, i) => {
                    const x = ((i + 1) / (awayFormation.mids.length + 1)) * 100;
                    html += createPitchNodeHtml(p, x, 45, false);
                });
                awayFormation.fwds.forEach((p, i) => {
                    const x = ((i + 1) / (awayFormation.fwds.length + 1)) * 100;
                    html += createPitchNodeHtml(p, x, 22, false);
                });
            }

            layer.innerHTML = html;
        }

        function switchLineupView(view) {
            currentLineupView = view;
            const btnPitch = document.getElementById('btn-lineup-view-pitch');
            const btnList = document.getElementById('btn-lineup-view-list');
            const panePitch = document.getElementById('lineup-view-pitch-pane');
            const paneList = document.getElementById('lineup-view-list-pane');

            if (view === 'pitch') {
                if (btnPitch) btnPitch.classList.add('active');
                if (btnList) btnList.classList.remove('active');
                if (panePitch) panePitch.style.display = 'block';
                if (paneList) paneList.style.display = 'none';
                renderFormationPitch(lineupPitchFilter);
            } else {
                if (btnList) btnList.classList.add('active');
                if (btnPitch) btnPitch.classList.remove('active');
                if (paneList) paneList.style.display = 'block';
                if (panePitch) panePitch.style.display = 'none';
            }
        }

        function setLineupPitchFilter(filter) {
            lineupPitchFilter = filter;
            document.querySelectorAll('.lineup-filter-btn').forEach(btn => {
                btn.classList.toggle('active', btn.id === 'filter-lineup-' + filter);
            });
            renderFormationPitch(filter);
        }

        window.switchLineupView = switchLineupView;
        window.setLineupPitchFilter = setLineupPitchFilter;
        window.renderFormationPitch = renderFormationPitch;

        updateFormationHeaderBadges();
        renderFormationPitch('both');

        const statGroups = [
            {
                title: "Summary",
                stats: [
                    ["Rating", p => p.rating, null, "rating-value"],
                    ["Minutes Played", p => p.minutes_played, "'"],
                ],
            },
            {
                title: "Attacking",
                stats: [
                    ["Goals", p => p.goals],
                    ["Assists", p => p.assists],
                    ["Shots", p => p.shots],
                    ["Shots on Target", p => p.shots_on_target],
                    ["Expected Goals (xG)", p => p.xg],
                    ["Key Passes", p => p.key_passes],
                    ["Dribbles Completed", p => p.dribbles_completed],
                    ["Offsides", p => p.offsides],
                ],
            },
            {
                title: "Passing",
                stats: [
                    ["Passes Attempted", p => p.passes_attempted],
                    ["Passes Completed", p => p.passes_completed],
                    ["Pass Accuracy", p => p.pass_accuracy, "%"],
                ],
            },
            {
                title: "Defending",
                stats: [
                    ["Tackles", p => p.tackles],
                    ["Interceptions", p => p.interceptions],
                    ["Clearances", p => p.clearances],
                ],
            },
            {
                title: "Discipline",
                stats: [
                    ["Fouls Committed", p => p.fouls_committed],
                    ["Fouls Suffered", p => p.fouls_suffered],
                    ["Yellow Cards", p => p.yellow_cards],
                    ["Red Cards", p => p.red_cards],
                ],
            },
            {
                title: "Physical",
                stats: [
                    ["Distance Covered", p => p.distance_km, " km"],
                    ["Top Speed", p => p.top_speed, " km/h"],
                    ["Average Speed", p => p.average_speed, " km/h"],
                ],
            },
        ];

        const PLAYER_HEATMAP_URL_BASE = cfg.playerHeatmapBaseUrl || "";

        // Maps a stat's label to its "is this real?" flag and its
        // "is this confirmed vs. unverified-guess?" flag on the player
        // row. Most real stats share distance_confirmed (they all come
        // from the same TrackPlayerIdentification row) — Goals is the
        // exception: it's manually entered via MatchGoal, always
        // confirmed the moment it's real, hence its own goals_confirmed
        // flag rather than reusing distance_confirmed. See
        // _dummy_player_rows in views.py for what's real vs. dummy.
        const TRACKED_STAT_FLAGS = {
            'Rating': { real: 'rating_is_real', confirmed: 'distance_confirmed' },
            'Distance Covered': { real: 'distance_is_real', confirmed: 'distance_confirmed' },
            'Top Speed': { real: 'top_speed_is_real', confirmed: 'distance_confirmed' },
            'Average Speed': { real: 'average_speed_is_real', confirmed: 'distance_confirmed' },
            'Passes Completed': { real: 'passes_completed_is_real', confirmed: 'distance_confirmed' },
            'Passes Attempted': { real: 'passes_attempted_is_real', confirmed: 'distance_confirmed' },
            'Pass Accuracy': { real: 'pass_accuracy_is_real', confirmed: 'distance_confirmed' },
            'Shots': { real: 'shots_is_real', confirmed: 'distance_confirmed' },
            'Shots on Target': { real: 'shots_on_target_is_real', confirmed: 'distance_confirmed' },
            'Expected Goals (xG)': { real: 'xg_is_real', confirmed: 'distance_confirmed' },
            'Key Passes': { real: 'key_passes_is_real', confirmed: 'distance_confirmed' },
            'Dribbles Completed': { real: 'dribbles_completed_is_real', confirmed: 'distance_confirmed' },
            'Tackles': { real: 'tackles_is_real', confirmed: 'distance_confirmed' },
            'Interceptions': { real: 'interceptions_is_real', confirmed: 'distance_confirmed' },
            'Clearances': { real: 'clearances_is_real', confirmed: 'distance_confirmed' },
            'Goals': { real: 'goals_is_real', confirmed: 'goals_confirmed' },
        };
        // label -> [label of the real stat that, when real, makes THIS
        // label's dummy value worth flagging as "(estimated)" since it
        // has no real counterpart of its own]
        const ESTIMATED_WHEN_REAL = {
            'Assists': 'passes_completed_is_real',
            'Fouls Committed': 'distance_is_real',
            'Fouls Suffered': 'distance_is_real',
            'Yellow Cards': 'distance_is_real',
            'Red Cards': 'distance_is_real',
            'Offsides': 'distance_is_real',
        };

        function statLabelSuffix(label, p) {
            const flags = TRACKED_STAT_FLAGS[label];
            if (flags && p[flags.real]) {
                return p[flags.confirmed]
                    ? ' <span style="color:#4FAE79;font-weight:700;font-size:10px;">(tracked)</span>'
                    : ' <span style="color:#E0B34E;font-weight:700;font-size:10px;">(tracked, unverified)</span>';
            }
            const estimatedFlag = ESTIMATED_WHEN_REAL[label];
            if (estimatedFlag && p[estimatedFlag]) {
                return ' <span style="color:#999;font-weight:700;font-size:10px;" title="No real counterpart is detected for this stat — still an estimated placeholder">(estimated)</span>';
            }
            return '';
        }

        let currentModalPlayer = null;

        function openPlayer(containerId, index) {
            const isHome = containerId === 'lineup-home';
            const players = isHome ? homePlayers : awayPlayers;
            const teamName = isHome ? HOME_TEAM_NAME : AWAY_TEAM_NAME;
            const p = players[index];
            currentModalPlayer = p;

            document.getElementById('modal-jersey').textContent = p.jersey_number;
            document.getElementById('modal-name').textContent = p.name;
            document.getElementById('modal-meta').textContent = p.position + ' · ' + teamName;

            const attackDirEl = document.getElementById('modal-attack-dir');
            if (attackDirEl) {
                attackDirEl.textContent = isHome ? 'Attacking Right →' : '← Attacking Left';
            }

            const container = document.getElementById('modal-stat-groups');
            container.innerHTML = statGroups.map(group => `
                <div class="stat-group">
                    <div class="stat-group-title">${group.title}</div>
                    ${group.stats.map(([label, getValue, suffix, extraClass]) => `
                        <div class="stat-row">
                            <span class="stat-row-label">${label}${statLabelSuffix(label, p)}</span>
                            <span class="stat-row-value ${extraClass || ''}">${getValue(p)}${suffix || ''}</span>
                        </div>
                    `).join('')}
                </div>
            `).join('');

            // Per-player heatmap
            const heatmapWrap = document.getElementById('modal-heatmap-wrap');
            heatmapWrap.className = 'modal-pitch';
            if (p.distance_is_real && p.lineup_entry_id) {
                heatmapWrap.innerHTML = '<span class="modal-pitch-note">Loading heatmap&hellip;</span>';
                const img = new Image();
                img.onload = () => {
                    heatmapWrap.className = '';
                    heatmapWrap.innerHTML = '';
                    img.style.width = '100%';
                    img.style.borderRadius = '10px';
                    img.style.display = 'block';
                    heatmapWrap.appendChild(img);
                };
                img.onerror = () => {
                    heatmapWrap.innerHTML = '<span class="modal-pitch-note">Not enough tracked positions for a heatmap yet</span>';
                };
                img.src = `${PLAYER_HEATMAP_URL_BASE}${p.lineup_entry_id}/`;
            } else {
                heatmapWrap.innerHTML = '<span class="modal-pitch-note">'
                    + (p.distance_is_real ? 'Not enough tracked positions for a heatmap yet' : 'Not tracked yet — identify this player on /identify/ to see their heatmap')
                    + '</span>';
            }

            document.getElementById('player-modal').classList.add('open');
        }

        function closeModal() {
            const modal = document.getElementById('player-modal');
            if (modal) modal.classList.remove('open');
            currentModalPlayer = null;
        }

        function comparePlayerFromModal() {
            if (!currentModalPlayer) return;
            const targetId = currentModalPlayer.lineup_entry_id || currentModalPlayer.id || currentModalPlayer.name;
            const isHome = homePlayers.some(hp => (hp.lineup_entry_id === targetId || hp.name === currentModalPlayer.name));
            closeModal();
            showTab('comparison');
            if (!comparisonStudioInitialized) {
                initComparisonStudio();
            }
            if (isHome) {
                const sel1 = document.getElementById('select-player-1');
                if (sel1) sel1.value = targetId;
            } else {
                const sel2 = document.getElementById('select-player-2');
                if (sel2) sel2.value = targetId;
            }
            onPlayerSelectionChange();
        }

        function openPlayerByLineupId(lineupId) {
            if (!lineupId) return;
            const hIdx = homePlayers.findIndex(p => p.lineup_entry_id === lineupId || p.id === lineupId);
            if (hIdx !== -1) {
                openPlayer('lineup-home', hIdx);
                return;
            }
            const aIdx = awayPlayers.findIndex(p => p.lineup_entry_id === lineupId || p.id === lineupId);
            if (aIdx !== -1) {
                openPlayer('lineup-away', aIdx);
                return;
            }
        }
        window.openPlayerByLineupId = openPlayerByLineupId;
        window.openPlayer = openPlayer;
        window.closeModal = closeModal;
        window.comparePlayerFromModal = comparePlayerFromModal;

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeModal();
        });

        function showTab(tab) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.toggle('active', btn.dataset.tab === tab));
            document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.toggle('active', panel.id === 'tab-' + tab));
            if (tab !== 'overview' && videoEl && !videoEl.paused) {
                videoEl.pause();
            }
            if (tab === 'lineup') {
                if (currentLineupView === 'pitch') {
                    renderFormationPitch(lineupPitchFilter);
                }
            }
            if (tab === 'comparison') {
                if (!comparisonStudioInitialized) {
                    initComparisonStudio();
                } else {
                    renderPlayerRadar();
                    renderTeamRadar();
                }
            }
        }


        // --- Tactical Pitch Visualizer Controller (Shots, Passing Network, Team Shape) ---
        (function () {
            const shotsDataEl = document.getElementById('shots-data');
            const tacticalDataEl = document.getElementById('tactical-data');
            const passesDataEl = document.getElementById('passes-data');
            const shotsData = JSON.parse(shotsDataEl ? shotsDataEl.textContent || '[]' : '[]');
            const tacticalData = JSON.parse(tacticalDataEl ? tacticalDataEl.textContent || '{}' : '{}');
            const passesData = JSON.parse(passesDataEl ? passesDataEl.textContent || '[]' : '[]');

            const trajGroup = document.getElementById('shot-trajectories');
            const shotNodeGroup = document.getElementById('shot-nodes');
            const hullGroup = document.getElementById('tactical-hull-group');
            const linkGroup = document.getElementById('tactical-network-links');
            const netNodeGroup = document.getElementById('tactical-network-nodes');
            const passEventsGroup = document.getElementById('pass-events-group');
            const tooltip = document.getElementById('shot-map-tooltip');
            const pitchWrapper = document.getElementById('shot-pitch-wrapper');

            let currentMode = 'shots';
            let shotFilter = 'all';
            let tacticalFilter = 'both';
            let passEventFilter = 'all';
            let passTeamFilter = 'both';

            const homeColor = '#4FAE79';
            const awayColor = '#5B9BD5';
            const homeShort = HOME_TEAM_SHORT;
            const awayShort = AWAY_TEAM_SHORT;

            function pitchToSvg(px, py) {
                const numX = parseFloat(px);
                const numY = parseFloat(py);
                const validX = !isNaN(numX) ? numX : 0;
                const validY = !isNaN(numY) ? numY : 0;
                const clampedX = Math.max(-53.5, Math.min(53.5, validX));
                const clampedY = Math.max(-34.5, Math.min(34.5, validY));
                return {
                    x: 25 + (clampedX + 52.5) * 10,
                    y: 20 + (34.0 - clampedY) * 10,
                };
            }

            function clearLayers() {
                if (trajGroup) trajGroup.innerHTML = '';
                if (shotNodeGroup) shotNodeGroup.innerHTML = '';
                if (hullGroup) hullGroup.innerHTML = '';
                if (linkGroup) linkGroup.innerHTML = '';
                if (netNodeGroup) netNodeGroup.innerHTML = '';
                if (passEventsGroup) passEventsGroup.innerHTML = '';
            }

            function updateTooltipPos(evt) {
                if (!pitchWrapper || !tooltip) return;
                const rect = pitchWrapper.getBoundingClientRect();
                const x = Math.max(10, Math.min(rect.width - 240, evt.clientX - rect.left + 15));
                const y = Math.max(10, Math.min(rect.height - 120, evt.clientY - rect.top + 15));
                tooltip.style.left = `${x}px`;
                tooltip.style.top = `${y}px`;
            }

            // 1. Shots Rendering
            function renderShots(filterSide) {
                if (!trajGroup || !shotNodeGroup) return;
                trajGroup.innerHTML = '';
                shotNodeGroup.innerHTML = '';

                shotsData.forEach((shot, idx) => {
                    const isGoal = (shot.outcome || '').toLowerCase() === 'goal';
                    if (filterSide === 'home' && shot.side !== 'home') return;
                    if (filterSide === 'away' && shot.side !== 'away') return;
                    if (filterSide === 'goal' && !isGoal) return;

                    const origin = pitchToSvg(shot.pitch_x, shot.pitch_y);
                    const target = (shot.target_goal_x != null && shot.target_goal_y != null)
                        ? pitchToSvg(shot.target_goal_x, shot.target_goal_y)
                        : null;

                    const teamColor = isGoal ? '#F1C40F' : (shot.side === 'home' ? homeColor : awayColor);
                    const xgVal = typeof shot.xg === 'number' ? shot.xg : parseFloat(shot.xg) || 0.05;
                    const radius = Math.max(6.5, Math.min(22, Math.sqrt(xgVal) * 30));

                    if (target) {
                        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                        line.setAttribute('x1', origin.x);
                        line.setAttribute('y1', origin.y);
                        line.setAttribute('x2', target.x);
                        line.setAttribute('y2', target.y);
                        line.setAttribute('stroke', teamColor);
                        line.setAttribute('stroke-width', isGoal ? '2.5' : '1.5');
                        line.setAttribute('stroke-dasharray', '4,4');
                        line.setAttribute('stroke-opacity', '0.65');
                        line.setAttribute('class', `shot-traj shot-traj-${idx}`);
                        trajGroup.appendChild(line);
                    }

                    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                    circle.setAttribute('cx', origin.x);
                    circle.setAttribute('cy', origin.y);
                    circle.setAttribute('r', radius);
                    circle.setAttribute('fill', teamColor);
                    circle.setAttribute('stroke', '#ffffff');
                    circle.setAttribute('stroke-width', '1.5');
                    circle.setAttribute('fill-opacity', '0.9');
                    circle.setAttribute('class', `shot-node shot-node-${idx}`);
                    circle.setAttribute('data-idx', idx);

                    circle.addEventListener('mouseenter', (e) => highlightShot(idx, e));
                    circle.addEventListener('mousemove', (e) => updateTooltipPos(e));
                    circle.addEventListener('mouseleave', () => unhighlightShot(idx));
                    circle.addEventListener('click', () => {
                        const row = document.querySelector(`.shot-row[data-shot-index="${idx}"]`);
                        if (row) {
                            row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                            row.classList.add('active');
                            setTimeout(() => row.classList.remove('active'), 1500);
                        }
                        if (shot.frame_idx != null) {
                            seekVideoToFrame(shot.frame_idx);
                        }
                    });

                    shotNodeGroup.appendChild(circle);
                });
            }

            function highlightShot(idx, evt) {
                const shot = shotsData[idx];
                if (!shot) return;

                document.querySelectorAll('.shot-traj').forEach(l => l.classList.add('dimmed'));
                const activeLine = document.querySelector(`.shot-traj-${idx}`);
                if (activeLine) {
                    activeLine.classList.remove('dimmed');
                    activeLine.classList.add('active');
                }
                const activeNode = document.querySelector(`.shot-node-${idx}`);
                if (activeNode) activeNode.classList.add('active');

                document.querySelectorAll('.shot-row').forEach(r => r.classList.remove('active'));
                const activeRow = document.querySelector(`.shot-row[data-shot-index="${idx}"]`);
                if (activeRow) activeRow.classList.add('active');

                if (tooltip) {
                    const inferredHtml = shot.player_inferred ? ' <span style="opacity:0.7;font-size:10px;">(inferred)</span>' : '';
                    const distHtml = shot.distance_m ? ` &bull; ${Number(shot.distance_m).toFixed(1)}m` : '';
                    const speedHtml = shot.speed_mps ? ` &bull; ${Number(shot.speed_mps).toFixed(1)}m/s` : '';
                    const xgNum = typeof shot.xg === 'number' ? shot.xg : parseFloat(shot.xg) || 0;
                    tooltip.innerHTML = `
                        <div class="shot-tooltip-player">${shot.player}${inferredHtml}</div>
                        <div class="shot-tooltip-meta">${shot.minute}' &bull; ${shot.side ? shot.side.toUpperCase() : ''} &bull; <span class="shot-tooltip-highlight">xG ${xgNum.toFixed(2)}</span></div>
                        <div class="shot-tooltip-meta">${shot.outcome}${distHtml}${speedHtml}</div>
                    `;
                    tooltip.style.display = 'block';
                    updateTooltipPos(evt);
                }
            }

            function unhighlightShot(idx) {
                document.querySelectorAll('.shot-traj').forEach(l => l.classList.remove('dimmed', 'active'));
                document.querySelectorAll('.shot-node').forEach(n => n.classList.remove('active'));
                document.querySelectorAll('.shot-row').forEach(r => r.classList.remove('active'));
                if (tooltip) tooltip.style.display = 'none';
            }

            // 2. Passing Network Rendering
            function renderPassingNetwork(teamFilter) {
                if (!linkGroup || !netNodeGroup || !tacticalData) return;
                linkGroup.innerHTML = '';
                netNodeGroup.innerHTML = '';

                const sidesToRender = teamFilter === 'both' ? ['home', 'away'] : [teamFilter];

                sidesToRender.forEach(side => {
                    const tData = tacticalData[side];
                    if (!tData || !tData.nodes) return;

                    const teamColor = side === 'home' ? homeColor : awayColor;
                    const nodeMap = {};
                    tData.nodes.forEach(n => { nodeMap[n.id] = n; });

                    if (tData.links) {
                        tData.links.forEach(link => {
                            const src = nodeMap[link.source];
                            const tgt = nodeMap[link.target];
                            if (!src || !tgt) return;

                            const p1 = pitchToSvg(src.x, src.y);
                            const p2 = pitchToSvg(tgt.x, tgt.y);
                            const weight = Math.max(1.5, Math.min(6.5, Math.sqrt(link.count) * 1.8));

                            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                            line.setAttribute('x1', p1.x);
                            line.setAttribute('y1', p1.y);
                            line.setAttribute('x2', p2.x);
                            line.setAttribute('y2', p2.y);
                            line.setAttribute('stroke', teamColor);
                            line.setAttribute('stroke-width', weight);
                            line.setAttribute('stroke-opacity', '0.5');
                            line.setAttribute('stroke-linecap', 'round');
                            line.setAttribute('class', 'tactical-link');

                            line.addEventListener('mouseenter', (e) => {
                                line.setAttribute('stroke-opacity', '1');
                                if (tooltip) {
                                    tooltip.innerHTML = `
                                        <div class="shot-tooltip-player">${link.source_name} &harr; ${link.target_name}</div>
                                        <div class="shot-tooltip-meta"><span class="shot-tooltip-highlight">${link.count} Passes</span> &bull; ${link.completed} Completed</div>
                                        <div style="font-size:10px;color:#aaa;margin-top:4px;">Click to seek video replay</div>
                                    `;
                                    tooltip.style.display = 'block';
                                    updateTooltipPos(e);
                                }
                            });
                            line.addEventListener('mousemove', (e) => updateTooltipPos(e));
                            line.addEventListener('mouseleave', () => {
                                line.setAttribute('stroke-opacity', '0.5');
                                if (tooltip) tooltip.style.display = 'none';
                            });
                            line.addEventListener('click', () => {
                                if (link.first_frame) seekVideoToFrame(link.first_frame);
                            });

                            linkGroup.appendChild(line);
                        });
                    }

                    tData.nodes.forEach(node => {
                        const pt = pitchToSvg(node.x, node.y);
                        const radius = Math.max(12, Math.min(20, 12 + Math.sqrt(node.passes_made || 0) * 1.2));

                        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                        g.setAttribute('class', 'tactical-node');
                        g.setAttribute('transform', `translate(${pt.x}, ${pt.y})`);

                        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                        circle.setAttribute('r', radius);
                        circle.setAttribute('fill', teamColor);
                        circle.setAttribute('stroke', '#ffffff');
                        circle.setAttribute('stroke-width', '2');
                        circle.setAttribute('fill-opacity', '0.92');
                        g.appendChild(circle);

                        const txt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                        txt.setAttribute('text-anchor', 'middle');
                        txt.setAttribute('dy', '4');
                        txt.setAttribute('fill', '#ffffff');
                        txt.setAttribute('font-size', '11');
                        txt.setAttribute('font-weight', '800');
                        txt.setAttribute('font-family', 'sans-serif');
                        txt.textContent = node.jersey != null ? node.jersey : '';
                        g.appendChild(txt);

                        const lbl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                        lbl.setAttribute('text-anchor', 'middle');
                        lbl.setAttribute('y', radius + 11);
                        lbl.setAttribute('fill', 'rgba(255, 255, 255, 0.9)');
                        lbl.setAttribute('font-size', '10');
                        lbl.setAttribute('font-weight', '600');
                        lbl.setAttribute('font-family', 'sans-serif');
                        lbl.textContent = node.name || '';
                        g.appendChild(lbl);

                        g.addEventListener('mouseenter', (e) => {
                            if (tooltip) {
                                const trackedBadge = node.is_tracked ? '<span style="color:#4FAE79;font-size:10px;">(Tracked)</span>' : '<span style="color:#e0b34e;font-size:10px;">(Estimated)</span>';
                                tooltip.innerHTML = `
                                    <div class="shot-tooltip-player">${node.name} #${node.jersey || ''} ${trackedBadge}</div>
                                    <div class="shot-tooltip-meta">${node.position} &bull; ${side.toUpperCase()}</div>
                                    <div class="shot-tooltip-meta">Passes: ${node.passes_made} made &bull; ${node.passes_received} received</div>
                                    <div style="font-size:10.5px;color:#aaa;">Avg Pitch Pos: (${node.x}m, ${node.y}m)</div>
                                `;
                                tooltip.style.display = 'block';
                                updateTooltipPos(e);
                            }
                        });
                        g.addEventListener('mousemove', (e) => updateTooltipPos(e));
                        g.addEventListener('mouseleave', () => {
                            if (tooltip) tooltip.style.display = 'none';
                        });
                        g.style.cursor = 'pointer';
                        g.addEventListener('click', () => {
                            if (node.lineup_id && window.openPlayerByLineupId) {
                                window.openPlayerByLineupId(node.lineup_id);
                            } else {
                                const pShot = shotsData.find(s => s.player && s.player.includes(node.name));
                                if (pShot && pShot.frame_idx != null) {
                                    seekVideoToFrame(pShot.frame_idx);
                                }
                            }
                        });

                        netNodeGroup.appendChild(g);
                    });
                });
            }

            // 3. Team Shape & Convex Hull Rendering
            function renderTeamShape(teamFilter) {
                if (!hullGroup || !tacticalData) return;
                hullGroup.innerHTML = '';

                const sidesToRender = teamFilter === 'both' ? ['home', 'away'] : [teamFilter];

                const homeAreaEl = document.getElementById('hud-home-area');
                const awayAreaEl = document.getElementById('hud-away-area');
                const homeDimEl = document.getElementById('hud-home-dim');
                const awayDimEl = document.getElementById('hud-away-dim');

                if (tacticalData.home && tacticalData.home.shape) {
                    const hs = tacticalData.home.shape;
                    if (homeAreaEl) homeAreaEl.textContent = `${hs.area_sqm} m²`;
                    if (homeDimEl) homeDimEl.textContent = `${hs.length_m}m × ${hs.width_m}m`;
                }
                if (tacticalData.away && tacticalData.away.shape) {
                    const as = tacticalData.away.shape;
                    if (awayAreaEl) awayAreaEl.textContent = `${as.area_sqm} m²`;
                    if (awayDimEl) awayDimEl.textContent = `${as.length_m}m × ${as.width_m}m`;
                }

                sidesToRender.forEach(side => {
                    const tData = tacticalData[side];
                    if (!tData || !tData.shape || !tData.shape.hull_vertices) return;

                    const teamColor = side === 'home' ? homeColor : awayColor;
                    const shape = tData.shape;

                    const ptsStr = shape.hull_vertices.map(v => {
                        const p = pitchToSvg(v.x, v.y);
                        return `${p.x},${p.y}`;
                    }).join(' ');

                    const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
                    poly.setAttribute('points', ptsStr);
                    poly.setAttribute('fill', teamColor);
                    poly.setAttribute('fill-opacity', '0.22');
                    poly.setAttribute('stroke', teamColor);
                    poly.setAttribute('stroke-width', '2.5');
                    poly.setAttribute('stroke-dasharray', '6,4');
                    poly.setAttribute('class', 'tactical-hull-poly');
                    hullGroup.appendChild(poly);

                    // Centroid Marker
                    const cPt = pitchToSvg(shape.centroid.x, shape.centroid.y);

                    const cCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                    cCircle.setAttribute('cx', cPt.x);
                    cCircle.setAttribute('cy', cPt.y);
                    cCircle.setAttribute('r', '6');
                    cCircle.setAttribute('fill', '#ffffff');
                    cCircle.setAttribute('stroke', teamColor);
                    cCircle.setAttribute('stroke-width', '2.5');
                    hullGroup.appendChild(cCircle);

                    const cRing = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                    cRing.setAttribute('cx', cPt.x);
                    cRing.setAttribute('cy', cPt.y);
                    cRing.setAttribute('r', '13');
                    cRing.setAttribute('fill', 'none');
                    cRing.setAttribute('stroke', teamColor);
                    cRing.setAttribute('stroke-width', '1.5');
                    cRing.setAttribute('stroke-dasharray', '3,3');
                    cRing.setAttribute('stroke-opacity', '0.7');
                    hullGroup.appendChild(cRing);

                    const cText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    cText.setAttribute('x', cPt.x);
                    cText.setAttribute('y', cPt.y - 14);
                    cText.setAttribute('text-anchor', 'middle');
                    cText.setAttribute('fill', '#ffffff');
                    cText.setAttribute('font-size', '10.5');
                    cText.setAttribute('font-weight', '700');
                    cText.setAttribute('font-family', 'sans-serif');
                    cText.textContent = `${side === 'home' ? homeShort : awayShort} Centroid`;
                    hullGroup.appendChild(cText);

                    // Player node positions inside the hull
                    if (tData.nodes) {
                        tData.nodes.forEach(node => {
                            const pt = pitchToSvg(node.x, node.y);
                            const nCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                            nCircle.setAttribute('cx', pt.x);
                            nCircle.setAttribute('cy', pt.y);
                            nCircle.setAttribute('r', '5');
                            nCircle.setAttribute('fill', teamColor);
                            nCircle.setAttribute('stroke', '#ffffff');
                            nCircle.setAttribute('stroke-width', '1.5');
                            hullGroup.appendChild(nCircle);

                            const nLbl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                            nLbl.setAttribute('x', pt.x);
                            nLbl.setAttribute('y', pt.y + 12);
                            nLbl.setAttribute('text-anchor', 'middle');
                            nLbl.setAttribute('fill', 'rgba(255,255,255,0.8)');
                            nLbl.setAttribute('font-size', '9');
                            nLbl.setAttribute('font-weight', '600');
                            nLbl.setAttribute('font-family', 'sans-serif');
                            nLbl.textContent = `#${node.jersey || ''}`;
                            hullGroup.appendChild(nLbl);
                        });
                    }
                });
            }

            // 4. Individual Pass Events Rendering
            function renderPassEvents() {
                if (!passEventsGroup) return;
                passEventsGroup.innerHTML = '';

                const filtered = passesData.filter(p => {
                    if (passTeamFilter === 'home' && p.side !== 'home') return false;
                    if (passTeamFilter === 'away' && p.side !== 'away') return false;

                    if (passEventFilter === 'completed' && !p.is_completed) return false;
                    if (passEventFilter === 'progressive' && !p.is_progressive) return false;
                    if (passEventFilter === 'crosses' && !p.is_cross) return false;
                    if (passEventFilter === 'key' && !p.is_key_pass) return false;

                    return true;
                });

                filtered.forEach(pass => {
                    const p1 = pitchToSvg(pass.start_x, pass.start_y);
                    const p2 = pitchToSvg(pass.end_x, pass.end_y);
                    const isHome = pass.side === 'home';
                    const teamColor = isHome ? homeColor : awayColor;
                    const markerId = isHome ? 'url(#arrow-home)' : 'url(#arrow-away)';

                    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                    line.setAttribute('x1', p1.x);
                    line.setAttribute('y1', p1.y);
                    line.setAttribute('x2', p2.x);
                    line.setAttribute('y2', p2.y);
                    line.setAttribute('stroke', teamColor);
                    line.setAttribute('stroke-width', pass.is_progressive ? '2.8' : '1.8');
                    line.setAttribute('stroke-opacity', pass.is_completed ? '0.8' : '0.45');
                    if (!pass.is_completed) {
                        line.setAttribute('stroke-dasharray', '4,4');
                    } else {
                        line.setAttribute('marker-end', markerId);
                    }
                    line.setAttribute('class', 'pass-vector-line');

                    const originCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                    originCircle.setAttribute('cx', p1.x);
                    originCircle.setAttribute('cy', p1.y);
                    originCircle.setAttribute('r', '3.5');
                    originCircle.setAttribute('fill', teamColor);
                    originCircle.setAttribute('stroke', '#ffffff');
                    originCircle.setAttribute('stroke-width', '1');

                    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                    g.setAttribute('class', 'pass-event-item');
                    g.style.cursor = 'pointer';
                    g.appendChild(line);
                    g.appendChild(originCircle);

                    let tags = [];
                    if (pass.is_completed) tags.push('<span style="color:#4FAE79;font-weight:700;">Completed</span>');
                    else tags.push('<span style="color:#e53e3e;font-weight:700;">Incomplete</span>');
                    if (pass.is_progressive) tags.push('<span style="color:#f6ad55;font-weight:700;">⚡ Progressive</span>');
                    if (pass.is_cross) tags.push('<span style="color:#63b3ed;font-weight:700;">🎯 Cross</span>');
                    if (pass.is_key_pass) tags.push('<span style="color:#f6e05e;font-weight:700;">⭐ Key Pass</span>');

                    g.addEventListener('mouseenter', (e) => {
                        line.setAttribute('stroke-opacity', '1');
                        line.setAttribute('stroke-width', '3.8');
                        if (tooltip) {
                            tooltip.innerHTML = `
                                <div class="shot-tooltip-player">${pass.passer} &rarr; ${pass.receiver}</div>
                                <div class="shot-tooltip-meta">Minute ${pass.minute}' &bull; ${pass.distance_m}m &bull; ${pass.speed_mps} m/s</div>
                                <div style="margin-top:3px;font-size:11px;">${tags.join(' ')}</div>
                                <div style="font-size:10px;color:#aaa;margin-top:4px;">Click to seek video replay</div>
                            `;
                            tooltip.style.display = 'block';
                            updateTooltipPos(e);
                        }
                    });
                    g.addEventListener('mousemove', (e) => updateTooltipPos(e));
                    g.addEventListener('mouseleave', () => {
                        line.setAttribute('stroke-opacity', pass.is_completed ? '0.8' : '0.45');
                        line.setAttribute('stroke-width', pass.is_progressive ? '2.8' : '1.8');
                        if (tooltip) tooltip.style.display = 'none';
                    });
                    g.addEventListener('click', () => {
                        if (pass.start_frame != null) {
                            seekVideoToFrame(pass.start_frame);
                        } else if (pass.frame_idx != null) {
                            seekVideoToFrame(pass.frame_idx);
                        }
                    });

                    passEventsGroup.appendChild(g);
                });
            }

            window.setPitchMode = function (mode) {
                currentMode = mode;
                clearLayers();

                document.querySelectorAll('.pitch-mode-btn').forEach(btn => {
                    btn.classList.toggle('active', btn.dataset.mode === mode || btn.id === 'btn-mode-' + mode);
                });

                const shotsFilters = document.getElementById('subfilters-shots');
                const passesFilters = document.getElementById('subfilters-passes');
                const tacticalFilters = document.getElementById('subfilters-tactical');
                const shotsLegend = document.getElementById('hud-shots-legend');
                const passesLegend = document.getElementById('hud-passes-legend');
                const eventsLegend = document.getElementById('hud-events-legend');
                const shapeMetrics = document.getElementById('hud-shape-metrics');
                const shotsList = document.getElementById('shots-list-container');
                const caption = document.getElementById('tactical-mode-caption');

                if (shotsFilters) shotsFilters.style.display = mode === 'shots' ? 'flex' : 'none';
                if (passesFilters) passesFilters.style.display = mode === 'events' ? 'flex' : 'none';
                if (tacticalFilters) tacticalFilters.style.display = (mode === 'passes' || mode === 'shape') ? 'flex' : 'none';
                if (shotsLegend) shotsLegend.style.display = mode === 'shots' ? 'flex' : 'none';
                if (passesLegend) passesLegend.style.display = mode === 'passes' ? 'flex' : 'none';
                if (eventsLegend) eventsLegend.style.display = mode === 'events' ? 'flex' : 'none';
                if (shapeMetrics) shapeMetrics.style.display = mode === 'shape' ? 'grid' : 'none';
                if (shotsList) shotsList.style.display = mode === 'shots' ? 'block' : 'none';

                if (mode === 'shots') {
                    if (caption) caption.textContent = 'Shot trajectories & xG model derived from computer vision tracking.';
                    renderShots(shotFilter);
                } else if (mode === 'passes') {
                    if (caption) caption.textContent = 'Passing network based on average pitch coordinates and pair pass frequencies.';
                    renderPassingNetwork(tacticalFilter);
                } else if (mode === 'events') {
                    if (caption) caption.textContent = 'Individual pass event vectors with progressive advancement and cross detection.';
                    renderPassEvents();
                } else if (mode === 'shape') {
                    if (caption) caption.textContent = 'Team tactical shape, length, width, and outfield convex hull compactness.';
                    renderTeamShape(tacticalFilter);
                }
            };

            window.setPassEventFilter = function (filter) {
                passEventFilter = filter;
                document.querySelectorAll('#subfilters-passes [data-pfilter]').forEach(b => {
                    b.classList.toggle('active', b.dataset.pfilter === filter);
                });
                renderPassEvents();
            };

            window.setPassTeamFilter = function (team) {
                passTeamFilter = team;
                document.querySelectorAll('#subfilters-passes [data-pteam]').forEach(b => {
                    b.classList.toggle('active', b.dataset.pteam === team);
                });
                renderPassEvents();
            };

            window.setTacticalTeamFilter = function (tFilter) {
                tacticalFilter = tFilter;
                document.querySelectorAll('#subfilters-tactical .shot-filter-btn').forEach(btn => {
                    btn.classList.toggle('active', btn.dataset.tfilter === tFilter);
                });
                if (currentMode === 'passes') {
                    renderPassingNetwork(tacticalFilter);
                } else if (currentMode === 'shape') {
                    renderTeamShape(tacticalFilter);
                }
            };

            document.querySelectorAll('.shot-row').forEach(row => {
                const idx = parseInt(row.getAttribute('data-shot-index'), 10);
                row.addEventListener('mouseenter', () => {
                    if (currentMode !== 'shots') return;
                    const node = document.querySelector(`.shot-node-${idx}`);
                    if (node && pitchWrapper) {
                        const nodeRect = node.getBoundingClientRect();
                        highlightShot(idx, {
                            clientX: nodeRect.left + nodeRect.width / 2,
                            clientY: nodeRect.top + nodeRect.height / 2
                        });
                    }
                });
                row.addEventListener('mouseleave', () => {
                    if (currentMode === 'shots') unhighlightShot(idx);
                });
                row.addEventListener('click', () => {
                    const shot = shotsData[idx];
                    if (shot && shot.frame_idx != null) {
                        seekVideoToFrame(shot.frame_idx);
                    }
                });
            });

            document.querySelectorAll('.pitch-mode-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    const mode = btn.dataset.mode || (btn.id ? btn.id.replace('btn-mode-', '') : null);
                    if (mode && typeof window.setPitchMode === 'function') {
                        window.setPitchMode(mode);
                    }
                });
            });

            document.querySelectorAll('#subfilters-shots .shot-filter-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    document.querySelectorAll('#subfilters-shots .shot-filter-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    shotFilter = btn.dataset.filter;
                    renderShots(shotFilter);
                    document.querySelectorAll('.shot-row').forEach(row => {
                        const side = row.dataset.side;
                        const outcome = row.dataset.outcome;
                        if (shotFilter === 'all') row.style.display = '';
                        else if (shotFilter === 'home') row.style.display = side === 'home' ? '' : 'none';
                        else if (shotFilter === 'away') row.style.display = side === 'away' ? '' : 'none';
                        else if (shotFilter === 'goal') row.style.display = outcome === 'goal' ? '' : 'none';
                    });
                });
            });

            renderShots('all');
        })();

        // --- Export Dropdown Controller ---
        function toggleExportMenu(e) {
            e.stopPropagation();
            const menu = document.getElementById('export-dropdown-menu');
            if (menu) menu.classList.toggle('show');
        }
        document.addEventListener('click', (e) => {
            const menu = document.getElementById('export-dropdown-menu');
            if (menu && !menu.contains(e.target)) {
                menu.classList.remove('show');
            }
        });

        // --- Integrated Head-to-Head Comparison Studio Controller ---
        let selectedP1 = null;
        let selectedP2 = null;
        let comparisonStudioInitialized = false;

        function switchComparisonSubtab(subtab) {
            const btnP = document.getElementById('btn-comp-players');
            const btnT = document.getElementById('btn-comp-teams');
            const paneP = document.getElementById('pane-comp-players');
            const paneT = document.getElementById('pane-comp-teams');

            if (subtab === 'players') {
                if (btnP) btnP.classList.add('active');
                if (btnT) btnT.classList.remove('active');
                if (paneP) paneP.classList.add('active');
                if (paneT) paneT.classList.remove('active');
                renderPlayerRadar();
            } else {
                if (btnT) btnT.classList.add('active');
                if (btnP) btnP.classList.remove('active');
                if (paneT) paneT.classList.add('active');
                if (paneP) paneP.classList.remove('active');
                renderTeamRadar();
            }
        }

        function initComparisonStudio() {
            if (comparisonStudioInitialized) return;
            const sel1 = document.getElementById('select-player-1');
            const sel2 = document.getElementById('select-player-2');
            if (!sel1 || !sel2 || !ALL_PLAYERS || ALL_PLAYERS.length === 0) return;

            sel1.innerHTML = '';
            sel2.innerHTML = '';

            const homeGroup1 = document.createElement('optgroup');
            homeGroup1.label = `${HOME_TEAM_NAME} (Home)`;
            const awayGroup1 = document.createElement('optgroup');
            awayGroup1.label = `${AWAY_TEAM_NAME} (Away)`;

            ALL_PLAYERS.forEach(p => {
                const opt1 = document.createElement('option');
                opt1.value = p.lineup_id || p.name;
                opt1.textContent = `#${p.jersey_number} ${p.name} (${p.position}) [Rating: ${p.rating}]`;
                if (p.team_side === 'HOME') {
                    homeGroup1.appendChild(opt1);
                } else {
                    awayGroup1.appendChild(opt1);
                }
            });

            const homeGroup2 = homeGroup1.cloneNode(true);
            const awayGroup2 = awayGroup1.cloneNode(true);

            sel1.appendChild(homeGroup1);
            sel1.appendChild(awayGroup1);

            sel2.appendChild(awayGroup2);
            sel2.appendChild(homeGroup2);

            const homePlayersList = ALL_PLAYERS.filter(p => p.team_side === 'HOME');
            const awayPlayersList = ALL_PLAYERS.filter(p => p.team_side === 'AWAY');

            if (homePlayersList.length > 0) {
                sel1.value = homePlayersList[0].lineup_id || homePlayersList[0].name;
            }
            if (awayPlayersList.length > 0) {
                const topScorer = awayPlayersList.find(p => p.goals > 0) || awayPlayersList[0];
                sel2.value = topScorer.lineup_id || topScorer.name;
            }

            comparisonStudioInitialized = true;
            onPlayerSelectionChange();
            renderTeamRadar();
        }

        function onPlayerSelectionChange() {
            const sel1 = document.getElementById('select-player-1');
            const sel2 = document.getElementById('select-player-2');
            if (!sel1 || !sel2 || !ALL_PLAYERS || ALL_PLAYERS.length === 0) return;

            const val1 = sel1.value;
            const val2 = sel2.value;

            selectedP1 = ALL_PLAYERS.find(p => (p.lineup_id || p.name).toString() === val1.toString()) || ALL_PLAYERS[0];
            selectedP2 = ALL_PLAYERS.find(p => (p.lineup_id || p.name).toString() === val2.toString()) || ALL_PLAYERS[1] || ALL_PLAYERS[0];

            updateHeroCards();
            updateStatBars();
            renderPlayerRadar();
            updateHeatmaps();
        }

        function updateHeroCards() {
            if (!selectedP1 || !selectedP2) return;

            const p1Name = document.getElementById('p1-name');
            const p1Pos = document.getElementById('p1-pos');
            const p1Team = document.getElementById('p1-team');
            const p1Jersey = document.getElementById('p1-jersey');
            const p1Init = document.getElementById('p1-initials');
            const p1Min = document.getElementById('p1-minutes');
            const p1Rat = document.getElementById('p1-rating');
            const leg1 = document.getElementById('legend-p1-label');

            if (p1Name) p1Name.textContent = selectedP1.name;
            if (p1Pos) p1Pos.textContent = selectedP1.position;
            if (p1Team) p1Team.textContent = selectedP1.team_name;
            if (p1Jersey) p1Jersey.textContent = `#${selectedP1.jersey_number}`;
            if (p1Init) p1Init.textContent = getPlayerInitials(selectedP1.name);
            if (p1Min) p1Min.textContent = `${selectedP1.minutes_played || 90} mins`;
            if (p1Rat) p1Rat.textContent = (selectedP1.rating || 6.0).toFixed(1);
            if (leg1) leg1.textContent = `${selectedP1.name} (#${selectedP1.jersey_number})`;

            const p2Name = document.getElementById('p2-name');
            const p2Pos = document.getElementById('p2-pos');
            const p2Team = document.getElementById('p2-team');
            const p2Jersey = document.getElementById('p2-jersey');
            const p2Init = document.getElementById('p2-initials');
            const p2Min = document.getElementById('p2-minutes');
            const p2Rat = document.getElementById('p2-rating');
            const leg2 = document.getElementById('legend-p2-label');

            if (p2Name) p2Name.textContent = selectedP2.name;
            if (p2Pos) p2Pos.textContent = selectedP2.position;
            if (p2Team) p2Team.textContent = selectedP2.team_name;
            if (p2Jersey) p2Jersey.textContent = `#${selectedP2.jersey_number}`;
            if (p2Init) p2Init.textContent = getPlayerInitials(selectedP2.name);
            if (p2Min) p2Min.textContent = `${selectedP2.minutes_played || 90} mins`;
            if (p2Rat) p2Rat.textContent = (selectedP2.rating || 6.0).toFixed(1);
            if (leg2) leg2.textContent = `${selectedP2.name} (#${selectedP2.jersey_number})`;
        }

        function getPlayerInitials(name) {
            if (!name) return "P";
            const parts = name.trim().split(/\s+/);
            if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
            return name.slice(0, 2).toUpperCase();
        }

        function updateStatBars() {
            if (!selectedP1 || !selectedP2) return;

            const metrics = [
                { key: 'goals', label: 'Goals Scored', max: 3, format: v => v },
                { key: 'shots_on_target', label: 'Shots on Target', max: 5, format: (v, p) => `${v} / ${p.shots || v}` },
                { key: 'xg', label: 'Expected Goals (xG)', max: 1.0, format: v => Number(v).toFixed(2) },
                { key: 'pass_accuracy', label: 'Pass Accuracy %', max: 100, format: v => `${Math.round(v)}%` },
                { key: 'passes_completed', label: 'Passes Completed', max: 50, format: (v, p) => `${v} / ${p.passes_attempted || v}` },
                { key: 'key_passes', label: 'Key Passes Created', max: 5, format: v => v },
                { key: 'tackles', label: 'Tackles Won', max: 6, format: v => v },
                { key: 'interceptions', label: 'Interceptions', max: 6, format: v => v },
                { key: 'distance_km', label: 'Distance Covered (km)', max: 13.0, format: v => `${Number(v).toFixed(1)} km` },
                { key: 'top_speed', label: 'Top Sprint Speed (km/h)', max: 35.0, format: v => `${Number(v).toFixed(1)} km/h` },
            ];

            const container = document.getElementById('stat-bars-container');
            if (!container) return;
            container.innerHTML = '';

            metrics.forEach(m => {
                const val1 = Number(selectedP1[m.key] || 0);
                const val2 = Number(selectedP2[m.key] || 0);
                const total = (val1 + val2) || 1;
                const pct1 = Math.round((val1 / total) * 100);
                const pct2 = 100 - pct1;

                const row = document.createElement('div');
                row.className = 'stat-bar-row';
                row.innerHTML = `
                    <div class="stat-bar-labels">
                        <span class="stat-val" style="${val1 > val2 ? 'color: #10b981;' : ''}">${m.format(val1, selectedP1)}</span>
                        <span class="stat-name">${m.label}</span>
                        <span class="stat-val" style="${val2 > val1 ? 'color: #10b981;' : ''}">${m.format(val2, selectedP2)}</span>
                    </div>
                    <div class="stat-dual-progress">
                        <div class="prog-bar-p1" style="width: ${pct1}%;"></div>
                        <div class="prog-bar-p2" style="width: ${pct2}%;"></div>
                    </div>
                `;
                container.appendChild(row);
            });
        }

        function renderPlayerRadar() {
            if (!selectedP1 || !selectedP2) return;
            const svg = document.getElementById('radar-svg');
            if (!svg) return;

            const cx = 200, cy = 200, r = 130;
            const dimensions = [
                { name: 'Shooting Threat', getVal: p => Math.min(1.0, ((p.shots || 0) * 0.2 + (p.xg || 0) * 1.5)) },
                { name: 'Passing Precision', getVal: p => Math.min(1.0, (p.pass_accuracy || 0) / 100.0) },
                { name: 'Creation / Assists', getVal: p => Math.min(1.0, ((p.key_passes || 0) * 0.3 + (p.assists || 0) * 0.4)) },
                { name: 'Defending', getVal: p => Math.min(1.0, ((p.tackles || 0) + (p.interceptions || 0) + (p.clearances || 0)) / 5.0) },
                { name: 'Work Rate', getVal: p => Math.min(1.0, (p.distance_km || 1.0) / 10.0) },
                { name: 'Pace & Speed', getVal: p => Math.min(1.0, (p.top_speed || 20.0) / 32.0) }
            ];

            const n = dimensions.length;
            let svgHtml = '';

            for (let level = 1; level <= 4; level++) {
                const curR = (r / 4) * level;
                let pts = [];
                for (let i = 0; i < n; i++) {
                    const angle = (Math.PI * 2 / n) * i - Math.PI / 2;
                    pts.push(`${cx + curR * Math.cos(angle)},${cy + curR * Math.sin(angle)}`);
                }
                svgHtml += `<polygon points="${pts.join(' ')}" fill="none" stroke="rgba(255,255,255,0.07)" stroke-width="1"/>`;
            }

            for (let i = 0; i < n; i++) {
                const angle = (Math.PI * 2 / n) * i - Math.PI / 2;
                const ax = cx + r * Math.cos(angle);
                const ay = cy + r * Math.sin(angle);
                svgHtml += `<line x1="${cx}" y1="${cy}" x2="${ax}" y2="${ay}" stroke="rgba(255,255,255,0.1)" stroke-width="1"/>`;

                const lx = cx + (r + 24) * Math.cos(angle);
                const ly = cy + (r + 24) * Math.sin(angle);
                const anchor = Math.abs(Math.cos(angle)) < 0.2 ? 'middle' : (Math.cos(angle) > 0 ? 'start' : 'end');
                svgHtml += `<text x="${lx}" y="${ly + 4}" fill="#94a3b8" font-size="11" font-weight="600" text-anchor="${anchor}">${dimensions[i].name}</text>`;
            }

            // Player 1 Polygon
            let p1Points = [];
            for (let i = 0; i < n; i++) {
                const angle = (Math.PI * 2 / n) * i - Math.PI / 2;
                const score = Math.max(0.15, dimensions[i].getVal(selectedP1));
                const px = cx + (r * score) * Math.cos(angle);
                const py = cy + (r * score) * Math.sin(angle);
                p1Points.push(`${px},${py}`);
            }
            svgHtml += `<polygon points="${p1Points.join(' ')}" fill="${selectedP1.team_color}" fill-opacity="0.25" stroke="${selectedP1.team_color}" stroke-width="2.5"/>`;

            // Player 2 Polygon
            let p2Points = [];
            for (let i = 0; i < n; i++) {
                const angle = (Math.PI * 2 / n) * i - Math.PI / 2;
                const score = Math.max(0.15, dimensions[i].getVal(selectedP2));
                const px = cx + (r * score) * Math.cos(angle);
                const py = cy + (r * score) * Math.sin(angle);
                p2Points.push(`${px},${py}`);
            }
            svgHtml += `<polygon points="${p2Points.join(' ')}" fill="${selectedP2.team_color}" fill-opacity="0.25" stroke="${selectedP2.team_color}" stroke-width="2.5"/>`;

            for (let i = 0; i < n; i++) {
                const [x1, y1] = p1Points[i].split(',');
                const [x2, y2] = p2Points[i].split(',');
                svgHtml += `<circle cx="${x1}" cy="${y1}" r="3.5" fill="${selectedP1.team_color}"/>`;
                svgHtml += `<circle cx="${x2}" cy="${y2}" r="3.5" fill="${selectedP2.team_color}"/>`;
            }

            svg.innerHTML = svgHtml;
        }

        function updateHeatmaps() {
            if (!selectedP1 || !selectedP2) return;

            const t1 = document.getElementById('hm-p1-title');
            const t2 = document.getElementById('hm-p2-title');
            if (t1) t1.textContent = `${selectedP1.name} (#${selectedP1.jersey_number})`;
            if (t2) t2.textContent = `${selectedP2.name} (#${selectedP2.jersey_number})`;

            const img1 = document.getElementById('p1-heatmap-img');
            const empty1 = document.getElementById('p1-heatmap-empty');
            if (img1 && empty1) {
                if (selectedP1.lineup_id) {
                    img1.src = `/matches/${MATCH_PUBLIC_ID}/player-heatmap/${selectedP1.lineup_id}/`;
                    img1.style.display = 'block';
                    empty1.style.display = 'none';
                    img1.onerror = () => { img1.style.display = 'none'; empty1.style.display = 'flex'; };
                } else {
                    img1.style.display = 'none';
                    empty1.style.display = 'flex';
                }
            }

            const img2 = document.getElementById('p2-heatmap-img');
            const empty2 = document.getElementById('p2-heatmap-empty');
            if (img2 && empty2) {
                if (selectedP2.lineup_id) {
                    img2.src = `/matches/${MATCH_PUBLIC_ID}/player-heatmap/${selectedP2.lineup_id}/`;
                    img2.style.display = 'block';
                    empty2.style.display = 'none';
                    img2.onerror = () => { img2.style.display = 'none'; empty2.style.display = 'flex'; };
                } else {
                    img2.style.display = 'none';
                    empty2.style.display = 'flex';
                }
            }
        }

        function renderTeamRadar() {
            const svg = document.getElementById('team-radar-svg');
            if (!svg) return;

            const cx = 200, cy = 200, r = 130;
            const dimensions = [
                { name: 'Attack & Shots', val1: 0.25, val2: 0.85 },
                { name: 'Conversion & xG', val1: 0.15, val2: 0.75 },
                { name: 'Possession Control', val1: 0.50, val2: 0.50 },
                { name: 'Passing Flow', val1: 0.35, val2: 0.65 },
                { name: 'Tackles & Duels', val1: 0.40, val2: 0.60 },
                { name: 'Work Rate Engine', val1: 0.48, val2: 0.55 },
            ];

            const n = dimensions.length;
            let svgHtml = '';

            for (let level = 1; level <= 4; level++) {
                const curR = (r / 4) * level;
                let pts = [];
                for (let i = 0; i < n; i++) {
                    const angle = (Math.PI * 2 / n) * i - Math.PI / 2;
                    pts.push(`${cx + curR * Math.cos(angle)},${cy + curR * Math.sin(angle)}`);
                }
                svgHtml += `<polygon points="${pts.join(' ')}" fill="none" stroke="rgba(255,255,255,0.07)" stroke-width="1"/>`;
            }

            for (let i = 0; i < n; i++) {
                const angle = (Math.PI * 2 / n) * i - Math.PI / 2;
                const ax = cx + r * Math.cos(angle);
                const ay = cy + r * Math.sin(angle);
                svgHtml += `<line x1="${cx}" y1="${cy}" x2="${ax}" y2="${ay}" stroke="rgba(255,255,255,0.1)" stroke-width="1"/>`;

                const lx = cx + (r + 24) * Math.cos(angle);
                const ly = cy + (r + 24) * Math.sin(angle);
                const anchor = Math.abs(Math.cos(angle)) < 0.2 ? 'middle' : (Math.cos(angle) > 0 ? 'start' : 'end');
                svgHtml += `<text x="${lx}" y="${ly + 4}" fill="#94a3b8" font-size="11" font-weight="600" text-anchor="${anchor}">${dimensions[i].name}</text>`;
            }

            let t1Pts = [];
            for (let i = 0; i < n; i++) {
                const angle = (Math.PI * 2 / n) * i - Math.PI / 2;
                const px = cx + (r * dimensions[i].val1) * Math.cos(angle);
                const py = cy + (r * dimensions[i].val1) * Math.sin(angle);
                t1Pts.push(`${px},${py}`);
            }
            svgHtml += `<polygon points="${t1Pts.join(' ')}" fill="${HOME_COLOR}" fill-opacity="0.25" stroke="${HOME_COLOR}" stroke-width="2.5"/>`;

            let t2Pts = [];
            for (let i = 0; i < n; i++) {
                const angle = (Math.PI * 2 / n) * i - Math.PI / 2;
                const px = cx + (r * dimensions[i].val2) * Math.cos(angle);
                const py = cy + (r * dimensions[i].val2) * Math.sin(angle);
                t2Pts.push(`${px},${py}`);
            }
            svgHtml += `<polygon points="${t2Pts.join(' ')}" fill="${AWAY_COLOR}" fill-opacity="0.25" stroke="${AWAY_COLOR}" stroke-width="2.5"/>`;

            svg.innerHTML = svgHtml;
        }

        window.switchComparisonSubtab = switchComparisonSubtab;
        window.onPlayerSelectionChange = onPlayerSelectionChange;
        window.initComparisonStudio = initComparisonStudio;
