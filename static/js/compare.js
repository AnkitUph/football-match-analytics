/**
 * Match & Player Comparison Studio Controller (compare.js)
 */

        const cfg = window.COMPARE_STUDIO_CONFIG || {};
        const ALL_PLAYERS = cfg.allPlayers || [];
        const MATCH_PUBLIC_ID = cfg.matchPublicId || "";
        const HOME_COLOR = cfg.homeColor || "#10b981";
        const AWAY_COLOR = cfg.awayColor || "#06b6d4";

        let selectedP1 = null;
        let selectedP2 = null;

        // Initialize selectors
        function initSelectors() {
            const sel1 = document.getElementById('select-player-1');
            const sel2 = document.getElementById('select-player-2');

            sel1.innerHTML = '';
            sel2.innerHTML = '';

            const homeGroup = document.createElement('optgroup');
            homeGroup.label = (cfg.homeTeamName || "Home") + " (Home)";
            const awayGroup = document.createElement('optgroup');
            awayGroup.label = (cfg.awayTeamName || "Away") + " (Away)";

            ALL_PLAYERS.forEach(p => {
                const opt1 = document.createElement('option');
                opt1.value = p.lineup_id || p.name;
                opt1.textContent = `#${p.jersey_number} ${p.name} (${p.position}) [Rating: ${p.rating}]`;
                
                const opt2 = opt1.cloneNode(true);

                if (p.team_side === 'HOME') {
                    homeGroup.appendChild(opt1);
                } else {
                    awayGroup.appendChild(opt1);
                }
            });

            const homeGroup2 = homeGroup.cloneNode(true);
            const awayGroup2 = awayGroup.cloneNode(true);

            sel1.appendChild(homeGroup);
            sel1.appendChild(awayGroup);

            sel2.appendChild(awayGroup2);
            sel2.appendChild(homeGroup2);

            // Pre-select default best players
            const homePlayers = ALL_PLAYERS.filter(p => p.team_side === 'HOME');
            const awayPlayers = ALL_PLAYERS.filter(p => p.team_side === 'AWAY');

            if (homePlayers.length > 0) {
                sel1.value = homePlayers[0].lineup_id || homePlayers[0].name;
            }
            if (awayPlayers.length > 0) {
                // Select attacker/scorer if available
                const topScorer = awayPlayers.find(p => p.goals > 0) || awayPlayers[0];
                sel2.value = topScorer.lineup_id || topScorer.name;
            }

            onPlayerSelectionChange();
        }

        function onPlayerSelectionChange() {
            const sel1 = document.getElementById('select-player-1');
            const sel2 = document.getElementById('select-player-2');

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

            // Player 1
            document.getElementById('p1-name').textContent = selectedP1.name;
            document.getElementById('p1-pos').textContent = selectedP1.position;
            document.getElementById('p1-team').textContent = selectedP1.team_name;
            document.getElementById('p1-jersey').textContent = `#${selectedP1.jersey_number}`;
            document.getElementById('p1-initials').textContent = getInitials(selectedP1.name);
            document.getElementById('p1-minutes').textContent = `${selectedP1.minutes_played || 90} mins`;
            document.getElementById('p1-rating').textContent = (selectedP1.rating || 6.0).toFixed(1);
            document.getElementById('legend-p1-label').textContent = `${selectedP1.name} (#${selectedP1.jersey_number})`;

            // Player 2
            document.getElementById('p2-name').textContent = selectedP2.name;
            document.getElementById('p2-pos').textContent = selectedP2.position;
            document.getElementById('p2-team').textContent = selectedP2.team_name;
            document.getElementById('p2-jersey').textContent = `#${selectedP2.jersey_number}`;
            document.getElementById('p2-initials').textContent = getInitials(selectedP2.name);
            document.getElementById('p2-minutes').textContent = `${selectedP2.minutes_played || 90} mins`;
            document.getElementById('p2-rating').textContent = (selectedP2.rating || 6.0).toFixed(1);
            document.getElementById('legend-p2-label').textContent = `${selectedP2.name} (#${selectedP2.jersey_number})`;
        }

        function getInitials(name) {
            if (!name) return "P";
            const parts = name.trim().split(/\s+/);
            if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
            return name.slice(0, 2).toUpperCase();
        }

        function updateStatBars() {
            if (!selectedP1 || !selectedP2) return;

            const metrics = [
                { key: 'goals', label: 'Goals', max: 3, format: v => v },
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
                        <span class="stat-val val-p1" style="${val1 > val2 ? 'color: var(--pitch-emerald);' : ''}">${m.format(val1, selectedP1)}</span>
                        <span class="stat-name">${m.label}</span>
                        <span class="stat-val val-p2" style="${val2 > val1 ? 'color: var(--pitch-emerald);' : ''}">${m.format(val2, selectedP2)}</span>
                    </div>
                    <div class="stat-dual-progress">
                        <div class="prog-bar-p1" style="width: ${pct1}%;"></div>
                        <div class="prog-bar-p2" style="width: ${pct2}%;"></div>
                    </div>
                `;
                container.appendChild(row);
            });
        }

        // SVG RADAR CHART GENERATOR
        function renderPlayerRadar() {
            if (!selectedP1 || !selectedP2) return;

            const svg = document.getElementById('radar-svg');
            const cx = 200, cy = 200, r = 135;

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

            // Draw concentric background webs
            for (let level = 1; level <= 4; level++) {
                const curR = (r / 4) * level;
                let pts = [];
                for (let i = 0; i < n; i++) {
                    const angle = (Math.PI * 2 / n) * i - Math.PI / 2;
                    pts.push(`${cx + curR * Math.cos(angle)},${cy + curR * Math.sin(angle)}`);
                }
                svgHtml += `<polygon points="${pts.join(' ')}" fill="none" stroke="rgba(255,255,255,0.07)" stroke-width="1"/>`;
            }

            // Draw spoke axes and labels
            for (let i = 0; i < n; i++) {
                const angle = (Math.PI * 2 / n) * i - Math.PI / 2;
                const ax = cx + r * Math.cos(angle);
                const ay = cy + r * Math.sin(angle);
                svgHtml += `<line x1="${cx}" y1="${cy}" x2="${ax}" y2="${ay}" stroke="rgba(255,255,255,0.1)" stroke-width="1"/>`;

                // Label offset
                const lx = cx + (r + 26) * Math.cos(angle);
                const ly = cy + (r + 26) * Math.sin(angle);
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

            // Draw vertex dots
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

            document.getElementById('hm-p1-title').textContent = `${selectedP1.name} (#${selectedP1.jersey_number})`;
            document.getElementById('hm-p2-title').textContent = `${selectedP2.name} (#${selectedP2.jersey_number})`;

            // Load Heatmap 1
            const img1 = document.getElementById('p1-heatmap-img');
            const empty1 = document.getElementById('p1-heatmap-empty');
            if (selectedP1.lineup_id) {
                const url1 = `/matches/${MATCH_PUBLIC_ID}/player-heatmap/${selectedP1.lineup_id}/`;
                img1.src = url1;
                img1.style.display = 'block';
                empty1.style.display = 'none';
                img1.onerror = () => { img1.style.display = 'none'; empty1.style.display = 'flex'; };
            } else {
                img1.style.display = 'none';
                empty1.style.display = 'flex';
            }

            // Load Heatmap 2
            const img2 = document.getElementById('p2-heatmap-img');
            const empty2 = document.getElementById('p2-heatmap-empty');
            if (selectedP2.lineup_id) {
                const url2 = `/matches/${MATCH_PUBLIC_ID}/player-heatmap/${selectedP2.lineup_id}/`;
                img2.src = url2;
                img2.style.display = 'block';
                empty2.style.display = 'none';
                img2.onerror = () => { img2.style.display = 'none'; empty2.style.display = 'flex'; };
            } else {
                img2.style.display = 'none';
                empty2.style.display = 'flex';
            }
        }

        // TEAM RADAR GENERATOR
        function renderTeamRadar() {
            const svg = document.getElementById('team-radar-svg');
            if (!svg) return;

            const cx = 200, cy = 200, r = 135;
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

                const lx = cx + (r + 26) * Math.cos(angle);
                const ly = cy + (r + 26) * Math.sin(angle);
                const anchor = Math.abs(Math.cos(angle)) < 0.2 ? 'middle' : (Math.cos(angle) > 0 ? 'start' : 'end');
                svgHtml += `<text x="${lx}" y="${ly + 4}" fill="#94a3b8" font-size="11" font-weight="600" text-anchor="${anchor}">${dimensions[i].name}</text>`;
            }

            // Team 1
            let t1Pts = [];
            for (let i = 0; i < n; i++) {
                const angle = (Math.PI * 2 / n) * i - Math.PI / 2;
                const px = cx + (r * dimensions[i].val1) * Math.cos(angle);
                const py = cy + (r * dimensions[i].val1) * Math.sin(angle);
                t1Pts.push(`${px},${py}`);
            }
            svgHtml += `<polygon points="${t1Pts.join(' ')}" fill="${HOME_COLOR}" fill-opacity="0.25" stroke="${HOME_COLOR}" stroke-width="2.5"/>`;

            // Team 2
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

        // Tab Switcher
        function switchTab(tab) {
            const btnP = document.getElementById('btn-tab-players');
            const btnT = document.getElementById('btn-tab-teams');
            const paneP = document.getElementById('pane-players');
            const paneT = document.getElementById('pane-teams');

            if (tab === 'players') {
                btnP.classList.add('active');
                btnT.classList.remove('active');
                paneP.classList.add('active');
                paneT.classList.remove('active');
            } else {
                btnT.classList.add('active');
                btnP.classList.remove('active');
                paneT.classList.add('active');
                paneP.classList.remove('active');
                renderTeamRadar();
            }
        }

        document.addEventListener('DOMContentLoaded', () => {
            initSelectors();
            renderTeamRadar();
        });
