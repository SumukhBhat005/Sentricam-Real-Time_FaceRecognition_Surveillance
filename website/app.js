/* ═══════════════════════════════════════════════════════
   SentriCam — Dashboard Client
   MJPEG streaming, multi-photo enrollment, live HUD,
   real-time detection feed, activity logs, settings.
   ═══════════════════════════════════════════════════════ */

// ── State ────────────────────────────────────────────────
let detectTimer = null;
let statsTimer = null;
let clockTimer = null;
let enrollPreviewTimer = null;
let currentSection = 'live';
let isEnrolling = false;

// Enrollment state
const ENROLL_POSES = [
    { step: 1, label: "Look straight at the camera",       icon: "😐" },
    { step: 2, label: "Turn your head slightly LEFT",      icon: "👈" },
    { step: 3, label: "Turn your head slightly RIGHT",     icon: "👉" },
    { step: 4, label: "Tilt your head slightly UP",        icon: "👆" },
    { step: 5, label: "Make a natural expression / smile",  icon: "😄" },
];
let enrollStep = 0;
let enrollName = "";
let enrollCaptured = 0;


// ── Landing / Dashboard Toggle ──────────────────────────────

function enterDashboard() {
    document.getElementById('landing').classList.add('hidden');
    document.getElementById('dashboard').classList.remove('hidden');
    document.body.style.overflow = 'hidden'; // Prevent scrolling on dashboard
    startStream();
    startDetectionPolling();
    startStatsPolling();
    startClock();
    loadUsers();
    refreshLogs();
}

function goToHero() {
    stopStream();
    stopDetectionPolling();
    stopStatsPolling();
    stopClock();
    document.getElementById('dashboard').classList.add('hidden');
    document.getElementById('landing').classList.remove('hidden');
    document.body.style.overflow = 'auto'; // Restore scrolling for landing page
}


// ── Section Navigation ───────────────────────────────────

function switchSection(name) {
    currentSection = name;

    document.querySelectorAll('.nav-item').forEach(el => {
        el.classList.toggle('active', el.dataset.section === name);
    });

    document.querySelectorAll('.section').forEach(el => {
        el.classList.remove('active-section');
    });
    const target = document.getElementById('section-' + name);
    if (target) target.classList.add('active-section');

    if (name === 'users') loadUsers();
    if (name === 'logs') refreshLogs();
}


// ── MJPEG Stream ─────────────────────────────────────────

function startStream() {
    const feed = document.getElementById('camera-feed');
    const overlay = document.getElementById('feed-overlay');

    feed.src = '/api/stream';

    feed.onerror = function() {
        overlay.classList.remove('connected');
        setTimeout(() => {
            feed.src = '/api/stream?t=' + Date.now();
        }, 2000);
    };

    feed.onload = function() {
        overlay.classList.add('connected');
        const statusEl = document.getElementById('camera-status');
        statusEl.querySelector('span:last-child').textContent = 'Live';
        statusEl.classList.add('online');
    };
}

function stopStream() {
    const feed = document.getElementById('camera-feed');
    feed.src = '';
}


// ── Live Clock on HUD ────────────────────────────────────

function startClock() {
    stopClock();
    updateClock();
    clockTimer = setInterval(updateClock, 1000);
}

function stopClock() {
    if (clockTimer) { clearInterval(clockTimer); clockTimer = null; }
}

function updateClock() {
    const now = new Date();
    const h = String(now.getHours()).padStart(2, '0');
    const m = String(now.getMinutes()).padStart(2, '0');
    const s = String(now.getSeconds()).padStart(2, '0');
    const el = document.getElementById('hud-clock');
    if (el) el.textContent = `${h}:${m}:${s}`;
}


// ── Detection Polling ────────────────────────────────────

function startDetectionPolling() {
    stopDetectionPolling();
    fetchDetections();
    detectTimer = setInterval(fetchDetections, 1000);
}

function stopDetectionPolling() {
    if (detectTimer) { clearInterval(detectTimer); detectTimer = null; }
}

function fetchDetections() {
    fetch('/api/detections')
        .then(r => r.json())
        .then(data => {
            renderDetections(data.detections || []);
            toggleAlert(data.has_unknown);

            const overlay = document.getElementById('feed-overlay');
            const statusEl = document.getElementById('camera-status');
            if (data.camera_ok) {
                overlay.classList.add('connected');
                statusEl.querySelector('span:last-child').textContent = 'Live';
                statusEl.classList.add('online');
            } else {
                overlay.classList.remove('connected');
                statusEl.querySelector('span:last-child').textContent = 'No Camera';
                statusEl.classList.remove('online');
            }

            // Update detection count stat
            const detEl = document.getElementById('stat-detections');
            if (detEl) detEl.textContent = (data.detections || []).length;
        })
        .catch(() => {});
}

function renderDetections(detections) {
    const container = document.getElementById('hud-detections');
    if (!container) return;

    if (detections.length === 0) {
        container.innerHTML = '';
        return;
    }

    container.innerHTML = detections.map(d => {
        const isUnknown = d.name === 'Unknown';
        const isScanning = d.name === 'Scanning...';
        let cls = 'known';
        if (isUnknown) cls = 'unknown';
        else if (isScanning) cls = 'scanning';

        const label = isScanning ? 'SCANNING...' : d.name.toUpperCase();
        return `<div class="hud-det-badge ${cls}">${label}</div>`;
    }).join('');
}

function toggleAlert(show) {
    const banner = document.getElementById('alert-banner');
    if (show) {
        banner.classList.remove('hidden');
        const timeEl = document.getElementById('alert-time');
        if (timeEl) {
            const now = new Date();
            timeEl.textContent = `${String(now.getHours()).padStart(2,'0')}:${String(now.getMinutes()).padStart(2,'0')}`;
        }
    } else {
        banner.classList.add('hidden');
    }
}


// ── Stats Polling ────────────────────────────────────────

function startStatsPolling() {
    stopStatsPolling();
    fetchStats();
    statsTimer = setInterval(fetchStats, 5000);
}

function stopStatsPolling() {
    if (statsTimer) { clearInterval(statsTimer); statsTimer = null; }
}

function fetchStats() {
    fetch('/api/stats')
        .then(r => r.json())
        .then(data => {
            document.getElementById('stat-users').textContent = data.total_users || 0;

            const upSec = data.uptime_seconds || 0;
            const hrs = Math.floor(upSec / 3600);
            const mins = Math.floor((upSec % 3600) / 60);
            document.getElementById('stat-uptime').textContent =
                hrs > 0 ? `${hrs}h${mins}m` : `${mins}m`;
        })
        .catch(() => {});
}


// ── User Management ──────────────────────────────────────

function loadUsers() {
    fetch('/api/users')
        .then(r => r.json())
        .then(data => {
            const users = data.users || [];
            document.getElementById('user-count').textContent = users.length;
            const badgeEl = document.getElementById('user-count-badge');
            if (badgeEl) badgeEl.textContent = `${users.length} enrolled`;

            const list = document.getElementById('user-list');

            if (users.length === 0) {
                list.innerHTML = '<div class="empty-state">No users enrolled yet</div>';
                return;
            }

            list.innerHTML = users.map(u => `
                <div class="user-card">
                    <div class="user-info">
                        <div class="user-avatar">${u.name.charAt(0).toUpperCase()}</div>
                        <div>
                            <div class="user-name">${escapeHtml(u.name)}</div>
                            <div class="user-photos">${u.photo_count} photo${u.photo_count !== 1 ? 's' : ''} enrolled</div>
                        </div>
                    </div>
                    <button class="btn-delete" onclick="deleteUser('${escapeHtml(u.name)}')">Remove</button>
                </div>
            `).join('');
        })
        .catch(() => {});
}

function deleteUser(name) {
    if (!confirm(`Remove "${name}" and all their photos?`)) return;

    fetch(`/api/users/${encodeURIComponent(name)}`, { method: 'DELETE' })
        .then(r => r.json())
        .then(() => { loadUsers(); fetchStats(); })
        .catch(() => alert('Failed to delete user'));
}


// ── Multi-Photo Enrollment ───────────────────────────────

function startEnrollment() {
    const nameInput = document.getElementById('enroll-name');
    enrollName = nameInput.value.trim();

    if (!enrollName) {
        nameInput.style.borderColor = 'var(--red)';
        setTimeout(() => nameInput.style.borderColor = '', 1500);
        return;
    }

    isEnrolling = true;
    enrollStep = 0;
    enrollCaptured = 0;

    document.getElementById('enroll-progress').classList.remove('hidden');
    document.getElementById('enroll-start-btn').disabled = true;
    document.getElementById('enroll-name').disabled = true;

    updateEnrollUI();
    startEnrollPreview();
}

function cancelEnrollment() {
    isEnrolling = false;
    enrollStep = 0;
    enrollCaptured = 0;
    stopEnrollPreview();

    document.getElementById('enroll-progress').classList.add('hidden');
    document.getElementById('enroll-start-btn').disabled = false;
    document.getElementById('enroll-name').disabled = false;
    document.getElementById('enroll-status').textContent = '';
    document.getElementById('enroll-status').className = 'enroll-status';
    document.getElementById('enroll-thumbs').innerHTML = '';
}

function startEnrollPreview() {
    stopEnrollPreview();
    updateEnrollPreview();
    enrollPreviewTimer = setInterval(updateEnrollPreview, 300);
}

function stopEnrollPreview() {
    if (enrollPreviewTimer) { clearInterval(enrollPreviewTimer); enrollPreviewTimer = null; }
}

function updateEnrollPreview() {
    const img = document.getElementById('enroll-preview');
    const newImg = new Image();
    newImg.onload = function () { img.src = this.src; };
    newImg.src = '/api/snapshot?t=' + Date.now();
}

function updateEnrollUI() {
    const pose = ENROLL_POSES[enrollStep];
    document.getElementById('enroll-step-label').textContent = `Step ${pose.step} of 5`;
    document.getElementById('enroll-pose-label').textContent = pose.label;
    document.getElementById('progress-fill').style.width = ((enrollCaptured / 5) * 100) + '%';

    const thumbs = document.getElementById('enroll-thumbs');
    thumbs.innerHTML = ENROLL_POSES.map((p, i) => {
        if (i < enrollCaptured) {
            return `<div class="enroll-thumb">✅</div>`;
        } else {
            return `<div class="enroll-thumb pending">${p.icon}</div>`;
        }
    }).join('');
}

async function captureEnrollPhoto() {
    const btn = document.getElementById('capture-btn');
    const status = document.getElementById('enroll-status');

    btn.disabled = true;
    btn.textContent = 'Processing...';
    status.textContent = '';
    status.className = 'enroll-status';

    try {
        const resp = await fetch('/api/users/enroll', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: enrollName }),
        });

        const data = await resp.json();

        if (resp.ok && data.success) {
            enrollCaptured++;
            enrollStep++;

            if (enrollCaptured >= 5) {
                document.getElementById('progress-fill').style.width = '100%';
                status.textContent = `${enrollName} enrolled successfully with 5 photos!`;
                status.className = 'enroll-status success';
                stopEnrollPreview();
                btn.textContent = 'Capture Photo';
                btn.disabled = true;

                setTimeout(() => {
                    cancelEnrollment();
                    document.getElementById('enroll-name').value = '';
                    loadUsers();
                    fetchStats();
                }, 2500);
                return;
            }

            updateEnrollUI();
            status.textContent = `Photo ${enrollCaptured} captured! Now: ${ENROLL_POSES[enrollStep].label}`;
            status.className = 'enroll-status success';
        } else {
            status.textContent = data.error || 'Failed to capture. Try again.';
            status.className = 'enroll-status error';
        }
    } catch (err) {
        status.textContent = 'Network error. Is the server running?';
        status.className = 'enroll-status error';
    }

    btn.disabled = false;
    btn.textContent = 'Capture Photo';
}


// ── Activity Logs ────────────────────────────────────────

function refreshLogs() {
    fetch('/api/logs?limit=50')
        .then(r => r.json())
        .then(data => {
            const logs = data.logs || [];
            const body = document.getElementById('log-body');

            if (logs.length === 0) {
                body.innerHTML = '<tr><td colspan="4" class="empty-state">No activity yet</td></tr>';
                return;
            }

            body.innerHTML = logs.map(log => {
                const isUnknown = log.name === 'Unknown';
                return `
                    <tr>
                        <td>${escapeHtml(log.timestamp)}</td>
                        <td><strong>${escapeHtml(log.name)}</strong></td>
                        <td>${log.confidence}</td>
                        <td><span class="log-status ${isUnknown ? 'unknown' : 'known'}">${isUnknown ? 'UNKNOWN' : 'KNOWN'}</span></td>
                    </tr>
                `;
            }).join('');
        })
        .catch(() => {});
}


// ── Settings ─────────────────────────────────────────────

function loadSettings() {
    fetch('/api/settings')
        .then(r => r.json())
        .then(data => {
            document.getElementById('set-threshold').value = data.threshold || 0.40;
            document.getElementById('threshold-val').textContent = data.threshold || 0.40;
            document.getElementById('set-quality').value = data.snapshot_quality || 65;
            document.getElementById('quality-val').textContent = data.snapshot_quality || 65;
        })
        .catch(() => {});
}

function saveSettings() {
    const payload = {
        threshold: parseFloat(document.getElementById('set-threshold').value),
        snapshot_quality: parseInt(document.getElementById('set-quality').value),
    };

    fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    })
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                const status = document.getElementById('settings-status');
                status.textContent = 'Settings saved';
                status.style.color = 'var(--green)';
                setTimeout(() => { status.textContent = ''; }, 2000);
            }
        })
        .catch(() => {
            const status = document.getElementById('settings-status');
            status.textContent = 'Failed to save';
            status.style.color = 'var(--red)';
        });
}

// Range input live value display
document.addEventListener('DOMContentLoaded', () => {
    const ranges = [
        { id: 'set-threshold', display: 'threshold-val' },
        { id: 'set-quality', display: 'quality-val' },
    ];

    ranges.forEach(r => {
        const el = document.getElementById(r.id);
        if (el) {
            el.addEventListener('input', () => {
                document.getElementById(r.display).textContent = el.value;
            });
        }
    });

    loadSettings();
});


// ── Utilities ────────────────────────────────────────────

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
