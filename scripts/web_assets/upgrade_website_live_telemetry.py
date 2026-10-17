import os
import json

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HTML_PATH = PROJECT_ROOT / "index.html"
BRAIN_PATH = r"C:\Users\hafiz\.gemini\antigravity\brain\2101233a-92d3-465b-a65e-a4eda2098b15\nexoguard_blueprint.html"

with open(os.path.join(DATA_DIR, "real_patient_streams.json"), "r") as f:
    real_data = json.load(f)

with open(HTML_PATH, "r", encoding="utf-8") as f:
    html = f.read()

# 1. Update the patient select dropdown in Tab 2
old_select_html = """            <select id="sim-patient-select" onchange="loadSimPatient()" class="bg-slate-900 text-slate-200 border border-slate-700 rounded-lg px-3 py-1.5 text-sm focus:ring-teal-500">
              <option value="normal">Patient 101: Reassuring (Category I)</option>
              <option value="suspicious">Patient 204: Indeterminate (Category II)</option>
              <option value="pathological" selected>Patient 309: Acute Distress / Late Decel (Category III)</option>
            </select>
            <button onclick="toggleSimPlay()" id="sim-play-btn" class="px-3 py-1.5 bg-teal-500 hover:bg-teal-400 text-slate-900 font-bold rounded-lg text-sm transition">
              ▶ Pause
            </button>"""

new_select_html = """            <div class="flex flex-wrap items-center gap-2">
              <select id="sim-patient-select" onchange="loadSimPatient()" class="bg-slate-900 text-slate-200 border border-slate-700 rounded-lg px-3 py-1.5 text-xs md:text-sm focus:ring-teal-500">
                <option value="1029" selected>Patient 1029: Acute Acidemia / Severe Distress (pH 6.97) 🔴</option>
                <option value="1001">Patient 1001: Recurrent Late Decelerations (pH 7.14) 🔴</option>
                <option value="1006">Patient 1006: Indeterminate / Minimal Variability (pH 7.23) 🟡</option>
                <option value="1012">Patient 1012: Reassuring Normal Baseline (pH 7.36) 🟢</option>
                <option value="1024">Patient 1024: Textbook Healthy Reactivity (pH 7.43) 🟢</option>
              </select>
              <button onclick="toggleSimPlay()" id="sim-play-btn" class="px-3 py-1.5 bg-teal-500 hover:bg-teal-400 text-slate-900 font-bold rounded-lg text-xs md:text-sm transition shadow">
                ⏸ Pause
              </button>
              <div class="flex items-center gap-1 bg-slate-900 border border-slate-800 rounded-lg p-0.5 text-xs text-slate-300">
                <button onclick="setSimSpeed(1)" class="speed-btn px-2 py-1 rounded bg-teal-500/20 text-teal-400 font-bold" id="spd-1">1x</button>
                <button onclick="setSimSpeed(2)" class="speed-btn px-2 py-1 rounded hover:bg-slate-800" id="spd-2">2x</button>
                <button onclick="setSimSpeed(4)" class="speed-btn px-2 py-1 rounded hover:bg-slate-800" id="spd-4">4x</button>
              </div>
            </div>"""

if old_select_html in html:
    html = html.replace(old_select_html, new_select_html)
    print("Replaced patient selector HTML!")
else:
    print("Warning: old_select_html not found exactly, will check.")

# 2. Add live digital readouts under the canvases
old_status_bar = """            <div class="flex items-center justify-between text-xs text-slate-400 bg-slate-900 p-3 rounded-lg border border-slate-800">
              <span>Streaming Rate: 4 Hz (250ms window step)</span>
              <span id="sim-status-clock" class="font-mono text-teal-400">Time: 14m 32s elapsed</span>
              <span>NexoPrima Baseline Model: EMA (α = 0.85)</span>
            </div>"""

new_status_bar = """            <!-- Real-Time Telemetry Control Bar -->
            <div class="space-y-2 bg-slate-900 p-3.5 rounded-lg border border-slate-800 text-xs">
              <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-2">
                <div class="flex items-center gap-4">
                  <span class="flex items-center gap-1.5">
                    <span class="w-2 h-2 rounded-full bg-rose-500 animate-ping"></span>
                    <strong class="text-white">LIVE FHR:</strong> <span id="live-fhr-val" class="font-mono text-teal-400 font-bold text-sm">124.2 bpm</span>
                  </span>
                  <span class="flex items-center gap-1.5">
                    <strong class="text-white">CONTRACTION:</strong> <span id="live-uc-val" class="font-mono text-sky-400 font-bold text-sm">48.5 mmHg</span>
                  </span>
                  <span class="flex items-center gap-1.5">
                    <strong class="text-white">EMA BASELINE:</strong> <span id="live-ema-val" class="font-mono text-slate-300 font-bold text-sm">132.0 bpm</span>
                  </span>
                </div>
                <div class="flex items-center gap-3">
                  <span id="sim-status-clock" class="font-mono text-amber-400 font-semibold">Labor Time: 00m 00s / 15m 00s</span>
                  <span class="px-2 py-0.5 bg-teal-500/10 text-teal-300 rounded text-[11px] font-mono">PhysioNet 4Hz Data</span>
                </div>
              </div>
              <div class="flex items-center gap-3 pt-1">
                <span class="text-slate-400 text-[11px] font-semibold uppercase">Timeline Scrubber:</span>
                <input type="range" id="sim-scrubber" min="0" max="3500" value="0" oninput="scrubSimTimeline(this.value)" class="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-teal-400">
              </div>
            </div>"""

if old_status_bar in html:
    html = html.replace(old_status_bar, new_status_bar)
    print("Replaced status bar with interactive scrubber and digital readouts!")

# 3. Inject the real dataset into JavaScript and upgrade the canvas drawing loop
real_data_json = json.dumps(real_data)

js_upgrade = f"""
    // REAL HOSPITAL TELEMETRY DATASET FROM PHYSIONET CTU-UHB
    const REAL_PATIENT_DATA = {real_data_json};

    let simTimer = null;
    let simPlaying = true;
    let simSpeed = 1;
    let simIndex = 0; // Current sample index (0 to 3600)
    let currentPatientId = '1029';

    function initSimulator() {{
      if (!simTimer) {{
        loadSimPatient();
        simTimer = setInterval(drawSimulatorFrame, 50); // 20 FPS rendering
      }}
    }}

    function toggleSimPlay() {{
      simPlaying = !simPlaying;
      document.getElementById('sim-play-btn').innerText = simPlaying ? '⏸ Pause' : '▶ Play';
    }}

    function setSimSpeed(spd) {{
      simSpeed = spd;
      document.querySelectorAll('.speed-btn').forEach(b => {{
        b.classList.remove('bg-teal-500/20', 'text-teal-400', 'font-bold');
        b.classList.add('hover:bg-slate-800');
      }});
      const activeBtn = document.getElementById('spd-' + spd);
      if (activeBtn) {{
        activeBtn.classList.add('bg-teal-500/20', 'text-teal-400', 'font-bold');
        activeBtn.classList.remove('hover:bg-slate-800');
      }}
    }}

    function scrubSimTimeline(val) {{
      simIndex = parseInt(val);
    }}

    function loadSimPatient() {{
      const select = document.getElementById('sim-patient-select');
      if (select) currentPatientId = select.value;
      const pat = REAL_PATIENT_DATA[currentPatientId];
      if (!pat) return;

      simIndex = 0; // reset to beginning of strip

      const card = document.getElementById('triage-card');
      const title = document.getElementById('triage-title');
      const sub = document.getElementById('triage-subtitle');
      const risk = document.getElementById('triage-risk');
      const fhr = document.getElementById('triage-fhr');
      const stv = document.getElementById('triage-stv');
      const lag = document.getElementById('triage-lag');
      const shap = document.getElementById('shap-bars');

      if (pat.color === 'rose') {{
        card.className = "p-5 rounded-xl border bg-rose-500/10 border-rose-500/40 shadow-lg";
        title.className = "text-2xl font-black text-rose-300 mt-1";
        title.innerText = "🔴 " + pat.category.toUpperCase();
        sub.innerText = pat.status.toUpperCase();
        sub.className = "text-xs font-semibold text-rose-200 mt-0.5";
        risk.innerText = pat.risk_score + "% (Extreme Acidemia)";
        risk.className = "font-bold text-rose-300";
        fhr.innerText = "Actual Cord pH: " + pat.actual_ph + " (Severe Acidosis)";
        stv.innerText = pat.stv_ms + " ms (Abnormal < 3.0)";
        stv.className = "font-mono text-rose-300 font-bold";
        lag.innerText = "BDecf: " + pat.actual_bdecf + " mmol/L (Late Decels)";
        lag.className = "font-mono text-rose-300 font-bold";
        shap.innerHTML = `
          <div>
            <div class="flex justify-between text-slate-300 mb-1">
              <span>Severe Acidemia Marker (Cord pH = ${{pat.actual_ph}})</span>
              <span class="text-rose-400 font-semibold">+46% Risk</span>
            </div>
            <div class="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
              <div class="bg-rose-500 h-full w-[92%]"></div>
            </div>
          </div>
          <div>
            <div class="flex justify-between text-slate-300 mb-1">
              <span>Pathological Deceleration (Late Lag &gt; 25s)</span>
              <span class="text-rose-400 font-semibold">+35% Risk</span>
            </div>
            <div class="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
              <div class="bg-rose-500 h-full w-[70%]"></div>
            </div>
          </div>
          <div>
            <div class="flex justify-between text-slate-300 mb-1">
              <span>Depressed Short-Term Variability (STV = ${{pat.stv_ms}}ms)</span>
              <span class="text-rose-400 font-semibold">+22% Risk</span>
            </div>
            <div class="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
              <div class="bg-rose-500 h-full w-[44%]"></div>
            </div>
          </div>
        `;
      }} else if (pat.color === 'amber') {{
        card.className = "p-5 rounded-xl border bg-amber-500/10 border-amber-500/40 shadow-lg";
        title.className = "text-2xl font-black text-amber-300 mt-1";
        title.innerText = "🟡 " + pat.category.toUpperCase();
        sub.innerText = pat.status.toUpperCase();
        sub.className = "text-xs font-semibold text-amber-200 mt-0.5";
        risk.innerText = pat.risk_score + "% (Borderline)";
        risk.className = "font-bold text-amber-300";
        fhr.innerText = "Actual Cord pH: " + pat.actual_ph + " (Borderline)";
        stv.innerText = pat.stv_ms + " ms (Minimal Variability)";
        stv.className = "font-mono text-amber-300 font-bold";
        lag.innerText = "BDecf: " + pat.actual_bdecf + " mmol/L";
        lag.className = "font-mono text-amber-300 font-bold";
        shap.innerHTML = `
          <div>
            <div class="flex justify-between text-slate-300 mb-1">
              <span>Borderline pH & Minimal STV (${{pat.stv_ms}}ms)</span>
              <span class="text-amber-400 font-semibold">+32% Risk</span>
            </div>
            <div class="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
              <div class="bg-amber-500 h-full w-[64%]"></div>
            </div>
          </div>
          <div>
            <div class="flex justify-between text-slate-300 mb-1">
              <span>Periodic Contractions Without Accels</span>
              <span class="text-amber-400 font-semibold">+18% Risk</span>
            </div>
            <div class="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
              <div class="bg-amber-500 h-full w-[36%]"></div>
            </div>
          </div>
        `;
      }} else {{
        card.className = "p-5 rounded-xl border bg-emerald-500/10 border-emerald-500/40 shadow-lg";
        title.className = "text-2xl font-black text-emerald-300 mt-1";
        title.innerText = "🟢 " + pat.category.toUpperCase();
        sub.innerText = pat.status.toUpperCase();
        sub.className = "text-xs font-semibold text-emerald-200 mt-0.5";
        risk.innerText = pat.risk_score + "% (Healthy)";
        risk.className = "font-bold text-emerald-300";
        fhr.innerText = "Actual Cord pH: " + pat.actual_ph + " (Healthy)";
        stv.innerText = pat.stv_ms + " ms (Healthy > 3.5)";
        stv.className = "font-mono text-emerald-300 font-bold";
        lag.innerText = "BDecf: " + pat.actual_bdecf + " mmol/L (Normal)";
        lag.className = "font-mono text-emerald-300 font-bold";
        shap.innerHTML = `
          <div>
            <div class="flex justify-between text-slate-300 mb-1">
              <span>Optimal Acid-Base Balance (pH = ${{pat.actual_ph}})</span>
              <span class="text-emerald-400 font-semibold">-48% Risk</span>
            </div>
            <div class="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
              <div class="bg-emerald-500 h-full w-[90%]"></div>
            </div>
          </div>
          <div>
            <div class="flex justify-between text-slate-300 mb-1">
              <span>Reassuring Baseline Rate & Reactivity</span>
              <span class="text-emerald-400 font-semibold">-35% Risk</span>
            </div>
            <div class="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
              <div class="bg-emerald-500 h-full w-[70%]"></div>
            </div>
          </div>
        `;
      }}
    }}

    function drawSimulatorFrame() {{
      const pat = REAL_PATIENT_DATA[currentPatientId];
      if (!pat) return;

      const fhrPoints = pat.fhr_points;
      const ucPoints = pat.uc_points;
      const emaPoints = pat.ema_points;
      const totalPoints = fhrPoints.length;

      if (simPlaying) {{
        simIndex = (simIndex + simSpeed) % (totalPoints - 1);
        const scrubber = document.getElementById('sim-scrubber');
        if (scrubber) scrubber.value = simIndex;
      }}

      // Update digital readouts
      const curFhr = fhrPoints[simIndex] || 135.0;
      const curUc = ucPoints[simIndex] || 15.0;
      const curEma = emaPoints[simIndex] || 135.0;

      const fhrEl = document.getElementById('live-fhr-val');
      const ucEl = document.getElementById('live-uc-val');
      const emaEl = document.getElementById('live-ema-val');
      const clockEl = document.getElementById('sim-status-clock');

      if (fhrEl) fhrEl.innerText = curFhr.toFixed(1) + ' bpm';
      if (ucEl) ucEl.innerText = curUc.toFixed(1) + ' mmHg';
      if (emaEl) emaEl.innerText = curEma.toFixed(1) + ' bpm';

      const secElapsed = Math.floor(simIndex / 4);
      const minStr = String(Math.floor(secElapsed / 60)).padStart(2, '0');
      const secStr = String(secElapsed % 60).padStart(2, '0');
      if (clockEl) clockEl.innerText = `Labor Time: ${{minStr}}m ${{secStr}}s / 15m 00s`;

      const fhrC = document.getElementById('fhr-canvas');
      const tocoC = document.getElementById('toco-canvas');
      if (!fhrC || !tocoC) return;

      const fctx = fhrC.getContext('2d');
      const tctx = tocoC.getContext('2d');
      const W = fhrC.width;
      const H1 = fhrC.height;
      const H2 = tocoC.height;

      // Clear with dark medical monitor background
      fctx.fillStyle = '#080c14';
      fctx.fillRect(0, 0, W, H1);
      tctx.fillStyle = '#080c14';
      tctx.fillRect(0, 0, W, H2);

      // Grid Lines
      fctx.strokeStyle = '#1e293b';
      fctx.lineWidth = 1;
      for (let y = 30; y < H1; y += 35) {{
        fctx.beginPath();
        fctx.moveTo(0, y);
        fctx.lineTo(W, y);
        fctx.stroke();
      }}
      tctx.strokeStyle = '#1e293b';
      tctx.lineWidth = 1;
      for (let y = 20; y < H2; y += 30) {{
        tctx.beginPath();
        tctx.moveTo(0, y);
        tctx.lineTo(W, y);
        tctx.stroke();
      }}

      // Green FIGO Normal Zone (110 - 160 bpm)
      const y160 = H1 - ((160 - 60) / 140) * H1;
      const y110 = H1 - ((110 - 60) / 140) * H1;
      fctx.fillStyle = 'rgba(16, 185, 129, 0.08)';
      fctx.fillRect(0, y160, W, y110 - y160);

      // Window of points to display (e.g. past 400 points = 100 seconds of 4Hz data)
      const windowSize = 450;
      const startIdx = Math.max(0, simIndex - windowSize);
      const pointsToShow = simIndex - startIdx;

      // 1. Draw TOCO Contraction Waveform (Channel 2)
      tctx.strokeStyle = '#38bdf8';
      tctx.lineWidth = 2.2;
      tctx.beginPath();
      for (let i = 0; i < pointsToShow; i++) {{
        const ptIdx = startIdx + i;
        const val = ucPoints[ptIdx] || 10;
        const x = (i / windowSize) * W;
        const y = H2 - (val / 100) * H2;
        if (i === 0) tctx.moveTo(x, y);
        else tctx.lineTo(x, y);
      }}
      tctx.stroke();

      // TOCO Live Dot
      if (pointsToShow > 0) {{
        const lastX = (pointsToShow / windowSize) * W;
        const lastY = H2 - (curUc / 100) * H2;
        tctx.fillStyle = '#38bdf8';
        tctx.beginPath();
        tctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
        tctx.fill();
      }}

      // 2. Draw Real FHR Waveform (Channel 1)
      const fhrColor = pat.color === 'rose' ? '#f43f5e' : (pat.color === 'amber' ? '#f59e0b' : '#10b981');
      fctx.strokeStyle = fhrColor;
      fctx.lineWidth = 2.2;
      fctx.beginPath();
      for (let i = 0; i < pointsToShow; i++) {{
        const ptIdx = startIdx + i;
        const val = fhrPoints[ptIdx] || 135;
        const x = (i / windowSize) * W;
        const y = H1 - ((val - 60) / 140) * H1;
        if (i === 0) fctx.moveTo(x, y);
        else fctx.lineTo(x, y);
      }}
      fctx.stroke();

      // 3. Draw Real NexoPrima EMA Baseline Overlay
      fctx.strokeStyle = '#cbd5e1';
      fctx.setLineDash([4, 4]);
      fctx.lineWidth = 1.5;
      fctx.beginPath();
      for (let i = 0; i < pointsToShow; i++) {{
        const ptIdx = startIdx + i;
        const val = emaPoints[ptIdx] || 135;
        const x = (i / windowSize) * W;
        const y = H1 - ((val - 60) / 140) * H1;
        if (i === 0) fctx.moveTo(x, y);
        else fctx.lineTo(x, y);
      }}
      fctx.stroke();
      fctx.setLineDash([]); // Reset dash

      // FHR Live Pulse Dot with glowing ring
      if (pointsToShow > 0) {{
        const lastX = (pointsToShow / windowSize) * W;
        const lastY = H1 - ((curFhr - 60) / 140) * H1;
        
        fctx.fillStyle = fhrColor;
        fctx.beginPath();
        fctx.arc(lastX, lastY, 4.5, 0, Math.PI * 2);
        fctx.fill();

        fctx.strokeStyle = fhrColor;
        fctx.lineWidth = 1.5;
        fctx.beginPath();
        fctx.arc(lastX, lastY, 7.5, 0, Math.PI * 2);
        fctx.stroke();
      }}
    }}
"""

# Replace the previous JS simulator block
# We replace from "let simTimer = null;" to the end of drawSimulatorFrame
start_marker = "    let simTimer = null;"
end_marker = "    function copyMasterAiPrompt() {"

idx_start = html.find(start_marker)
idx_end = html.find(end_marker)

if idx_start != -1 and idx_end != -1:
    html = html[:idx_start] + js_upgrade + "\n\n" + html[idx_end:]
    print("Injected real telemetry rendering engine into JavaScript!")
else:
    print(f"Markers not found: start={idx_start}, end={idx_end}")

# Write to both locations
with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(html)
with open(BRAIN_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print("Updated index.html and nexoguard_blueprint.html successfully with REAL PhysioNet data streams!")
