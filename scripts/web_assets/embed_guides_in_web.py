import json
import os
import re

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
html_path = PROJECT_ROOT / "index.html"
guides_dir = PROJECT_ROOT / "guides"

files = {
    "master": "PROJECT_MASTER_CONTEXT_FOR_AI.md",
    "p1": "ROLE_1_SIGNAL_ENGINEER.md",
    "p2": "ROLE_2_ML_ENGINEER.md",
    "p3": "ROLE_3_APP_DEVELOPER.md",
    "p4": "ROLE_4_PITCH_PRODUCT_LEAD.md"
}

guide_contents = {}
for key, filename in files.items():
    filepath = os.path.join(guides_dir, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        guide_contents[key] = f.read()

with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()

# Replace the 4 cards in "Individual Role Instruction Guides"
old_cards_block = """            <!-- Guide 1 -->
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
              <div>
                <div class="flex items-center justify-between mb-2">
                  <span class="text-teal-400 font-bold uppercase">Person 1: Signal Lead</span>
                  <span class="font-mono text-slate-500">ROLE_1_SIGNAL_ENGINEER.md</span>
                </div>
                <h4 class="text-sm font-bold text-white">Signal Cleaning &amp; Feature Extractor</h4>
                <p class="text-slate-400 mt-1">Contains full Python script `signal_pipeline.py` with median filter, Savitzky-Golay, NexoPrima EMA baseline, STV, and contraction-deceleration lag calculations.</p>
              </div>
              <div class="mt-4 pt-3 border-t border-slate-800 flex justify-between items-center text-slate-400">
                <span>📁 `guides/ROLE_1_SIGNAL_ENGINEER.md`</span>
                <span class="text-teal-400 font-medium">Ready</span>
              </div>
            </div>

            <!-- Guide 2 -->
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
              <div>
                <div class="flex items-center justify-between mb-2">
                  <span class="text-sky-400 font-bold uppercase">Person 2: ML Lead</span>
                  <span class="font-mono text-slate-500">ROLE_2_ML_ENGINEER.md</span>
                </div>
                <h4 class="text-sm font-bold text-white">XGBoost &amp; SHAP Explainability Engine</h4>
                <p class="text-slate-400 mt-1">Contains full Python script `model_pipeline.py` with cost-sensitive loss, Stratified K-Fold, FIGO deterministic safety override, and SHAP TreeExplainer attributions.</p>
              </div>
              <div class="mt-4 pt-3 border-t border-slate-800 flex justify-between items-center text-slate-400">
                <span>📁 `guides/ROLE_2_ML_ENGINEER.md`</span>
                <span class="text-sky-400 font-medium">Ready</span>
              </div>
            </div>

            <!-- Guide 3 -->
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
              <div>
                <div class="flex items-center justify-between mb-2">
                  <span class="text-purple-400 font-bold uppercase">Person 3: App Lead</span>
                  <span class="font-mono text-slate-500">ROLE_3_APP_DEVELOPER.md</span>
                </div>
                <h4 class="text-sm font-bold text-white">Streamlit Telemetry Dashboard</h4>
                <p class="text-slate-400 mt-1">Contains complete `app.py` script with dual-channel Plotly streaming strip, FIGO traffic-light status badges, and SHAP clinical justification cards.</p>
              </div>
              <div class="mt-4 pt-3 border-t border-slate-800 flex justify-between items-center text-slate-400">
                <span>📁 `guides/ROLE_3_APP_DEVELOPER.md`</span>
                <span class="text-purple-400 font-medium">Ready</span>
              </div>
            </div>

            <!-- Guide 4 -->
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
              <div>
                <div class="flex items-center justify-between mb-2">
                  <span class="text-amber-400 font-bold uppercase">Person 4: Pitch Lead</span>
                  <span class="font-mono text-slate-500">ROLE_4_PITCH_PRODUCT_LEAD.md</span>
                </div>
                <h4 class="text-sm font-bold text-white">10-Slide Deck &amp; 3-Minute Script</h4>
                <p class="text-slate-400 mt-1">Contains the 10-slide outline, word-for-word timed 3-minute pitch script, NexoPrima business positioning, and judge defense playbook.</p>
              </div>
              <div class="mt-4 pt-3 border-t border-slate-800 flex justify-between items-center text-slate-400">
                <span>📁 `guides/ROLE_4_PITCH_PRODUCT_LEAD.md`</span>
                <span class="text-amber-400 font-medium">Ready</span>
              </div>
            </div>"""

new_cards_block = """            <!-- Guide 1 -->
            <div class="bg-slate-900 border border-teal-500/30 rounded-xl p-5 flex flex-col justify-between shadow-md">
              <div>
                <div class="flex items-center justify-between mb-2">
                  <span class="text-teal-400 font-bold uppercase tracking-wider">Role 1: Signal Lead</span>
                  <span class="font-mono text-slate-400 bg-slate-950 px-2 py-0.5 rounded text-[11px]">ROLE_1_SIGNAL_ENGINEER.md</span>
                </div>
                <h4 class="text-base font-bold text-white">Signal Cleaning &amp; Feature Extractor</h4>
                <p class="text-slate-300 mt-1.5 leading-relaxed">
                  Full Python pipeline (<code class="text-teal-300">signal_pipeline.py</code>) with median filter, Savitzky-Golay, NexoPrima EMA baseline, STV, and contraction-deceleration lag.
                </p>
              </div>
              <div class="mt-5 pt-4 border-t border-slate-800 flex flex-wrap gap-2 items-center justify-between">
                <button onclick="openGuideModal('p1')" class="px-3.5 py-1.5 bg-teal-500 hover:bg-teal-400 text-slate-900 font-bold rounded-lg text-xs transition flex items-center gap-1.5 shadow">
                  📖 Read Full Guide &amp; Code
                </button>
                <div class="flex gap-1.5">
                  <button onclick="copyGuideText('p1')" class="px-2.5 py-1.5 bg-slate-800 hover:bg-slate-700 text-teal-300 rounded-lg text-xs border border-teal-500/30 transition">
                    📋 Copy
                  </button>
                  <a href="guides/ROLE_1_SIGNAL_ENGINEER.md" download class="px-2.5 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs border border-slate-700 transition">
                    ⬇️ .md
                  </a>
                </div>
              </div>
            </div>

            <!-- Guide 2 -->
            <div class="bg-slate-900 border border-sky-500/30 rounded-xl p-5 flex flex-col justify-between shadow-md">
              <div>
                <div class="flex items-center justify-between mb-2">
                  <span class="text-sky-400 font-bold uppercase tracking-wider">Role 2: ML Lead</span>
                  <span class="font-mono text-slate-400 bg-slate-950 px-2 py-0.5 rounded text-[11px]">ROLE_2_ML_ENGINEER.md</span>
                </div>
                <h4 class="text-base font-bold text-white">XGBoost &amp; SHAP Explainability Engine</h4>
                <p class="text-slate-300 mt-1.5 leading-relaxed">
                  Full Python pipeline (<code class="text-sky-300">model_pipeline.py</code>) with cost-sensitive loss (eliminates missed decels), Stratified K-Fold, FIGO safety guardrail, and SHAP.
                </p>
              </div>
              <div class="mt-5 pt-4 border-t border-slate-800 flex flex-wrap gap-2 items-center justify-between">
                <button onclick="openGuideModal('p2')" class="px-3.5 py-1.5 bg-sky-500 hover:bg-sky-400 text-slate-900 font-bold rounded-lg text-xs transition flex items-center gap-1.5 shadow">
                  📖 Read Full Guide &amp; Code
                </button>
                <div class="flex gap-1.5">
                  <button onclick="copyGuideText('p2')" class="px-2.5 py-1.5 bg-slate-800 hover:bg-slate-700 text-sky-300 rounded-lg text-xs border border-sky-500/30 transition">
                    📋 Copy
                  </button>
                  <a href="guides/ROLE_2_ML_ENGINEER.md" download class="px-2.5 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs border border-slate-700 transition">
                    ⬇️ .md
                  </a>
                </div>
              </div>
            </div>

            <!-- Guide 3 -->
            <div class="bg-slate-900 border border-purple-500/30 rounded-xl p-5 flex flex-col justify-between shadow-md">
              <div>
                <div class="flex items-center justify-between mb-2">
                  <span class="text-purple-400 font-bold uppercase tracking-wider">Role 3: App Lead</span>
                  <span class="font-mono text-slate-400 bg-slate-950 px-2 py-0.5 rounded text-[11px]">ROLE_3_APP_DEVELOPER.md</span>
                </div>
                <h4 class="text-base font-bold text-white">Streamlit Telemetry Dashboard</h4>
                <p class="text-slate-300 mt-1.5 leading-relaxed">
                  Full Streamlit script (<code class="text-purple-300">app.py</code>) with dual-channel Plotly streaming strip, FIGO traffic-light status badges, and SHAP attribution cards.
                </p>
              </div>
              <div class="mt-5 pt-4 border-t border-slate-800 flex flex-wrap gap-2 items-center justify-between">
                <button onclick="openGuideModal('p3')" class="px-3.5 py-1.5 bg-purple-500 hover:bg-purple-400 text-white font-bold rounded-lg text-xs transition flex items-center gap-1.5 shadow">
                  📖 Read Full Guide &amp; Code
                </button>
                <div class="flex gap-1.5">
                  <button onclick="copyGuideText('p3')" class="px-2.5 py-1.5 bg-slate-800 hover:bg-slate-700 text-purple-300 rounded-lg text-xs border border-purple-500/30 transition">
                    📋 Copy
                  </button>
                  <a href="guides/ROLE_3_APP_DEVELOPER.md" download class="px-2.5 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs border border-slate-700 transition">
                    ⬇️ .md
                  </a>
                </div>
              </div>
            </div>

            <!-- Guide 4 -->
            <div class="bg-slate-900 border border-amber-500/30 rounded-xl p-5 flex flex-col justify-between shadow-md">
              <div>
                <div class="flex items-center justify-between mb-2">
                  <span class="text-amber-400 font-bold uppercase tracking-wider">Role 4: Pitch Lead</span>
                  <span class="font-mono text-slate-400 bg-slate-950 px-2 py-0.5 rounded text-[11px]">ROLE_4_PITCH_PRODUCT_LEAD.md</span>
                </div>
                <h4 class="text-base font-bold text-white">10-Slide Deck &amp; 3-Minute Script</h4>
                <p class="text-slate-300 mt-1.5 leading-relaxed">
                  Full 10-slide outline, word-for-word timed 3-minute pitch script, NexoPrima business strategy, and the judge defense playbook for tough questions.
                </p>
              </div>
              <div class="mt-5 pt-4 border-t border-slate-800 flex flex-wrap gap-2 items-center justify-between">
                <button onclick="openGuideModal('p4')" class="px-3.5 py-1.5 bg-amber-500 hover:bg-amber-400 text-slate-900 font-bold rounded-lg text-xs transition flex items-center gap-1.5 shadow">
                  📖 Read Full Guide &amp; Code
                </button>
                <div class="flex gap-1.5">
                  <button onclick="copyGuideText('p4')" class="px-2.5 py-1.5 bg-slate-800 hover:bg-slate-700 text-amber-300 rounded-lg text-xs border border-amber-500/30 transition">
                    📋 Copy
                  </button>
                  <a href="guides/ROLE_4_PITCH_PRODUCT_LEAD.md" download class="px-2.5 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs border border-slate-700 transition">
                    ⬇️ .md
                  </a>
                </div>
              </div>
            </div>"""

if old_cards_block in html:
    html = html.replace(old_cards_block, new_cards_block)
    print("Replaced old card block successfully!")
else:
    print("Old card block not found exactly, will use regex or direct replacement.")

# Now add the guide modal HTML right before </body>
modal_html = """
  <!-- ================= IN-BROWSER GUIDE READER MODAL ================= -->
  <div id="guide-modal" class="fixed inset-0 z-50 hidden bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-3 md:p-6 no-print">
    <div class="bg-slate-900 border border-slate-700 w-full max-w-5xl h-[90vh] rounded-2xl shadow-2xl flex flex-col overflow-hidden animate-fade-in">
      <!-- Modal Header -->
      <div class="p-4 md:p-5 border-b border-slate-800 flex items-center justify-between bg-slate-950">
        <div class="flex items-center gap-3">
          <span id="modal-role-badge" class="px-2.5 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-teal-500/20 text-teal-400">ROLE</span>
          <h3 id="modal-title" class="text-base md:text-lg font-bold text-white">Guide Title</h3>
        </div>
        <div class="flex items-center gap-2">
          <button onclick="copyCurrentModalText()" id="modal-copy-btn" class="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-teal-400 border border-teal-500/30 rounded-lg text-xs font-semibold transition flex items-center gap-1.5">
            📋 Copy Entire Guide
          </button>
          <a id="modal-download-link" href="#" download class="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-lg text-xs font-semibold transition flex items-center gap-1.5">
            ⬇️ Download File
          </a>
          <button onclick="closeGuideModal()" class="p-1.5 text-slate-400 hover:text-white bg-slate-800 hover:bg-slate-700 rounded-lg text-sm transition ml-2">
            ✕
          </button>
        </div>
      </div>
      <!-- Modal Content Body -->
      <div class="p-5 md:p-8 overflow-y-auto flex-1 font-mono text-xs md:text-sm text-slate-200 leading-relaxed bg-slate-900/90 whitespace-pre-wrap select-text" id="modal-body-text">
      </div>
    </div>
  </div>
"""

# Now embed the guide contents and helper functions into <script>
json_guides = json.dumps(guide_contents)

js_code = f"""
    // Embedded Role Guides Data
    const ROLE_GUIDES = {json_guides};
    let activeModalKey = 'p1';

    const ROLE_METADATA = {{
      p1: {{ badge: 'Person 1: Signal Lead', color: 'bg-teal-500/20 text-teal-400', title: 'ROLE_1_SIGNAL_ENGINEER.md — Signal Cleaning & Features', file: 'guides/ROLE_1_SIGNAL_ENGINEER.md' }},
      p2: {{ badge: 'Person 2: ML Lead', color: 'bg-sky-500/20 text-sky-400', title: 'ROLE_2_ML_ENGINEER.md — XGBoost & SHAP Explainability', file: 'guides/ROLE_2_ML_ENGINEER.md' }},
      p3: {{ badge: 'Person 3: App Lead', color: 'bg-purple-500/20 text-purple-400', title: 'ROLE_3_APP_DEVELOPER.md — Streamlit Telemetry Dashboard', file: 'guides/ROLE_3_APP_DEVELOPER.md' }},
      p4: {{ badge: 'Person 4: Pitch Lead', color: 'bg-amber-500/20 text-amber-400', title: 'ROLE_4_PITCH_PRODUCT_LEAD.md — Slide Deck & 3-Min Pitch', file: 'guides/ROLE_4_PITCH_PRODUCT_LEAD.md' }}
    }};

    function openGuideModal(roleKey) {{
      activeModalKey = roleKey;
      const meta = ROLE_METADATA[roleKey];
      const content = ROLE_GUIDES[roleKey] || 'Guide not found.';

      document.getElementById('modal-role-badge').innerText = meta.badge;
      document.getElementById('modal-role-badge').className = 'px-2.5 py-1 rounded-full text-xs font-bold uppercase tracking-wider ' + meta.color;
      document.getElementById('modal-title').innerText = meta.title;
      document.getElementById('modal-download-link').href = meta.file;
      document.getElementById('modal-download-link').setAttribute('download', meta.file.split('/').pop());
      document.getElementById('modal-body-text').innerText = content;

      document.getElementById('guide-modal').classList.remove('hidden');
      document.body.style.overflow = 'hidden';
    }}

    function closeGuideModal() {{
      document.getElementById('guide-modal').classList.add('hidden');
      document.body.style.overflow = '';
    }}

    // Close on Escape key or clicking outside
    document.addEventListener('keydown', (e) => {{
      if (e.key === 'Escape') closeGuideModal();
    }});
    document.getElementById('guide-modal')?.addEventListener('click', (e) => {{
      if (e.target.id === 'guide-modal') closeGuideModal();
    }});

    function copyCurrentModalText() {{
      const text = document.getElementById('modal-body-text').innerText;
      navigator.clipboard.writeText(text).then(() => {{
        const btn = document.getElementById('modal-copy-btn');
        btn.innerText = '✅ Copied to Clipboard!';
        btn.classList.add('bg-teal-500', 'text-slate-900');
        setTimeout(() => {{
          btn.innerText = '📋 Copy Entire Guide';
          btn.classList.remove('bg-teal-500', 'text-slate-900');
        }}, 2500);
      }});
    }}

    function copyGuideText(roleKey) {{
      const content = ROLE_GUIDES[roleKey] || '';
      navigator.clipboard.writeText(content).then(() => {{
        alert('Copied ' + ROLE_METADATA[roleKey].title + ' to clipboard! Ready to paste into AI or IDE.');
      }});
    }}
"""

# Insert modal HTML before </body>
html = html.replace("</body>", modal_html + "\n</body>")

# Insert JS code before </script>
html = html.replace("  </script>", js_code + "\n  </script>")

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)

print("Updated index.html successfully with interactive in-browser modal reader and direct download links!")
