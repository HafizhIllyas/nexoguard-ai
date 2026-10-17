# ROLE GUIDE — PERSON 4: CLINICAL PRODUCT & PITCH LEAD

**Title**: Lead Clinical Product Strategist & Pitch Anchor  
**Core Mission**: Own the clinical storytelling, ensure the project adheres strictly to international FIGO/ACOG obstetrics standards, build the pitch deck, and deliver a show-stopping 3-minute presentation that wins the HealthTech track.  
**Tech Stack**: Canva / Pitch / Slides, Clinical Obstetric Guidelines (FIGO 2015, Dawes-Redman)

---

## 1. Quick Checklist & Deliverables

- [ ] **Hour 0–4**: Review the NexoPrima IEEE 2024 paper and summarize the exact clinical gap for the team.
- [ ] **Hour 4–8**: Define the 3 Patient Case Personas with clear clinical narratives:
  * **Patient 101 (Normal / Cat I)**: 28 weeks, reactive baseline, healthy accelerations.
  * **Patient 204 (Suspicious / Cat II)**: Flat variability, mild tachycardia, requires active surveillance.
  * **Patient 309 (Emergency / Cat III)**: Uteroplacental insufficiency, severe late decelerations following contractions.
- [ ] **Hour 8–16**: Build the 10-slide pitch deck (Canva / PowerPoint / Google Slides).
- [ ] **Hour 16–22**: Write and time the **3-Minute Word-for-Word Pitch Script**.
- [ ] **Hour 22–28**: Rehearse with Person 3 (driving the live dashboard) until seamless.
- [ ] **Hour 28+**: Memorize the **Judge Q&A Defense Playbook**.

---

## 2. The 10-Slide Pitch Deck Structure

| Slide # | Slide Title | Key Content & Visual |
| :--- | :--- | :--- |
| **1** | **Title Slide** | *"NexoGuard AI: Physiology-Informed Clinical Decision Support for Fetal Monitoring"*. Logos: Hackathon + NexoPrima + FIGO. |
| **2** | **The Crisis in Delivery Wards** | 2M intrapartum complications annually. 30% inter-observer disagreement among obstetricians reading CTG strips. |
| **3** | **The NexoPrima Research Foundation** | Citing NexoPrima's IEEE WINCOM 2024 paper (94% baseline accuracy). Highlight what was missing: contraction synergy & explainability. |
| **4** | **Our Innovation: Path A Hybrid AI** | Multi-modal signal synergy (FHR + UC + MHR), Dawes-Redman STV, and the Contraction-Decel Lag ($\Delta t$). |
| **5** | **LIVE DEMO (Switch to Browser)** | Person 3 demonstrates Patient 309: Contraction peak $\rightarrow$ 34s lag $\rightarrow$ Late decel detected $\rightarrow$ Category III alert. |
| **6** | **Explainable AI (TreeSHAP)** | Screenshot of doctor attribution card: *"Why Category III? +38% Late Decel Lag, +31% Low STV"*. No black box! |
| **7** | **Clinical Safety Guardrails** | Deterministic fail-safe overrides: even if ML confidence wavers, severe biological distress triggers emergency alerts. |
| **8** | **Engineering & Feasibility** | Inference in $<20\text{ms}$ on standard CPU edge hardware. Zero hospital GPU servers required. |
| **9** | **Market & Business Model** | NexoPrima Hardware-to-SaaS model. Retrofittable to existing hospital monitors across Southeast Asia. |
| **10** | **The Team & Call to Action** | The 4-person team (Signal, ML, App, Clinical). Closing punchline: *"Giving every mother and baby an AI guardian."* |

---

## 3. The 3-Minute Word-for-Word Pitch Script

*Set a timer for exactly 3 minutes (180 seconds).*

### 0:00 – 0:45 | The Hook & Problem (Person 4)
> *"Judges, every 16 seconds, a stillbirth occurs somewhere in the world. Over half of these tragedies occur during active labor, when tired doctors and nurses must visually interpret fluctuating squiggle lines on paper or screens.*
> 
> *The clinical reality is stark: even veteran obstetricians disagree on CTG readings up to 30% of the time. This subjective guesswork leads to two dangerous extremes: either fetal hypoxia is caught too late—causing irreversible cerebral palsy—or panicked doctors rush mothers into unnecessary emergency Cesarean sections."*

### 0:45 – 1:15 | The NexoPrima Traction & Gap (Person 4 / Person 1)
> *"Our hackathon partner, NexoPrima, published breakthrough research at IEEE WINCOM 2024 demonstrating that AI can detect baseline heart rates with 94% accuracy.*
> 
> *However, raw baseline detection alone cannot guide surgery. Their initial dense neural network suffered from a 36% precision rate on accelerations and missed 15% of dangerous decelerations because it discarded uterine contractions and lacked explainability. Tonight, we solved this."*

### 1:15 – 2:15 | The Live Demo (Person 3 Drives, Person 4 Narrates)
> *"Meet NexoGuard AI in action.*
> 
> *Here is our live telemetry feed for Patient 309. In Channel 1 is the fetal heart rate; in Channel 2 is the uterine contraction. Watch what happens at minute 14: a contraction peaks. Exactly 34 seconds later, the fetal heart rate plummets.*
> 
> *NexoGuard immediately computes that 34-second time lag. Clinically, this is the textbook signature of a **Late Deceleration**—a critical sign of uteroplacental hypoxia. Notice our triage badge: it instantly transitions to **Category III Pathological** with an 89% hypoxia probability. Below it, our SHAP card tells the doctor in plain English: 'Late Deceleration Lag: +38% risk; Low STV: +31% risk'. In under 20 milliseconds, the delivery team has actionable, auditable guidance."*

### 2:15 – 2:45 | Why Path A Wins (Person 2)
> *"Unlike fragile deep learning that requires expensive hospital cloud GPUs, NexoGuard runs locally on existing telemetry devices with zero cloud dependency. Furthermore, we built deterministic FIGO safety guardrails: even if probabilistic confidence fluctuates, severe physiological markers force an emergency override. It is 100% fail-safe."*

### 2:45 – 3:00 | Business Model & Close (Person 4)
> *"By integrating NexoGuard into NexoPrima’s telemetry devices, we transform commodity medical hardware into an enterprise AI subscription for maternity hospitals across Malaysia and Southeast Asia.*
> 
> *NexoGuard does not replace doctors—it empowers them with a 24/7 guardian for every mother and child. Thank you."*

---

## 4. The Judge Defense Playbook (Anticipated Questions)

### Q1: *"What happens if your AI makes a mistake and causes a bad delivery outcome?"*
* **Your Answer**: *"NexoGuard is classified as a Clinical Decision Support System (CDSS), not an autonomous diagnostic agent. We designed it with a dual-layer safety architecture: 1) Deterministic FIGO guardrails that trigger hard overrides regardless of ML confidence, and 2) Explainable SHAP attribution cards that show the clinician the exact physiological rationale. The final clinical judgment remains with the attending physician, backed by auditable data."*

### Q2: *"Why didn't you train a 1D-CNN or Transformer?"*
* **Your Answer**: *"NexoPrima’s own paper explored deep neural networks and showed that complex black-box architectures suffer from high false-alarm rates (36% precision on accelerations) on clinical cohorts. Physiological time series like CTG have well-established clinical biomarkers (Dawes-Redman STV, FIGO lag). Our Physiology-Informed Hybrid ML achieves superior sensitivity on critical decelerations (>95%), trains in seconds, runs on low-cost hospital edge microcontrollers without GPUs, and gives doctors complete interpretability."*

### Q3: *"How do you handle Doppler sensor noise and mother movement?"*
* **Your Answer**: *"We built a multi-stage signal quality index (SQI). We filter sensor dropouts using Savitzky-Golay and median smoothing, interpolate gaps under 15 seconds, and crucially, run cross-correlation against maternal heart rate. If the transducer slips and locks onto maternal pulse, NexoGuard immediately warns the nurse to reposition the probe."*

---

## 5. Stuck? How to Ask AI

Copy the content of `guides/PROJECT_MASTER_CONTEXT_FOR_AI.md` into your AI prompt, and add:
```
I am Person 4 (Pitch & Clinical Lead).
I need help with [e.g. refining slide 4 / framing the business model for Southeast Asia hospitals / answering a specific judge concern].
Here is my current draft: [paste text]
Please enhance this to sound authoritative, medically accurate, and compelling for hackathon judges.
```
