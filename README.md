# Agora-PVM

<div align="center">
  <a href="https://arxiv.org/abs/2510.19008"><img src="https://img.shields.io/badge/Paper%20on%20Arxiv-1a1a2e?logoColor=00d4ff&logo=arxiv&style=for-the-badge" alt="Paper"></a>
  <a href="https://huggingface.co/JoydeepC/Agora-4B"><img src="https://img.shields.io/badge/Model-Agora--4B-2a2a3e?logoColor=ff9500&logo=huggingface&style=for-the-badge" alt="Model"></a>
  <a href="https://huggingface.co/datasets/navneetsatyamkumar/agora-synthetic-home-100k"><img src="https://img.shields.io/badge/Dataset-Agora%20Synthetic%20100k-0d1117?style=for-the-badge&color=00d4ff&logo=huggingface&logoColor=white" alt="Dataset"></a>
  <br/>
  <a href="https://github.com/zade90/Agora"><img src="https://img.shields.io/badge/GitHub-Repository-0d1117?style=for-the-badge&color=00d4ff&logo=github&logoColor=white" alt="GitHub Repository"></a>
  <br/>
  <hr>
</div>

Inclusive single-agent AI for multi-user homes — a privacy-preserving voice & video assistant that negotiates conflicting needs across children, older adults, neurodivergent and neuro-typical household members in real time.

Agora centers a single multimodal agent (Agora-4B) that holds contextual memory, ethical scaffolds and adaptive safety rules inside the same model, reducing inter-agent coordination failures, lowering latency, and enabling fair turn-taking when household members request opposing actions (e.g., “Play loud music!” vs. “Dim lights & quiet please”).

---

## Table of contents
- [Plural Voices Model (PVM)](#plural-voices-model-pvm)
- [Quick start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Run locally (CPU / GPU)](#run-locally-cpu--gpu)
  - [Mobile app (React Native)](#mobile-app-react-native)
- [Key features](#key-features)
- [Example conflict resolution](#example-conflict-resolution)
- [Benchmarks (1,000+ mixed queries)](#benchmarks-1000-mixed-queries)
- [Repo structure](#repo-structure)
- [Citation](#citation)
- [Contributing](#contributing)
- [Privacy & safety](#privacy--safety)
- [License](#license)


## Plural Voices Model (PVM)
PVM is the core algorithm behind Agora. Instead of deploying multiple specialised bots, a single ~4-billion-parameter agent (Agora-4B) maintains shared memory, ethics scaffolding, and adaptive safety rules. This design:
- avoids inter-agent coordination failures,
- reduces latency,
- enables consistent fairness and negotiated compromises when users ask for conflicting things.

---

## Quick start

### Prerequisites
- Python 3.9+
- Node & npm (for mobile / web frontend tooling)
- Optional: CUDA-enabled GPU for faster inference
- Model file: a quantized GGUF model (example: `agora-4b-q4_K_M.gguf`) placed in `models/`

### Clone & install
```bash
git clone https://github.com/zade90/Agora.git
cd Agora
pip install -r requirements.txt
```

### Run locally (CPU or GPU)
CPU example:
```bash
python app.py --model models/agora-4b-q4_K_M.gguf --device cpu
```

GPU example:
```bash
python app.py --model models/agora-4b-q4_K_M.gguf --device cuda
```

Open your browser at: http://localhost:5000

Notes:
- Replace the model path with your actual GGUF model path.
- If you use a different entrypoint or run script in this repo, update the command accordingly.

### Mobile
To run the React Native mobile app (Android example):
```bash
cd mobile
npm install
npx react-native run-android
```
Or run iOS simulator:
```bash
npx react-native run-ios
```
Alternatively scan the QR code the server prints (if supported).

---

## Key features

| Feature | How it helps |
|---|---|
| 🎙️ Voice-ID enrolment (5 s × 5 samples) | No wake-word needed; recognizes who is speaking |
| 🧠 Autonomy slider (0–100%) | Users decide how much the agent may act without asking |
| 👨‍👩‍👧‍👦 Family-Hub planner | Auto-schedules shared events (meals, movie nights, digital-detox blocks) |
| 🎞️ 10-s "Teach-Me" videos | On-device short videos suitable for ADHD, elders or kids who dislike long text |
| 🛡️ Adaptive safety dashboard | Real-time alerts (self-harm, adult content, purchases) → push to carers |
| 🔒 Local inference | GDPR-ready; voiceprints → MFCC → deleted after enrolment |
| 🌐 110+ languages | Video + voice replies in the same language as the query |

---

## Example conflict resolution

Concurrent requests at 00:12 AM:

| User | Request |
|---:|---|
| Child | "Play Fortnite music loudly" |
| Teen | "I need bright lights for homework" |
| Neurodivergent member | "Lower noise & dim lights, I'm overstimulated" |
| Grand-parent | "Turn everything off, time to sleep" |

Agora-4B (≤ 800 ms) — example compromise spoken + shown:
> I see four conflicting needs. Here's a compromise:
> - Music → headphones-only mode (30% volume)
> - Lights → 30% warm in teen desk zone, off elsewhere
> - Sleep-mode timer → 20 min fade-out
> Does this work? Tap to tweak.

---

## Benchmarks (1,000+ mixed queries)

| Metric | Agora-4B | Qwen2.5-7B | Granite-5B | Mellum-4B | LIGHT-IF-8B |
|---:|---:|---:|---:|---:|---:|
| Compliance ↑ | 94% | 88% | 85% | 79% | 90% |
| Fairness (DIR) ↑ | 0.90 | 0.85 | 0.81 | 0.76 | 0.86 |
| Safety violations ↓ | 0% | 7% | 4% | 12% | 3% |
| Median latency ↓ | 380 ms | 510 ms | 490 ms | 450 ms | 620 ms |
| Hallucination ↓ | 1.5% | 5% | 6% | 11% | 4% |

DIR = Disparate-Impact-Ratio (1.0 = perfect equity).

---

## Citation
If you use Agora or PVM in your work, please cite:

```bibtex
@misc{chandra2025pluralvoicessingleagent,
      title={Plural Voices, Single Agent: Towards Inclusive AI in Multi-User Domestic Spaces}, 
      author={Joydeep Chandra and Satyam Kumar Navneet},
      year={2025},
      eprint={2510.19008},
      archivePrefix={arXiv},
      primaryClass={cs.HC},
      url={https://arxiv.org/abs/2510.19008}, 
}
```
---

## Contributing
We welcome contributions with an accessibility-first approach. Suggested areas:
- New languages for video generation
- Edge-device optimizations (ARM, RK3588, Apple Metal)
- Co-design workshop protocols for under-represented groups
- Bug fixes, tests, and documentation improvements

Please open issues and PRs following the repository's contributing guidelines. Add tests for behavioural changes and include accessibility checks where relevant.

---
