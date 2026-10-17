---
name: skill-library
description: Search and route requests to off-stack or auxiliary ECC agents, skills, and rules stored in the project library without loading them by default.
---

# Skill & Agent Library Router

This repository uses an evidence-backed `DAILY` vs `LIBRARY` separation established via the `agent-sort` methodology.

- **DAILY (`agents/`, `.claude/rules/`)**: Loaded every session. Contains only components matching the NexoGuard AI stack: Python 3.9+, FastAPI, Streamlit, DSP, XGBoost, FIGO 2015 CDS, and TDD verification.
- **LIBRARY (`library/`)**: Archived for on-demand access. Retained to prevent bloat and context clutter, but fully searchable and retrievable when specialized workflows arise.

## Library Catalog Locations

| Category | Path | Contents |
| :--- | :--- | :--- |
| **Agents** | `library/agents/*.md` | 53 specialized agents (Swift, Rust, Go, C++, C#, Java, Dart, PHP, Ruby, Vue, React, GAN, Homelab, etc.) |
| **Rules** | `library/rules/*.md` | Stale or secondary environment rules (e.g. `node.md`) |
| **Workflows** | `library/workflows/*` | Upstream ECC automation scripts (e.g. `ecc-pro-security-roadmap.js`) |

## Grouped Trigger Routing

When a task requires specialized expertise outside the core Python/FastAPI/ML stack, consult the matching library file:

### 1. Mobile & Cross-Platform
- Flutter / Dart: `library/agents/flutter-reviewer.md`, `library/agents/dart-build-resolver.md`
- iOS / Swift: `library/agents/swift-reviewer.md`, `library/agents/swift-build-resolver.md`
- HarmonyOS: `library/agents/harmonyos-app-resolver.md`

### 2. Alternative Backend & Systems
- Rust: `library/agents/rust-reviewer.md`, `library/agents/rust-build-resolver.md`
- Go: `library/agents/go-reviewer.md`, `library/agents/go-build-resolver.md`
- C++: `library/agents/cpp-reviewer.md`, `library/agents/cpp-build-resolver.md`
- C# / .NET / F#: `library/agents/csharp-reviewer.md`, `library/agents/fsharp-reviewer.md`
- Java / Kotlin: `library/agents/java-reviewer.md`, `library/agents/java-build-resolver.md`, `library/agents/kotlin-reviewer.md`
- PHP: `library/agents/php-reviewer.md`
- Ruby on Rails: `library/agents/ruby-reviewer.md`
- Django: `library/agents/django-reviewer.md`, `library/agents/django-build-resolver.md`

### 3. Frontend Frameworks
- React / Next.js: `library/agents/react-reviewer.md`, `library/agents/react-build-resolver.md`
- Vue / Nuxt: `library/agents/vue-reviewer.md`
- TypeScript: `library/agents/typescript-reviewer.md`

### 4. Infrastructure & Marketing
- Network & Routing: `library/agents/network-architect.md`, `library/agents/network-config-reviewer.md`, `library/agents/network-troubleshooter.md`
- Homelab: `library/agents/homelab-architect.md`
- Marketing & SEO: `library/agents/marketing-agent.md`, `library/agents/seo-specialist.md`
- Open-Source Packaging: `library/agents/opensource-forker.md`, `library/agents/opensource-packager.md`, `library/agents/opensource-sanitizer.md`
- GAN Harness: `library/agents/gan-evaluator.md`, `library/agents/gan-generator.md`, `library/agents/gan-planner.md`

## How to Promote a Component to DAILY

If the project stack evolves (e.g. adding a React native frontend or C++ DSP module):
1. Copy the relevant file from `library/agents/<agent>.md` into `agents/<agent>.md`.
2. Cite the new repository evidence in the commit message.
3. Verify test coverage and lint checks remain green.
