# NexoGuard AI Component Library

This directory houses auxiliary agents, rules, and workflows preserved from the full ECC bundle. Per the `agent-sort` methodology, components that do not match the active project stack are retained here in the `LIBRARY` bucket rather than being permanently deleted or loaded by default.

## Directory Structure

- `agents/`: 53 off-stack language reviewers, build resolvers, and non-core agents.
- `rules/`: Secondary environment rules (e.g. `node.md`).
- `workflows/`: Upstream ECC development workflows.

To route requests to these components, see the router skill at [`.claude/skills/skill-library/SKILL.md`](../.claude/skills/skill-library/SKILL.md).
