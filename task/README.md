# Task Directory

This directory stores task specification files to be executed by the agent workflow.

## File Naming Convention:
- The file name directly represents the Task ID / branch name (e.g., `PG-100.md`, `PG-101.md`, or `PG-100`).
- The Git branch will be named strictly after the file name (without the `.md` extension).

## Task File Template (`task/PG-100.md`):

```markdown
# [PG-100] Short Title of the Task

## 1. Description
Detailed description of the feature to build or bug to fix.

## 2. Affected / Target Files
- `src/...`
- `tests/...`

## 3. Acceptance Criteria
- [ ] Feature works according to specifications
- [ ] Tests written and passing: `pytest`
```
