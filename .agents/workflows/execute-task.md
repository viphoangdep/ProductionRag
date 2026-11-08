---
description: Automated workflow to read a task from the task/ directory, pull master, create a task-named branch, execute the task, commit, push, merge into master, and delete the branch.
---

# Workflow: Execute Task

This workflow automates the end-to-end task execution lifecycle: discovering the task specification, branching off `master`, implementing and verifying the solution, committing and pushing changes, merging into `master`, and cleaning up (deleting) the branch upon completion.

---

## Execution Steps

### Step 1: Identify and Read Task Description
1. **Locate Task File**:
   - Check if a task name or ID is provided as an argument (e.g., `PG-100`, `PG-101`).
   - Find the matching file in the `task/` directory (supports `task/<TASK_NAME>`, `task/<TASK_NAME>.md`, or `task/<TASK_NAME>.txt`).
   - If no argument is provided:
     - Scan all files inside the `task/` directory.
     - If exactly one task file exists: automatically select it.
     - If multiple task files exist or none are found: list available tasks and ask the user which one to execute.
2. **Parse Task Requirements**:
   - Read the task file thoroughly to understand:
     - Core goals and functionality required.
     - Target files and architecture impact.
     - Acceptance criteria and verification tests.

---

### Step 2: Prepare Git Environment and Create Branch
1. **Check Git Status**:
   ```bash
   git status
   ```
   - Ensure the working tree is clean before switching branches. Stash uncommitted changes with `git stash` if necessary.
2. **Pull Latest Changes from `master`**:
   ```bash
   git checkout master
   git pull origin master
   ```
3. **Create and Switch to Feature Branch**:
   - Create a new branch named after the task (e.g., `PG-100`):
   ```bash
   git checkout -b <TASK_NAME>
   ```

---

### Step 3: Implement and Verify the Task
1. **Explore Codebase**:
   - Identify relevant files, dependencies, and interfaces before making modifications.
2. **Code Implementation**:
   - Implement all requirements, bug fixes, or enhancements detailed in the task file.
3. **Run Tests & Verification**:
   - Run relevant tests and linters (e.g., `pytest tests/`) to ensure no regressions and verify that acceptance criteria are met.

---

### Step 4: Commit and Push the Feature Branch
1. **Review Diff and Stage Changes**:
   ```bash
   git status
   git diff
   git add .
   ```
2. **Commit Changes**:
   - Create a descriptive commit following Conventional Commits format referencing the task ID:
   ```bash
   git commit -m "feat(<TASK_NAME>): <concise description of changes>"
   ```
3. **Push Branch to Remote**:
   ```bash
   git push -u origin <TASK_NAME>
   ```

---

### Step 5: Merge into `master` and Push
1. **Switch Back to `master`**:
   ```bash
   git checkout master
   ```
2. **Pull Latest `master`**:
   - Ensure `master` is completely up-to-date to prevent merge conflicts:
   ```bash
   git pull origin master
   ```
3. **Merge the Feature Branch into `master`**:
   ```bash
   git merge <TASK_NAME> --no-ff -m "merge: complete task <TASK_NAME>"
   ```
   > *Note: If any merge conflicts arise, pause to resolve them thoroughly before concluding the merge.*
4. **Push Updated `master` to Remote**:
   ```bash
   git push origin master
   ```

---

### Step 6: Delete the Feature Branch (Cleanup)
After successfully merging into `master` and pushing to remote:
1. **Delete Local Branch**:
   ```bash
   git branch -d <TASK_NAME>
   ```
   *(Use `git branch -D <TASK_NAME>` only if `-d` issues a warning despite safe merge).*
2. **Delete Remote Branch**:
   ```bash
   git push origin --delete <TASK_NAME>
   ```

---

### Step 7: Summary and Report
Provide a clean summary report to the user:
- **Task ID**: `<TASK_NAME>`
- **Summary of Changes**: Key features implemented and files modified/created.
- **Verification Results**: Test pass/fail status and test suite outputs.
- **Git Operations**: Commit hash, merge status into `master`, and confirmation that `<TASK_NAME>` branch was deleted both locally and remotely.
