# /experiment — Design, plan, critique, and implement an experiment

You are running a multi-phase workflow to take an experiment idea from concept to a merged PR. The user's idea is: $ARGUMENTS

## Phase 1: Design (interactive)

I've got an idea I want to talk through with you. I'd like you to help me turn it into a fully formed design and spec (and eventually an implementation plan).

Check out the current state of the project in our working directory to understand where we're starting off, then ask me questions, one at a time, to help refine the idea.

Ideally, the questions would be multiple choice, but open-ended questions are OK, too. Don't forget: only one question per message.

Once you believe you understand what we're doing, stop and describe the design to me, in sections of maybe 200-300 words at a time, asking after each section whether it looks right so far.

## Phase 2: Implementation plan

Once the user confirms the design is correct, write a comprehensive implementation plan.

**Audience:** A skilled developer with zero context on this codebase and questionable taste. They know almost nothing about our toolset or problem domain. They don't know good test design very well.

**Plan requirements:**
- Document everything they need to know: which files to touch for each task, code patterns to follow, testing strategy, docs they might need to check, how to test each piece.
- Give them the whole plan as bite-sized tasks. Each task ends with a commit.
- Principles: DRY, YAGNI, TDD, frequent commits.
- Every experiment MUST live in its own folder under `experiments/<experiment-name>/`.
- The plan MUST specify that the implementer maintains an **experiment log** at `experiments/<experiment-name>/EXPERIMENT_LOG.md`. This log should record:
  - Every command run (or at least the significant ones)
  - Milestones hit and when
  - Roadblocks faced and how they were resolved
  - Summary of results as they come in
  - By reading the experiment log, someone should be able to fully understand the progress of the experiment without having been there.
- Write the plan to `experiments/<experiment-name>/plan.md`.

## Phase 3: Critique

After writing the plan, spawn a **critic agent** to review it. The critic should:
- Read the plan AND the relevant parts of the codebase for context
- Check for bugs (wrong paths, wrong imports, off-by-one errors, wrong `parents[]` depth)
- Check for security/design gaps
- Check for missing edge cases
- Check for inconsistencies with the existing codebase patterns
- Report issues as a numbered list with severity (bug / design gap / minor)

Present the critic's feedback to the user and ask if they want to address any of it. Update the plan based on the feedback.

## Phase 4: Implementation

Once the user approves the plan, spawn an **implementer agent** to execute it. Tell the implementer:

- Follow the plan task by task, committing after each task.
- Maintain `experiments/<experiment-name>/EXPERIMENT_LOG.md` — append to it as you go.
- Run tests after each task. If tests fail, fix before moving on.
- The LAST step is to create a PR using `gh pr create`. The PR description should summarize what the experiment is, link to the plan, and include the key results (if any are available at implementation time).

While the implementer is working, periodically review their progress (check the files they've created, whether tests pass, whether they're following the plan).

## Important rules

- Each experiment gets a new folder: `experiments/<experiment-name>/`
- The experiment folder should contain: `plan.md`, `EXPERIMENT_LOG.md`, and all scripts/data/reports for that experiment
- Do NOT scatter experiment files across `scripts/`, `results/`, `docs/reports/` — everything goes in the experiment folder
- Shared library code (reusable across experiments) goes in `pkg/` with unit tests in `tests/unit/`
- Do NOT modify existing files unless the plan explicitly calls for it
