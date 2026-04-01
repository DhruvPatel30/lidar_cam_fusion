You are an agent. Run and diagnose the verification script for this ROS 2 LiDAR-Camera fusion project.

Step to verify: $ARGUMENTS (if blank, detect the latest step from existing verify_step*.py files)

Follow these steps exactly:

## Step 1 — Identify the script
If $ARGUMENTS is a number (e.g. "3"), run `scripts/verify_step3.py`.
If blank, list `scripts/verify_step*.py` files and pick the highest-numbered one.

## Step 2 — Check prerequisites
Before running, verify:
- The Pixi environment is active (check if `rclpy` is importable by reading pixi.toml for the ROS dependency)
- The package is built: check if `install/` directory exists. If not, tell the user to run `pixi run build` first.
- Read the verify script to understand what it tests.

## Step 3 — Run the script
Execute: `pixi run verify<N>` (where N is the step number).
Capture full stdout and stderr.

## Step 4 — Parse results
For each FAIL line in the output:
1. Quote the exact failure message
2. Read the relevant source file(s) to find the root cause
3. Explain WHY it failed in plain English
4. Suggest the exact fix (code change, missing file, wrong parameter, etc.)

For each WARN line:
1. Explain what it means and whether it needs action

## Step 5 — Fix if possible
If any failures are due to missing files, incorrect entry points, or simple code issues:
- Apply the fix directly using the Edit or Write tools
- Re-run the verify script to confirm the fix worked

If failures require significant new implementation (e.g. a whole node is missing):
- Do NOT attempt to write the full implementation
- Instead clearly state: "This requires implementing X — use `/scaffold-node` to generate it"

## Step 6 — Report
Print a summary:
- Overall: PASSED / FAILED
- List of checks that passed
- List of checks that failed with root cause
- List of fixes applied (if any)
- Next steps for the user
