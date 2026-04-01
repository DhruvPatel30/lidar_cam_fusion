You are an agent. Scaffold a complete new ROS 2 node for this LiDAR-Camera fusion project.

Node to create: $ARGUMENTS

Follow these steps exactly:

## Step 1 — Read existing patterns
Read these files to understand the exact patterns to follow:
- `src/perception_pipeline/perception_pipeline/lidar_processor_node.py` (node template)
- `scripts/verify_step3.py` (verify script template)
- `src/perception_pipeline/launch/fusion_pipeline.launch.py` (launch file)
- `src/perception_pipeline/setup.py` (entry points)

## Step 2 — Create the node file
Create `src/perception_pipeline/perception_pipeline/<node_name>_node.py` following this exact structure:
1. Module docstring listing: what it does, subscribed topics, published topics, parameters
2. Pure Python/numpy/ML core class (no ROS imports) — named e.g. `CameraDetector`, fully unit-testable
3. ROS 2 Node wrapper class that: declares parameters → instantiates core class → sets up QoS → creates subscribers/publishers → implements callbacks
4. `main()` entry point with rclpy.init / spin / shutdown pattern
5. `if __name__ == "__main__": main()`

## Step 3 — Update setup.py
Add the new entry point to the `console_scripts` list in `src/perception_pipeline/setup.py`:
```
"<node_name> = perception_pipeline.<node_name>_node:main",
```

## Step 4 — Update launch file
Add the new Node() block to `src/perception_pipeline/launch/fusion_pipeline.launch.py` and add it to the LaunchDescription list. Follow the same style as the existing lidar_processor block. Add a comment marking which step it belongs to.

## Step 5 — Create verify script
Create `scripts/verify_step<N>.py` (use the next available number) with these sections:
- **A. Source files** — check node file exists, entry point in setup.py, node in launch file
- **B. Module imports** — import the core class and node class, fail fast if broken
- **C. Unit tests** — test the core class logic with synthetic data, no ROS needed
- **D. ROS node dry-run** — instantiate the node, publish a test message, verify output topics receive messages within 2 seconds using SingleThreadedExecutor
- **Summary** — print PASSED/FAILED with failure list, sys.exit(1) on failure

## Step 6 — Update pixi.toml
Add `verify<N> = "python scripts/verify_step<N>.py"` to the `[tasks]` section in `pixi.toml`.

## Step 7 — Report
List every file created or modified with a one-line summary of the change.
