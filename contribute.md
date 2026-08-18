# Contributing to Thermal Camera Sim (ThIsaac)

Thank you for your interest in contributing to the Thermal Camera Simulator for NVIDIA Isaac Sim! We welcome contributions to improve the thermal physics models, fix bugs, or help build out the Isaac Sim sensor integration.

## Getting Started

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd ThIsaac
   ```

2. **Set up the Python environment:**
   For pure Python development (Phase 1):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. **Isaac Sim Setup (Phase 2+):**
   For any work inside `isaac_ext/` or `spg/`, use the Python interpreter bundled with Isaac Sim (e.g., `python.sh`), not your system Python.
   - Please read `docs/isaacsim_implementation_plan.md` for prerequisites.

## Project Structure

Before jumping into the code, check out how the project is organized:

- `docs/`: Design documents and build checklists. **Read these first.**
- `ROADMAP.md`: Shows the current phase and build order.
- `thermal_physics/`: Pure Python models (emission, optics, noise).
- `isaac_ext/` & `spg/`: Isaac Sim specific extensions and shaders.
- `tests/`: Test suite.
- `legacy/`: Version 0 scripts, previous prototypes, and older media captures.

## Contribution Workflow

1. **Pick a Task:**
   Review `ROADMAP.md` and `docs/isaacsim_checklist.md`. We build strictly phase-by-phase. If you are starting something new, create an issue or let the maintainers know.

2. **Create a Branch:**
   Create a descriptive branch name for your work:

   ```bash
   git checkout -b feature/add-thermal-noise
   ```

3. **Write Code:**
   - Include type hints for all new Python functions.
   - Encapsulate logic clearly; do not inline massive setups into single scripts.
   - Follow Isaac Sim API guidelines (e.g., `isaacsim.sensors.*` namespace for Isaac Sim 6.0+).

4. **Test Your Changes:**
   Ensure existing tests pass and write new tests for your features.

   ```bash
   pytest tests/
   ```

5. **Commit:**
   - **Important Constraint:** Do **NOT** commit large binary assets (`.usd`, heavy textures, checkpoints) to Git. Keep the repository lightweight.
   - Write clear and descriptive commit messages.

6. **Submit a Pull Request:**
   Describe what your PR solves, link to any relevant roadmap phases, and request a review.

## Using AI Assistants

If you use an AI coding assistant, point it to `AGENTS.md` and `.claude/skills/thermal-camera-sim/SKILL.md` before it makes any changes. These files contain hard constraints regarding Isaac Sim versions, namespaces, and file structures.
