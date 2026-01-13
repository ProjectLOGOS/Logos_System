# Contributing Guidelines

Thank you for supporting the LOGOS proof-aligned demo. Follow the steps below to
submit high-quality changes.

## 1. Environment Setup

1. Install the system prerequisites listed in `README.md` (Python 3.11+, Coq,
   GNU Make).
2. Clone the repository and optionally create a virtual environment:

   ```bash
   git clone https://github.com/ProjectLOGOS/pxl_demo_wcoq_proofs.git
   cd pxl_demo_wcoq_proofs
   python3 -m venv .venv
   . .venv/bin/activate
   pip install -r requirements.txt
   ```

## 2. Development Workflow

1. Make your code changes.
2. Run the full verification suite:

   ```bash
   python3 test_lem_discharge.py
   python3 scripts/boot_aligned_agent.py
   ```

3. (Optional) run the investor scripts to ensure the translated outputs remain
   consistent:

   ```bash
   python3 scripts/investor_demo.py
   python3 scripts/investor_dashboard.py
   ```

4. Confirm that `state/alignment_LOGOS-AGENT-OMEGA.json` contains a fresh audit
   entry documenting the constructive LEM verification.

## 3. Coding Standards

- Python scripts should remain compatible with Python 3.11+ and use only the
  standard library unless a new dependency is justified.
- Keep comments concise and reserved for non-obvious logic.
- Maintain ASCII source files unless Unicode is required for correctness.

## 4. Submitting Changes

1. Open a pull request targeting the `main` branch.
2. Provide a short summary describing the motivation, the verification steps you
   ran, and any residual risks.
3. Ensure GitHub Actions (`Constructive LEM Gate`) passes before requesting
   review.
4. Respond promptly to review feedback and keep the branch up to date with
   `main`.

## 5. Security

The agent boot process includes cryptographic hashing to bind the agent
identity to its constructive proof. If you discover a vulnerability that could
undermine this guarantee, please report it privately to the maintainers before
opening a public issue.
