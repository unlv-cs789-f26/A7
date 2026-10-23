#!/usr/bin/env bash
# Instructor: publish this folder as the public repo unlv-cs789-f26/A7 (run once, on your Mac).
# instructor/, *.docx/*.pdf and the instructor tools are excluded by .gitignore.
# Publish after the A6 deadline (Oct 21): reference/a6_reference.py is the A6 solution.
set -e
cd "$(dirname "$0")"
rm -f .git/*.lock .git/objects/*.lock 2>/dev/null || true
[ -d .git ] || git init -b main
git add -A
git commit -m "CS 789 A7: SOC Copilot II - n8n triage, Heretic abliteration, dashboards" || true
gh repo create unlv-cs789-f26/A7 --public --source=. --remote=origin --push || git push -u origin main
echo "Students open: https://colab.research.google.com/github/unlv-cs789-f26/A7/blob/main/A7_SOC_Copilot_II.ipynb"
