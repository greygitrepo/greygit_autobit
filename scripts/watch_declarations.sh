#!/usr/bin/env bash
# Commits strategies/*/DECLARATIONS.md as soon as a team writes/changes it, so the git timestamp proves the
# declaration preceded the validation runs. Commits ONLY those files. Runs as unit autobit-decl-watch.
cd "$(dirname "$0")/.."
while true; do
  changed=$(git status --porcelain -- 'strategies/*/DECLARATIONS.md' | awk '{print $2}')
  if [ -n "$changed" ]; then
    for i in 1 2 3 4 5; do
      if git add -- $changed && git commit -q -m "Pre-registration: $(echo $changed | tr '\n' ' ')(auto, before validation)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- $changed; then
        git push -q 2>/dev/null || true
        echo "$(date '+%F %T') committed $changed"; break
      fi
      sleep 3
    done
  fi
  sleep 20
done
