#!/usr/bin/env bash
# Refresh the Kaggle slim snapshot branch of experiments.stochastic_fit.
#
# Kaggle caps a kernel's source at ~1 MB, so omnirun jobs on it are submitted
# from an orphan branch that holds only the job's import closure (this package
# + a five-dependency pyproject + omnirun.toml), checked out in
# .worktrees/kaggle-slim-stochfit. Run from the stochastic-fit worktree after
# committing; then submit from the slim worktree.
set -euo pipefail
BRANCH=kaggle-slim-stochfit
SLIM=$(mktemp -d)
mkdir -p "$SLIM/src/experiments/stochastic_fit"
cp src/experiments/__init__.py "$SLIM/src/experiments/"
cp src/experiments/stochastic_fit/{__init__,data,model,fit,rig,run}.py "$SLIM/src/experiments/stochastic_fit/"
cat > "$SLIM/pyproject.toml" <<'EOF'
[project]
name = "stochastic-fit-slim"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["numpy>=2,<3", "scipy>=1.13", "torch==2.7.0", "soundfile>=0.12", "boto3>=1.34"]
EOF
cat > "$SLIM/omnirun.toml" <<'EOF'
[job]
outputs = ["results/**"]

[job.resources]
gpus = 1
time = "8h"

[job.env]
# uv, not system: Kaggle's image ships a torch without sm_60 kernels (P100);
# PyPI torch 2.7.0 (cu126) still has them.
kind = "uv"
EOF
printf '__pycache__/\nresults/\n.cache/\n*.log\n.env\n' > "$SLIM/.gitignore"
export GIT_INDEX_FILE=$(mktemp -u)
git --work-tree="$SLIM" add -A
TREE=$(git write-tree)
COMMIT=$(echo "kaggle slim snapshot of experiments.stochastic_fit @ $(git rev-parse --short HEAD)" | git commit-tree "$TREE")
unset GIT_INDEX_FILE
rm -rf "$SLIM"
MAIN=$(git rev-parse --path-format=absolute --git-common-dir); MAIN=${MAIN%/.git}
WT="$MAIN/.worktrees/$BRANCH"
if [ -d "$WT" ]; then
  git -C "$WT" checkout -q --detach "$COMMIT"   # release the branch, then move it
  git branch -f "$BRANCH" "$COMMIT"
  git -C "$WT" checkout -q "$BRANCH"
else
  git branch -f "$BRANCH" "$COMMIT"
  git -C "$MAIN" worktree add "$WT" "$BRANCH" >/dev/null
fi
git push -q -f origin "$BRANCH"
[ -e "$WT/.env" ] || ln -s ../../.env "$WT/.env"
echo "$WT @ $(git rev-parse --short "$COMMIT")"
