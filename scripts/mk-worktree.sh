#!/usr/bin/env bash
# Create (or find) a worktree of this repo under <main-checkout>/.worktrees/<name>,
# wire up the shared, gitignored resources, allow direnv, and print the path.
#
# Everything informational goes to stderr; stdout is ONLY the worktree path, so
# the `mk-worktree` shell function can do `cd "$(mk-worktree.sh "$@")"`.
#
# Usage: mk-worktree.sh <name> [<base-ref>]
#   <name>      worktree directory name AND branch name
#   <base-ref>  start point for a new branch (default: HEAD of the caller)
set -euo pipefail

say() { printf '%s\n' "$*" >&2; }
die() { printf 'mk-worktree: %s\n' "$*" >&2; exit 1; }

[ $# -ge 1 ] || die "usage: mk-worktree.sh <name> [<base-ref>]"
NAME="$1"
BASE="${2:-HEAD}"

case "$NAME" in
  -*|/*|*..*) die "invalid worktree name: $NAME" ;;
esac

GIT_COMMON="$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null)" \
  || die "not in a git repository"
MAIN="${GIT_COMMON%/.git}"
MAIN="${MAIN%/}"
WT="$MAIN/.worktrees/$NAME"

# --- create the worktree (idempotent: an existing one is just reported) ------
# NB: command substitution, not a pipe - `grep -q` closing the pipe early makes
# git die on SIGPIPE, which `set -o pipefail` would turn into a failure.
WT_LIST="$(git -C "$MAIN" worktree list --porcelain)"
if grep -qxF "worktree $WT" <<<"$WT_LIST"; then
  say "worktree already exists: $WT"
elif [ -e "$WT" ]; then
  die "$WT exists but is not a registered worktree"
else
  START="$(git rev-parse --verify "$BASE^{commit}")"
  mkdir -p "$MAIN/.worktrees"
  if git -C "$MAIN" show-ref --verify --quiet "refs/heads/$NAME"; then
    say "branch $NAME exists -> checking it out in $WT"
    git -C "$MAIN" worktree add "$WT" "$NAME" >&2
  else
    say "new branch $NAME from $BASE ($(git rev-parse --short "$START"))"
    git -C "$MAIN" worktree add -b "$NAME" "$WT" "$START" >&2
  fi
fi

# --- link the shared, gitignored resources ----------------------------------
# Relative link targets, so the whole .worktrees/ tree stays movable.
link_shared() {
  local rel="$1" up="$2" src="$MAIN/$rel" dst="$WT/$1"
  [ -e "$src" ] || return 0
  if [ -e "$dst" ] || [ -L "$dst" ]; then return 0; fi
  mkdir -p "$(dirname "$dst")"
  ln -s "$up$rel" "$dst"
  say "  linked $rel"
}

say "linking shared resources:"
for rel in .env .cache .venv data datasets models .checkpoints-cache.json; do
  link_shared "$rel" "../../"
done
link_shared ".claude/settings.local.json" "../../../"

# results/ cannot be one symlink: part of it (results/blind_corpus) is tracked,
# so git owns the directory. Link the main checkout's result dirs one by one and
# skip anything git already put there. Re-run mk-worktree to pick up new ones.
if [ -d "$MAIN/results" ]; then
  linked=0
  mkdir -p "$WT/results"
  while IFS= read -r entry; do
    base="$(basename "$entry")"
    dst="$WT/results/$base"
    if [ -e "$dst" ] || [ -L "$dst" ]; then continue; fi
    ln -s "../../../results/$base" "$dst"
    linked=$((linked + 1))
  done < <(find "$MAIN/results" -mindepth 1 -maxdepth 1)
  say "  linked $linked result dir(s) from the main checkout"
fi

# Hydra run dirs (outputs/) and wandb/ stay local to the worktree on purpose.

# --- direnv -----------------------------------------------------------------
if command -v direnv >/dev/null 2>&1; then
  direnv allow "$WT" >&2 && say "direnv allowed"
else
  say "direnv not found - skipping 'direnv allow'"
fi

say ""
say "NOTE: .venv is SHARED with the main checkout. Do not run 'uv sync' here"
say "      unless you mean to change it for every checkout. The .envrc puts"
say "      this worktree's src/ on PYTHONPATH so imports stay local."
printf '%s\n' "$WT"
