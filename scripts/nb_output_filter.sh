# Git clean/smudge filter: notebooks enter git without outputs, yet git never
# deletes the outputs on disk. Built into the nix store by flake.nix
# (`nb-output-filter`); the devShell points filter.nbstripout.{clean,smudge} at it.
#
#   clean  <path>: stdin = working-tree notebook, stdout = nbstripout(stdin).
#                  If stripping changed anything, cache the full file for <path>,
#                  keyed by the hash of its stripped form.
#   smudge <path>: stdin = stored (stripped) blob. If the cache for <path> holds
#                  a notebook whose stripped form is exactly this blob, emit the
#                  cached full file (outputs restored); otherwise pass through.
#
# This makes every git rewrite of a notebook with unchanged inputs lossless:
# pre-commit's stash (`git checkout -- .`, then `git apply`, which cleans the
# old file and smudges the patched one) and checkouts to identical inputs.
# One entry per absolute path, so the cache is bounded by the notebooks' size.
set -euo pipefail

mode=$1
path=$2
cache="${XDG_CACHE_HOME:-$HOME/.cache}/nb-output-filter"
# Git runs filters from the worktree top with <path> relative to it, so the
# absolute path separates worktrees.
key=$(printf '%s' "$PWD/$path" | sha256sum | cut -d' ' -f1)
full_cached="$cache/$key.ipynb"
sum_cached="$cache/$key.sha256"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cat >"$tmp/in"

case "$mode" in
clean)
  nbstripout <"$tmp/in" >"$tmp/out"
  if ! cmp -s "$tmp/in" "$tmp/out" &&
    ! { [ -f "$sum_cached" ] && cmp -s "$tmp/in" "$full_cached"; }; then
    mkdir -p "$cache"
    # Write-then-rename so a concurrent smudge never sees a torn entry; the
    # hash is written last, so a stale hash can only miss, never mismatch.
    rm -f "$sum_cached"
    cp "$tmp/in" "$tmp/full" && mv "$tmp/full" "$full_cached"
    sha256sum <"$tmp/out" | cut -d' ' -f1 >"$tmp/sum" && mv "$tmp/sum" "$sum_cached"
  fi
  cat "$tmp/out"
  ;;
smudge)
  if [ -f "$sum_cached" ] && [ -f "$full_cached" ] &&
    [ "$(sha256sum <"$tmp/in" | cut -d' ' -f1)" = "$(cat "$sum_cached")" ]; then
    cat "$full_cached"
  else
    cat "$tmp/in"
  fi
  ;;
*)
  echo "usage: nb-output-filter clean|smudge <path>" >&2
  exit 2
  ;;
esac
