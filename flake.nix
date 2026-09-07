{
  description = "Python project with uv";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    git-hooks.url = "github:cachix/git-hooks.nix";
  };

  outputs = { self, nixpkgs, flake-utils, git-hooks }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python312;

        # `mk-worktree <name> [<base-ref>]` — creates .worktrees/<name>, links the
        # shared gitignored resources, then drops you into a shell there.
        # Only a launcher: the logic stays in the repo at scripts/mk-worktree.sh
        # so edits take effect without a flake rebuild.
        mk-worktree = pkgs.writeShellScriptBin "mk-worktree" ''
          set -euo pipefail

          no_shell=0
          args=()
          for a in "$@"; do
            case "$a" in
              -n|--no-shell) no_shell=1 ;;
              *) args+=("$a") ;;
            esac
          done

          common="$(${pkgs.git}/bin/git rev-parse --path-format=absolute --git-common-dir 2>/dev/null)" || {
            echo "mk-worktree: not in a git repository" >&2
            exit 1
          }
          main="''${common%/.git}"
          main="''${main%/}"
          script="$main/scripts/mk-worktree.sh"
          [ -x "$script" ] || {
            echo "mk-worktree: $script not found or not executable" >&2
            exit 1
          }

          dest="$("$script" "''${args[@]}")"

          if [ "$no_shell" = 1 ]; then
            printf '%s\n' "$dest"
            exit 0
          fi

          # An executable cannot cd its parent shell, so start a shell in the
          # worktree instead. direnv loads the worktree env; `exit` returns you
          # to where you were.
          cd "$dest"
          exec "''${SHELL:-${pkgs.bashInteractive}/bin/bash}"
        '';

        pre-commit-check = git-hooks.lib.${system}.run {
          src = ./. ;
          hooks = {
            ruff = {
              enable = true;
              # ruff is a placeholder in git-hooks.nix; provide real package
              package = pkgs.ruff;
            };
            ruff-format = {
              enable = true;
              package = pkgs.ruff;
            };
            pyright = {
              enable = true;
            };
            # Syntax-check every YAML file (Hydra configs included). Uses the
            # pre-commit-hooks PyYAML, independent of the project .venv.
            check-yaml = {
              enable = true;
            };
            validate-experiment-docs = {
              enable = true;
              name = "experiment docs";
              entry = "${python}/bin/python scripts/validate_experiment_docs.py";
              # Whole-set contract (not per-file): run once whenever any
              # experiment config or experiment doc changes.
              files = "^(conf/experiment/.*\\.(yaml|md)|docs/experiments/.*\\.md|scripts/validate_experiment_docs\\.py)$";
              pass_filenames = false;
              language = "system";
            };
            lint-imports = {
              enable = true;
              name = "import-linter";
              # import-linter is a project dev dependency; the devshell hook
              # activates .venv, so the console script is on PATH.
              entry = "lint-imports";
              # Whole-graph contract (not per-file): run once whenever any
              # Python module changes.
              files = "\\.py$";
              pass_filenames = false;
              language = "system";
            };
          };
        };
      in
      {
        # Run hooks with `nix fmt`
        formatter =
          let
            inherit (pre-commit-check.config) package configFile;
            script = ''
              ${pkgs.lib.getExe package} run --all-files --config ${configFile}
            '';
          in
          pkgs.writeShellScriptBin "pre-commit-run" script;

        # Run hooks sandboxed with `nix flake check`
        checks = {
          inherit pre-commit-check;
        };

        devShells.default = pkgs.mkShell {
          buildInputs = pre-commit-check.enabledPackages ++ [ mk-worktree ] ++ (with pkgs; [
            python
            uv
            # C++ standard library for NumPy and other native dependencies
            stdenv.cc.cc.lib
            # Additional libraries commonly needed by Python packages
            zlib
            libffi
            # Graphviz for pygraphviz
            graphviz
            pkg-config
            # Playwright browser automation — python package + NixOS-provided browsers.
            # The nixpkgs python package is patched to use store paths for the node
            # driver, so it works on NixOS without nix-ld.  playwright-driver.browsers
            # includes chromium + headless_shell (required by default launch()).
            python312Packages.playwright
            playwright-driver.browsers
            # LaTeX toolchain for writing/papers: Tectonic — a self-contained
            # modern engine (XeTeX core) that fetches packages on demand into
            # its own cache, replacing the multi-GB texlive.combine set.
            # Build: `tectonic main.tex` (or `tectonic -X compile`); biblatex
            # workflows use `tectonic -X build` with a Tectonic.toml, classic
            # bibtex ones just work via the automatic rerun logic.
            tectonic
	    # easier latex for easier docs
	    typst
            # for looking at resulting pdfs
            poppler-utils
	    # unavoidable js
	    nodejs
          ]);

          shellHook = ''
            ${pre-commit-check.shellHook}
            if [ ! -d .venv ]; then
              uv venv
            fi
            source .venv/bin/activate
            # Set LD_LIBRARY_PATH to find C++ standard library and other native libraries
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:${pkgs.graphviz}/lib:$LD_LIBRARY_PATH"
            export PKG_CONFIG_PATH="${pkgs.graphviz}/lib/pkgconfig:$PKG_CONFIG_PATH"
            # Point Playwright at the NixOS-provided browsers (chromium, headless_shell, ffmpeg).
            export PLAYWRIGHT_BROWSERS_PATH="${pkgs.playwright-driver.browsers}"
            # Skip host-requirements validation — NixOS browsers are already patched.
            export PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS=true
            echo "Python $(python --version) with uv $(uv --version)"
          '';
        };
      });
}
