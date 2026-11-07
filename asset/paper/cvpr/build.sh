#!/usr/bin/env bash
# Compile in an isolated temporary directory; publish only a successful PDF.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
tex_bin=/data/local-data/data/code/.TinyTeX/bin/x86_64-linux
if [[ -d "$tex_bin" ]]; then export PATH="$tex_bin:$PATH"; fi
mode=build
target=main
for arg in "$@"; do
  case "$arg" in
    main|supplementary) target="$arg" ;;
    build|watch) mode="$arg" ;;
    *) echo 'Usage: ./build.sh [main|supplementary] [build|watch]' >&2; exit 2 ;;
  esac
done
command -v latexmk >/dev/null || { echo 'latexmk not found; set PATH to your TeX Live bin directory.' >&2; exit 127; }
build_dir=$(mktemp -d)
trap 'rm -rf -- "$build_dir"' EXIT
args=(-xelatex -halt-on-error -interaction=nonstopmode -file-line-error -outdir="$build_dir")
if [[ "$mode" == watch ]]; then
  # latexmk invokes success_cmd after each successful recompilation.
  success_setting="\$success_cmd = 'cp -- \"$build_dir/$target.pdf\" \"./$target.pdf\"';"
  exec_status=0
  latexmk "${args[@]}" -pvc -view=none -e "$success_setting" "$target.tex" || exec_status=$?
  exit "$exec_status"
fi
if latexmk "${args[@]}" "$target.tex"; then
  cp -- "$build_dir/$target.pdf" "./$target.pdf"
else
  if [[ -f "$build_dir/$target.log" ]]; then
    cp -- "$build_dir/$target.log" "./last-failed-$target.log"
    echo "Failure log retained: last-failed-$target.log" >&2
  fi
  exit 1
fi
