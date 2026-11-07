# CVPR 2027 编译与编辑

在 H20：

```bash
cd /data/local-data/data/code/yunhengwang/kimodo-polocy/controlnet_v1.2/asset/paper/cvpr
./build.sh main                 # 生成 main.pdf
./build.sh supplementary        # 生成 supplementary.pdf
./build.sh main watch           # 监听主文相关文件，成功后更新 main.pdf
./build.sh supplementary watch  # 监听补充材料
```

Ctrl+C 停止监听。关闭 SSH 后仍需运行时，可在已有 tmux 会话中运行。每次脚本启动创建独立临时构建目录；本次 watch 生命周期内复用该目录，退出后清除。只有编译成功才覆盖目标 PDF，失败保留上一次成功的 PDF；单次构建失败额外保存 `last-failed-main.log` 或 `last-failed-supplementary.log`。不同目标可以分别监听，同一个目标建议只启动一个监听进程。

脚本仅在自身进程中将现有 `/data/local-data/data/code/.TinyTeX/bin/x86_64-linux` 加入 PATH，不修改 shell 配置或 TinyTeX。缺失的小依赖包已放在项目目录内。

主文入口 `main.tex`，正文 `sections/abstract.tex`、`introduction.tex`、`related_work.tex`、`method.tex`、`experiments.tex`、`conclusion.tex`；表格在 `tables/`；补充内容在 `sections/supplement.tex`。参考文献 `ref.bib`，图像 `image/`；新增图片也可放 `figures/` 并在正文通过 `\includegraphics{figures/文件名}` 引用。未被引用的图片不会影响 PDF。

保留原来的单次命令也可以：

```bash
(
  cd /data/local-data/data/code/yunhengwang/kimodo-polocy/controlnet_v1.2/asset/paper/cvpr || exit 1
  export PATH=/data/local-data/data/code/.TinyTeX/bin/x86_64-linux:$PATH
  build_dir=$(mktemp -d) || exit 1
  trap 'rm -rf -- "$build_dir"' EXIT
  latexmk -xelatex -halt-on-error -interaction=nonstopmode \
    -outdir="$build_dir" main.tex &&
    cp -- "$build_dir/main.pdf" ./main.pdf
)
```

通过 VS Code Remote SSH 打开该目录后，用 LaTeX Workshop 或远程 PDF 预览查看 `main.pdf`。更新的是 H20 文件，不会自动同步 Overleaf。

官方格式来源：https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines
原始官方 ZIP：`author-kit-CVPR2027-v1-latex.zip`。当前使用 review 匿名模式，主文末含 LLM Usage Statement；内容取舍和待作者核对项见 `REVISION_NOTES.md`。
