from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PATH_PARTS = {
    "AI审查的工程细节.md",
    "CONTEXT_PACK.md",
    "IMPLEMENTATION_LOG.md",
    "人工审查记录.docx",
    ".~人工审查记录.docx",
    "人工审查文档.md",
    "CLAUDE.md",
}
SECRET_PATTERNS = {
    "Anthropic key": re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    "OpenAI-style key": re.compile(r"sk-[A-Za-z0-9]{20,}"),
    "GitHub token": re.compile(r"gh[opsu]_[A-Za-z0-9]{20,}"),
    "Private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
TEXT_SUFFIXES = {
    ".css",
    ".example",
    ".html",
    ".js",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="扫描公开仓库中的密钥模式和内部文件")
    parser.add_argument("--worktree-only", action="store_true")
    return parser.parse_args()


def git_lines(*args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def worktree_files() -> list[Path]:
    tracked = git_lines("ls-files")
    untracked = git_lines("ls-files", "--others", "--exclude-standard")
    return [PROJECT_ROOT / name for name in dict.fromkeys([*tracked, *untracked])]


def scan_names(names: list[str], scope: str) -> list[str]:
    violations = []
    for name in names:
        if any(part in FORBIDDEN_PATH_PARTS for part in Path(name).parts):
            violations.append(f"{scope} 包含内部文件：{name}")
    return violations


def scan_text(label: str, content: str) -> list[str]:
    violations = []
    for pattern_name, pattern in SECRET_PATTERNS.items():
        if pattern.search(content):
            violations.append(f"{label} 命中疑似 {pattern_name}")
    return violations


def scan_worktree() -> list[str]:
    files = worktree_files()
    relative_names = [str(path.relative_to(PROJECT_ROOT)) for path in files]
    violations = scan_names(relative_names, "工作树")
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        violations.extend(scan_text(str(path.relative_to(PROJECT_ROOT)), content))
    return violations


def scan_history() -> list[str]:
    commits = git_lines("rev-list", "--all")
    violations = []
    for commit in commits:
        names = git_lines("ls-tree", "-r", "--name-only", commit)
        violations.extend(scan_names(names, f"提交 {commit[:12]}"))
        for name in names:
            if Path(name).suffix.lower() not in TEXT_SUFFIXES:
                continue
            result = subprocess.run(
                ["git", "show", f"{commit}:{name}"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
                errors="ignore",
            )
            violations.extend(scan_text(f"提交 {commit[:12]}:{name}", result.stdout))
    return violations


def main() -> None:
    args = parse_args()
    violations = scan_worktree()
    if not args.worktree_only:
        violations.extend(scan_history())
    if violations:
        raise SystemExit("公开仓库扫描失败：\n" + "\n".join(sorted(set(violations))))
    scope = "工作树" if args.worktree_only else "工作树和全部 Git 历史"
    print(f"公开仓库{scope}扫描通过")


if __name__ == "__main__":
    main()
