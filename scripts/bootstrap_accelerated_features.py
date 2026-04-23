from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


REPO_URL = "https://github.com/verlab/accelerated_features.git"
DEFAULT_REF = "main"


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def git_in(target: Path, *args: str) -> list[str]:
    return ["git", "-c", f"safe.directory={target.resolve().as_posix()}", "-C", str(target), *args]


def has_xfeat_modules(path: Path) -> bool:
    return (path / "modules" / "xfeat.py").exists() and (path / "modules" / "lighterglue.py").exists()


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone/update official verlab/accelerated_features for XFeat + LighterGlue.")
    parser.add_argument("--target", default="third_party/accelerated_features", help="Local clone directory.")
    parser.add_argument("--ref", default=DEFAULT_REF, help="Git branch/tag/SHA to checkout.")
    args = parser.parse_args()

    target = Path(args.target)
    ref = (args.ref or DEFAULT_REF).strip()
    if target.exists():
        if not (target / ".git").exists():
            raise SystemExit(f"{target} exists but is not a git clone.")
        run(git_in(target, "fetch", "--depth", "1", "origin", ref))
        if ref in {"main", "master"}:
            run(git_in(target, "checkout", ref))
            run(git_in(target, "pull", "--ff-only"))
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", REPO_URL, str(target)])
        if ref not in {"main", "master"}:
            run(["git", "-C", str(target), "fetch", "--depth", "1", "origin", ref])

    if ref not in {"main", "master"}:
        run(git_in(target, "checkout", ref))

    if not has_xfeat_modules(target):
        raise SystemExit(f"{target} does not contain official XFeat/LighterGlue modules after bootstrap.")
    print(f"Ready: {target}")


if __name__ == "__main__":
    main()
