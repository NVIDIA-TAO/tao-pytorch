#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Update README supported-command documentation from setup.py entrypoints."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / "README.md"
SETUP_PATH = REPO_ROOT / "setup.py"
BEGIN_MARKER = "<!-- BEGIN GENERATED: supported-commands -->"
END_MARKER = "<!-- END GENERATED: supported-commands -->"
SPECIAL_CATEGORIES = {
    "bevfusion": "3D / point cloud",
    "sparse4d": "3D / point cloud",
}
CATEGORY_BY_PACKAGE_ROOT = {
    "cv": "Computer vision",
    "multimodal": "Multimodal",
    "pointcloud": "3D / point cloud",
    "sdg": "Synthetic data generation",
    "ssl": "Self-supervised learning",
}
CATEGORY_ORDER = [
    "Computer vision",
    "3D / point cloud",
    "Multimodal",
    "Self-supervised learning",
    "Synthetic data generation",
]


@dataclass(frozen=True)
class Command:
    """Installed TAO command and its backing package."""

    name: str
    module: str
    package: str
    category: str
    subtasks: tuple[str, ...]


def _literal_console_scripts(setup_path: Path) -> list[str]:
    """Read console_scripts literals from setup.py without importing it."""
    tree = ast.parse(setup_path.read_text(encoding="utf-8"), filename=str(setup_path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "setup"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "setuptools"
        ):
            continue
        for keyword in node.keywords:
            if keyword.arg != "entry_points" or not isinstance(keyword.value, ast.Dict):
                continue
            for key, value in zip(keyword.value.keys, keyword.value.values):
                if not (
                    isinstance(key, ast.Constant)
                    and key.value == "console_scripts"
                    and isinstance(value, ast.List)
                ):
                    continue
                return [
                    item.value
                    for item in value.elts
                    if isinstance(item, ast.Constant) and isinstance(item.value, str)
                ]
    raise RuntimeError(f"Could not find entry_points['console_scripts'] in {setup_path}")


def _package_from_entrypoint(module: str) -> str:
    """Convert an entrypoint module path to the owning model package."""
    marker = ".entrypoint."
    if marker not in module:
        raise ValueError(f"Entrypoint module does not contain {marker!r}: {module}")
    return module.split(marker, maxsplit=1)[0]


def _category(command_name: str, package: str) -> str:
    """Assign a README category from the command name and package root."""
    if command_name in SPECIAL_CATEGORIES:
        return SPECIAL_CATEGORIES[command_name]

    parts = package.split(".")
    if len(parts) < 2 or parts[0] != "nvidia_tao_pytorch":
        return "Other"
    return CATEGORY_BY_PACKAGE_ROOT.get(parts[1], "Other")


def _package_to_path(package: str) -> Path:
    """Convert a Python package name to a repo-local path."""
    return REPO_ROOT.joinpath(*package.split("."))


def _subtasks(package: str) -> tuple[str, ...]:
    """Discover subtasks from a package's scripts directory."""
    scripts_dir = _package_to_path(package) / "scripts"
    subtasks = []
    if scripts_dir.is_dir():
        subtasks = [
            path.stem
            for path in scripts_dir.glob("*.py")
            if path.stem != "__init__"
        ]

    # core.entrypoint.get_subtasks adds default_specs for every registered model.
    subtasks.append("default_specs")
    return tuple(sorted(set(subtasks)))


def discover_commands(setup_path: Path) -> list[Command]:
    """Discover installed model-family commands from setup.py."""
    commands = []
    for item in _literal_console_scripts(setup_path):
        command_name, target = item.split("=", maxsplit=1)
        module = target.split(":", maxsplit=1)[0]
        package = _package_from_entrypoint(module)
        commands.append(
            Command(
                name=command_name,
                module=module,
                package=package,
                category=_category(command_name, package),
                subtasks=_subtasks(package),
            )
        )
    return sorted(commands, key=lambda command: command.name)


def _code_list(items: list[str] | tuple[str, ...]) -> str:
    return ", ".join(f"`{item}`" for item in items)


def render_supported_commands(commands: list[Command]) -> str:
    """Render the generated README block."""
    lines = [
        BEGIN_MARKER,
        "",
        "The package installs one console command per model family. Each command discovers",
        "its available subtasks from that model's `scripts` package. The `default_specs`",
        "subtask is added by the shared TAO entrypoint for every registered model.",
        "",
        "Run this command after adding or removing a model entrypoint:",
        "",
        "```sh",
        "python tools/update_readme_supported_commands.py",
        "```",
        "",
        "| Domain | Commands |",
        "| :--- | :--- |",
    ]

    by_category: dict[str, list[str]] = {}
    for command in commands:
        by_category.setdefault(command.category, []).append(command.name)

    ordered_categories = [
        *[category for category in CATEGORY_ORDER if category in by_category],
        *sorted(category for category in by_category if category not in CATEGORY_ORDER),
    ]
    for category in ordered_categories:
        lines.append(f"| {category} | {_code_list(by_category[category])} |")

    lines.extend([
        "",
        "| Command | Package | Subtasks |",
        "| :--- | :--- | :--- |",
    ])
    for command in commands:
        package = command.package.removeprefix("nvidia_tao_pytorch.")
        lines.append(
            f"| `{command.name}` | `{package}` | {_code_list(command.subtasks)} |"
        )

    lines.extend([
        "",
        "For a specific model family, run the command with no subtask or with `-h` to see",
        "the supported subtasks:",
        "",
        "```sh",
        "depth_net -h",
        "depth_net train -h",
        "```",
        "",
        END_MARKER,
    ])
    return "\n".join(lines)


def replace_generated_block(readme_text: str, generated: str) -> str:
    """Replace the generated supported-command block in README text."""
    if BEGIN_MARKER not in readme_text or END_MARKER not in readme_text:
        raise RuntimeError(
            f"README must contain {BEGIN_MARKER!r} and {END_MARKER!r} markers"
        )
    prefix, rest = readme_text.split(BEGIN_MARKER, maxsplit=1)
    _, suffix = rest.split(END_MARKER, maxsplit=1)
    return prefix.rstrip() + "\n\n" + generated + suffix


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Regenerate the README supported-command section."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if README.md is not up to date.",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=README_PATH,
        help="Path to README.md.",
    )
    parser.add_argument(
        "--setup",
        type=Path,
        default=SETUP_PATH,
        help="Path to setup.py.",
    )
    return parser.parse_args()


def main() -> None:
    """Update or check the generated README section."""
    args = parse_args()
    commands = discover_commands(args.setup)
    generated = render_supported_commands(commands)
    readme_text = args.readme.read_text(encoding="utf-8")
    updated = replace_generated_block(readme_text, generated)

    if args.check:
        if updated != readme_text:
            raise SystemExit(
                "README.md is out of date. Run "
                "`python tools/update_readme_supported_commands.py`."
            )
        print("README.md supported-command section is up to date.")
        return

    args.readme.write_text(updated, encoding="utf-8")
    print(f"Updated {args.readme} with {len(commands)} commands.")


if __name__ == "__main__":
    main()
