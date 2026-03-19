from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np


DEFAULT_SCENE_DIR_NAME = "data/put_bread/"
SECTION_WIDTH = 80


def print_banner(title: str) -> None:
    print("\n" + "=" * SECTION_WIDTH)
    print(title)
    print("=" * SECTION_WIDTH)


def print_section(title: str) -> None:
    print(f"\n{title}:")


def print_kv(label: str, value: object, indent: int = 2) -> None:
    print(f"{' ' * indent}{label}: {value}")


def strip_matching_quotes(raw_value: str) -> str:
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1].strip()
    return value


def prompt_with_default(message: str, default: Optional[str] = None) -> str:
    prompt = f"{message} ({default}): " if default else f"{message}: "
    response = strip_matching_quotes(input(prompt))
    if response:
        return response
    if default is not None:
        return default
    raise ValueError(f"No value provided for {message}.")


def resolve_scene_dir(scene_dir: Optional[str]) -> Path:
    raw_value = scene_dir or prompt_with_default("Scene directory", DEFAULT_SCENE_DIR_NAME)
    return Path(raw_value).expanduser().resolve()


def resolve_scene_path(scene_dir: Path, raw_value: str) -> Path:
    path = Path(strip_matching_quotes(raw_value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (scene_dir / path).resolve()


def prompt_scene_path(
    scene_dir: Path,
    label: str,
    default_name: str,
    override: Optional[str] = None,
) -> Path:
    raw_value = override or prompt_with_default(label, default_name)
    return resolve_scene_path(scene_dir, raw_value)


def choose_option(
    label: str,
    options: Iterable[str],
    default: str,
    override: Optional[str] = None,
) -> str:
    option_list = tuple(options)
    if default not in option_list:
        options_str = ", ".join(option_list)
        raise ValueError(f"Default for {label} must be one of: {options_str}")

    default_index = option_list.index(default) + 1

    if override is not None:
        selected = strip_matching_quotes(override).strip().lower()
        if selected in option_list:
            return selected
        if selected.isdigit():
            selected_index = int(selected)
            if 1 <= selected_index <= len(option_list):
                return option_list[selected_index - 1]
        options_str = ", ".join(f"[{i}] {option}" for i, option in enumerate(option_list, start=1))
        raise ValueError(f"{label} must be one of: {options_str}")

    options_text = " ".join(f"[{i}] {option}" for i, option in enumerate(option_list, start=1))
    while True:
        raw_value = strip_matching_quotes(
            input(f"{label} {options_text} (default: [{default_index}] {default}): ")
        )
        if not raw_value:
            return default
        if raw_value.isdigit():
            selected_index = int(raw_value)
            if 1 <= selected_index <= len(option_list):
                return option_list[selected_index - 1]
        print(f"Please enter a number between 1 and {len(option_list)}.")


def ensure_existing_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def parse_float_list(raw_value: str, expected_length: int, label: str) -> np.ndarray:
    parsed = np.fromstring(strip_matching_quotes(raw_value), sep=",", dtype=np.float64)
    if parsed.size != expected_length:
        raise ValueError(f"{label} expected {expected_length} comma-separated values.")
    return parsed


def load_float_list_file(path: Path, expected_length: int, label: str) -> np.ndarray:
    raw_value = path.read_text(encoding="utf-8").replace("\n", ",").strip()
    return parse_float_list(raw_value, expected_length, label)


def resolve_float_list(
    inline_value: Optional[str],
    scene_dir: Path,
    default_filename: str,
    expected_length: int,
    label: str,
    exists_hint: Optional[Callable[[np.ndarray], bool]] = None,
) -> np.ndarray:
    if inline_value:
        values = parse_float_list(inline_value, expected_length, label)
        if exists_hint is not None and not exists_hint(values):
            raise ValueError(f"{label} did not satisfy validation.")
        return values

    default_path = scene_dir / default_filename
    if default_path.is_file():
        values = load_float_list_file(default_path, expected_length, label)
        if exists_hint is not None and not exists_hint(values):
            raise ValueError(f"{label} in {default_path} did not satisfy validation.")
        return values

    while True:
        raw_value = prompt_with_default(label, None)
        values = parse_float_list(raw_value, expected_length, label)
        if exists_hint is None or exists_hint(values):
            return values
        print(f"{label} did not satisfy validation.")
