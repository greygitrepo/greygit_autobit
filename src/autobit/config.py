from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class AppConfig:
    bybit_api_key: str
    bybit_api_secret: str
    bybit_env: str = "testnet"  # or "mainnet"
    bybit_recv_window: int = 5000


def _get_env(name: str, default: str | None = None) -> str:
    val = os.getenv(name, default)
    if val is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


def _parse_dotenv(lines: Iterable[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            env[key] = value
    return env


def _load_dotenv_if_present() -> None:
    # Search common locations for a .env file and load values not already set
    candidates: list[Path] = []
    try:
        candidates.append(Path(os.getcwd()) / ".env")
    except Exception:
        pass
    p = Path(__file__).resolve()
    for parent in list(p.parents)[:6]:  # up to repo root
        candidates.append(parent / ".env")

    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            try:
                content = path.read_text(encoding="utf-8").splitlines()
                env = _parse_dotenv(content)
                for k, v in env.items():
                    if k not in os.environ:
                        os.environ[k] = v
                # Stop at the first .env found to avoid surprising overrides
                break
            except Exception:
                # Silent fallback; config will error if required vars missing
                pass


def load_config() -> AppConfig:
    # Try loading .env before reading variables
    _load_dotenv_if_present()
    env = os.getenv("BYBIT_ENV", "testnet").strip().lower()
    if env not in {"testnet", "mainnet"}:
        raise RuntimeError("BYBIT_ENV must be 'testnet' or 'mainnet'")

    recv_window_str = os.getenv("BYBIT_RECV_WINDOW", "5000")
    try:
        recv_window = int(recv_window_str)
    except ValueError as e:
        raise RuntimeError("BYBIT_RECV_WINDOW must be an integer") from e

    return AppConfig(
        bybit_api_key=_get_env("BYBIT_API_KEY"),
        bybit_api_secret=_get_env("BYBIT_API_SECRET"),
        bybit_env=env,
        bybit_recv_window=recv_window,
    )
