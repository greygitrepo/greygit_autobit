Autobit: Bybit Futures Auto-Trading (Skeleton)

Overview

- Python 3.10.12, VS Code, Ubuntu 22.04.
- Modular, class-based structure. Initial implementation includes account connection to Bybit.
- Testnet/Mainnet selectable via environment.

Quick Start

1) Create and populate a `.env` or export environment variables (the app will auto-load `.env` if present, and VS Code launch uses it too):

   - `BYBIT_API_KEY` = your Bybit API key
   - `BYBIT_API_SECRET` = your Bybit API secret
   - `BYBIT_ENV` = `testnet` or `mainnet` (defaults to `testnet`)
   - `BYBIT_RECV_WINDOW` = optional, e.g. `5000`

2) Run the CLI to check account connection:

   - `PYTHONPATH=src python -m autobit check-connection`

   You should see either a success message with key info or a failure reason.

Project Structure

src/
- autobit/
  - __init__.py
  - __main__.py               (CLI entry)
  - config.py                 (config/env loader)
  - logger.py                 (logging setup)
  - engine.py                 (orchestration skeleton)
  - exchange/
    - __init__.py
    - bybit_client.py         (REST client; connection check implemented)
    - endpoints.py            (Bybit endpoints/constants)
  - strategy/
    - __init__.py
    - base.py                 (strategy skeleton)
  - risk/
    - __init__.py
    - base.py                 (risk skeleton)
  - trade/
    - __init__.py
    - executor.py             (order execution skeleton)
  - data/
    - __init__.py
    - market_data.py          (market data skeleton)

Dependencies

- Uses only Python standard library (urllib, hmac, hashlib, json, time) for REST calls.
- No external packages required to run the connection check.

Notes

- For production, consider adding robust retry/backoff, time sync tolerance, and async I/O.
- WebSocket streaming is not included yet; the skeleton leaves room for it.

VS Code Debugging

- A launch config is provided to run the module with `src` layout and `.env` file.
- Use the "Autobit: check-connection" configuration in the Run and Debug panel.
