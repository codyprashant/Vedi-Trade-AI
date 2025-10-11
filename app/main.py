import os
from datetime import datetime

import MetaTrader5 as mt5


def init_mt5() -> bool:
    """Initialize connection to MetaTrader 5 terminal.

    Optionally uses env vars `MT5_PATH`, `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`.
    Returns True if initialized and logged in (when creds provided), else False.
    """
    mt5_path = os.environ.get("MT5_PATH")
    login = os.environ.get("MT5_LOGIN")
    password = os.environ.get("MT5_PASSWORD")
    server = os.environ.get("MT5_SERVER")

    initialized = mt5.initialize(mt5_path) if mt5_path else mt5.initialize()
    if not initialized:
        print(f"MT5 initialize failed: {mt5.last_error()}")
        return False

    # If credentials are provided, try explicit login
    if login and password and server:
        if not mt5.login(int(login), password=password, server=server):
            print(f"MT5 login failed: {mt5.last_error()}")
            mt5.shutdown()
            return False

    return True


def get_btcusd_tick(symbol: str = "BTCUSD"):
    """Ensure symbol is selected and return current tick info for symbol."""
    if not mt5.symbol_select(symbol, True):
        print(f"Failed to select symbol: {symbol}")
        return None
    tick = mt5.symbol_info_tick(symbol)
    return tick


def main():
    if not init_mt5():
        return
    try:
        preferred = os.environ.get("MT5_SYMBOL", "XAUUSD")

        # Try preferred symbol; if not available, search installed symbols for closest match
        symbol = preferred
        if mt5.symbol_info(symbol) is None:
            candidates = [
                s.name
                for s in mt5.symbols_get()
                if ("XAUUSD" in s.name.upper() or "XAU" in s.name.upper())
            ]
            if candidates:
                symbol = candidates[0]
                print(f"Using discovered symbol: {symbol}")

        tick = get_btcusd_tick(symbol)
        if tick is None:
            print(f"No tick data for {symbol}. Check symbol name availability.")
            return
        ts = datetime.fromtimestamp(tick.time)
        print(
            f"{symbol} @ {ts} — bid: {tick.bid}, ask: {tick.ask}, last: {getattr(tick, 'last', None)}"
        )
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()