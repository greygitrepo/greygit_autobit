MAINNET_BASE = "https://api.bybit.com"
TESTNET_BASE = "https://api-testnet.bybit.com"


class V5:
    USER_QUERY_API = "/v5/user/query-api"  # GET (auth): query API key info
    SERVER_TIME = "/v5/market/time"        # GET (public): server time
    ACCOUNT_WALLET_BALANCE = "/v5/account/wallet-balance"  # GET (auth)
    POSITION_LIST = "/v5/position/list"                   # GET (auth)
    MARKET_KLINE = "/v5/market/kline"                     # GET (public)
    ORDER_CREATE = "/v5/order/create"                     # POST (auth)
    ORDER_CANCEL_ALL = "/v5/order/cancel-all"             # POST (auth)
    POSITION_SET_LEVERAGE = "/v5/position/set-leverage"   # POST (auth)
    POSITION_TRADING_STOP = "/v5/position/trading-stop"   # POST (auth)
    POSITION_SWITCH_ISOLATED = "/v5/position/switch-isolated" # POST (auth)
    MARKET_INSTRUMENTS_INFO = "/v5/market/instruments-info"   # GET (public)
    MARKET_TICKERS = "/v5/market/tickers"                     # GET (public)
