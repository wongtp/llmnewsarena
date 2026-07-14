"""Domain models shared across the pipeline."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class NewsItem:
    id: str
    title: str
    body: str
    source: Optional[str]
    link: Optional[str]
    time_ms: int
    received_ms: int
    icon: Optional[str] = None
    image: Optional[str] = None
    is_reply: bool = False
    is_retweet: bool = False
    is_quote: bool = False
    author_id: Optional[str] = None   # Tree's `twitterId` is the AUTHOR account id, NOT the tweet id
    coin_hint: Optional[str] = None          # Tree's auto-detected primary coin (crypto only)
    symbol_hints: list[str] = field(default_factory=list)  # from Tree `suggestions`
    raw: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return f"{self.title}\n{self.body}".strip()

    @property
    def age_seconds(self) -> float:
        return max(0.0, (now_ms() - self.time_ms) / 1000.0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw", None)
        return d

    @classmethod
    def from_telegram(cls, *, channel: str, channel_title: str, msg_id: int, text: str,
                      date_ms: int, chat_id) -> Optional["NewsItem"]:
        """Build a NewsItem from a Telegram channel post."""
        if not text or not text.strip():
            return None
        link = f"https://t.me/{channel}/{msg_id}" if channel else None
        return cls(
            id=f"tg:{channel or chat_id}:{msg_id}",
            title=channel_title or channel or "Telegram",
            body=text.strip(),
            source=f"Telegram:{channel}" if channel else "Telegram",
            link=link,
            time_ms=date_ms,
            received_ms=now_ms(),
            author_id=str(chat_id),
        )

    @classmethod
    def from_tree(cls, msg: dict) -> Optional["NewsItem"]:
        """Parse a Tree of Alpha websocket message into a NewsItem.

        Returns None for control/heartbeat frames that aren't news items.
        Schema: docs.treeofalpha.com/websockets/response
        """
        if not isinstance(msg, dict):
            return None
        nid = msg.get("_id") or msg.get("id")
        # A news item must have an id and some body/title.
        if not nid or not (msg.get("body") or msg.get("title")):
            return None
        info = msg.get("info") or {}
        suggestions = msg.get("suggestions") or []
        symbol_hints: list[str] = []
        for s in suggestions:
            coin = s.get("coin") if isinstance(s, dict) else None
            if coin:
                symbol_hints.append(coin)
        return cls(
            id=str(nid),
            title=msg.get("title") or "",
            body=msg.get("body") or "",
            source=msg.get("source"),
            link=msg.get("link") or msg.get("url"),
            time_ms=int(msg.get("time") or now_ms()),
            received_ms=int(msg.get("rt") or now_ms()),
            icon=msg.get("icon"),
            image=msg.get("image"),
            is_reply=bool(info.get("isReply")),
            is_retweet=bool(info.get("isRetweet")),
            is_quote=bool(info.get("isQuote")),
            author_id=str(info.get("twitterId")) if info.get("twitterId") else None,
            coin_hint=msg.get("coin"),
            symbol_hints=symbol_hints,
            raw=msg,
        )


@dataclass
class Analysis:
    news_id: str
    ticker: Optional[str]
    asset_class: str        # equity | index | commodity | crypto | none
    direction: str          # long | short | none
    confidence: float       # 0..1
    time_sensitivity: str   # immediate | hours | days | none
    is_stale: bool
    rationale: str
    model: str
    # How the ticker relates to the news: "direct" = the news is about this asset / its issuer
    # (incl. company-name-only headlines, e.g. "SpaceX" -> SPCX); "derived" = a 2nd-order
    # inference (e.g. "Alphabet capex up" -> NVDA). Defaults to "derived" so a missing value
    # (old backtest cache, or a non-strict model that omits it) falls back to the regex
    # direct-mention check and reproduces prior behavior.
    subject_relation: str = "derived"
    # ALL tradable symbols the news materially affects ([{ticker, asset_class}, ...],
    # primary first) — display-only (UI ticker boxes); the trade uses `ticker` above.
    related_tickers: list[dict] = field(default_factory=list)
    # Observability (shown next to the analysis in the UI): wall-clock time to produce
    # this analysis (triage + model call + any escalation; excludes queue wait) and the
    # estimated $ cost of the API calls behind it (see analysis/pricing.py).
    latency_ms: int = 0
    cost_usd: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Market:
    name: str          # order/coin name used by the SDK, e.g. "xyz:MRVL" or "BTC"
    symbol: str        # bare symbol, e.g. "MRVL", "BTC"
    dex: str           # "xyz" for trade.xyz equities, "" for HL crypto perps
    asset_class: str   # equity | index | commodity | crypto
    sz_decimals: int
    max_leverage: int
    day_volume_usd: float = 0.0   # 24h notional volume (liquidity proxy; crypto only)
    first_seen_ms: float = 0.0    # when this market first appeared in our universe (0 = baseline/old)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Decision:
    news_id: str
    action: str             # enter | reject
    reason: str
    market: Optional[Market] = None
    side: Optional[str] = None   # long | short
    notional_usd: float = 0.0
    size: float = 0.0
    leverage: int = 0
    entry_px: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    trail_pct: float = 0.0
    time_exit_seconds: int = 0
    confidence: float = 0.0
    model: str = ""         # arena: analyzer model that produced this decision (lane partition)

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.market:
            d["market"] = self.market.to_dict()
        return d


@dataclass
class Position:
    id: str
    news_id: str
    market: str         # order/coin name
    symbol: str
    dex: str
    side: str           # long | short
    size: float
    entry_px: float
    stop_loss: float
    take_profit: float
    leverage: int
    notional_usd: float
    opened_ms: int
    time_exit_ms: int
    dry_run: bool
    trail_pct: float = 0.0
    peak_px: float = 0.0      # high-water mark for trailing (best price seen)
    stop_order_id: int = 0    # resting reduce-only exchange stop oid (0 = none; live only)
    status: str = "open"      # open | closed
    exit_px: float = 0.0
    exit_decision_px: float = 0.0  # mid that TRIGGERED the close, captured before the fill
                                   # overwrites exit_px (exit-slippage attribution; 0 = unknown)
    pnl_usd: float = 0.0      # price PnL net of taker fees (funding shown separately)
    partial_pnl_usd: float = 0.0  # realized by partial close fills so far; folded into
                                  # pnl_usd by the final _record_close
    funding_usd: float = 0.0  # funding paid while held (+ = cost). LIVE: exchange cumFunding
                              # sinceOpen; PAPER: accrued estimate from the hourly rate.
    exit_reason: str = ""
    closed_ms: int = 0
    link: Optional[str] = None
    confidence: float = 0.0
    rationale: str = ""
    news_title: str = ""
    news_source: str = ""
    model_id: str = ""        # arena: the analyzer model that produced this trade (lane partition)

    @property
    def is_long(self) -> bool:
        return self.side == "long"

    def to_dict(self) -> dict:
        return asdict(self)


def to_jsonable(obj: Any) -> Any:
    """Best-effort conversion of domain objects to JSON-serializable structures."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj
