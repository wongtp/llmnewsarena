"""System prompt + structured-output tool schema for the news analyzer."""
from __future__ import annotations

import re

from ..models import NewsItem

# CJK / fullwidth / kana / hangul — used to strip the Chinese translation that one of our
# aggregators appends, so the same wire from different feeds normalizes to identical text.
_CJK = re.compile(r"[　-鿿＀-￯゠-ヿ가-힯]")
_BOILERPLATE = {"link", "open source", "source", "↗ open source", "read more"}
# Feeds re-broadcast each other with channel-attribution prefixes ("AggrNews: ..."), and
# append separator + timestamp lines. Strip all of it so the core wire is what's analyzed.
_ATTRIB = re.compile(r"^\s*(aggrnews|aggrnewswire|bwenews|tree|trad[_ ]?fin)\s*:\s*", re.I)
_SEP = re.compile(r"^[\s\-—–_=•·.*~]+$")
_TS = re.compile(r"^\s*\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?\s*$")

# Model families with sampling params REMOVED (Opus 4.7+, Fable): sending temperature /
# top_p / top_k returns a 400. Older families (Sonnet, Haiku, Opus <= 4.6) still accept it.
_NO_SAMPLING = ("opus-4-7", "opus-4-8", "fable")


def supports_temperature(model: str) -> bool:
    m = (model or "").lower()
    return not any(t in m for t in _NO_SAMPLING)


def clean_news_text(text: str) -> str:
    """Normalize a headline/body for analysis so two aggregators' versions of the same wire
    are byte-identical: strip channel-attribution prefixes, appended translation lines,
    separator/timestamp lines, link/source boilerplate and URLs."""
    if not text:
        return ""
    kept = []
    for ln in re.split(r"[\r\n]+", text):
        s = _ATTRIB.sub("", ln.strip()).strip()    # drop a leading "<Feed>:" attribution
        if not s or s.lower() in _BOILERPLATE:
            continue
        if _SEP.match(s) or _TS.match(s):           # separator / timestamp lines the feeds append
            continue
        cjk = len(_CJK.findall(s))
        if cjk and cjk >= 0.25 * len(s):            # a mostly-CJK line is a translation -> drop it
            continue
        kept.append(s)
    cleaned = re.sub(r"https?://\S+", "", " ".join(kept))   # strip URLs
    cleaned = _CJK.sub("", cleaned)                          # strip any residual inline CJK
    return re.sub(r"\s+", " ", cleaned).strip()

INSTRUCTIONS = """You are a financial-news analyst for an automated trading bot that \
trades perpetual futures on Hyperliquid. You receive one breaking-news item or tweet \
and must decide whether it is a tradable catalyst.

Rules:
- Identify the SINGLE most relevant, directly-affected asset and its ticker. The ticker \
MUST be one of the tradable symbols listed below. If none of the listed symbols is the \
clear subject, return ticker="" and direction="none".
- direction: "long" if the news is clearly bullish for that asset, "short" if clearly \
bearish, "none" if neutral/ambiguous/not actionable.

- related_tickers: SEPARATELY from the single trade ticker above, list the primary ticker \
plus AT MOST ONE secondary listed symbol — the single most-affected other name (the other \
party to a deal, the most exposed direct competitor, a key supplier/customer, or the \
underlying asset of a fund/treasury). Never more than 2 entries total. This list is for \
DISPLAY ONLY and does not affect the trade — still fill it when the trade call is "none". \
Only symbols from the tradable lists below; [] when none qualify. Do not add a secondary \
name unless it is materially affected.

- SIGNIFICANCE OVER MAGNITUDE: judge the SIGNALING significance, not just the dollar/ \
quantity size. A small action that breaks a long-established pattern or reveals a shift \
by a pivotal actor can be a high-impact catalyst even when the nominal amount is tiny — \
e.g. a company's FIRST sale of an asset in years, a founder/CEO reversing a famous \
stance, a long-time holder changing direction, a regulator's first approval or denial. \
Such regime-change signals often move markets far beyond the headline number. \
Conversely, large numbers that are routine or already expected are NOT catalysts.

- BE SELECTIVE AND DECISIVE: most news is NOT tradable. Routine updates, partnerships, \
integrations, incremental metrics, already-known or priced-in information, opinions, \
promotions, and generic macro should score LOW (<0.4) or "none". Do not park borderline \
items at ~0.7 — either it is a genuine, fresh, market-moving catalyst (then score it \
honestly high) or it is not (then score it low). Avoid clustering scores near the threshold.

- LOW-EDGE MACRO NOISE: routine commodity, index, or FX price-move headlines (e.g. "oil \
rises", "Brent slips 2%", "gold dips", "stocks open higher", "yields tick up") are \
mean-reverting macro chatter with no durable trading edge — score them LOW or "none". Treat \
commodities/indices as actionable ONLY on a genuine structural catalyst (e.g. an OPEC supply \
decision, a major supply disruption or sanctions, a surprise inflation/jobs print, or a \
geopolitical shock).

- confidence (0..1): probability that opening this position immediately is positive \
expected value. Reserve >0.85 for unambiguous, market-moving, primary-source news \
(confirmed M&A, blowout/awful earnings, major regulatory action, confirmed listing, or a \
clear regime-change signal per the rule above).
- is_stale: true if the news is old (see AGE_SECONDS), a rumor, already widely known / \
priced in, a recycled/retweeted story, or speculation rather than confirmed fact.
- time_sensitivity: how long the edge PERSISTS — this sets the holding period, so be careful:
  * "immediate" = a one-off data point the market fully prices within minutes and then \
mean-reverts (a single liquidation cascade, a tiny in-line beat). Use SPARINGLY.
  * "hours" = develops over one trading session (a product launch, one analyst action).
  * "days" = a STRUCTURAL / REGIME-CHANGE catalyst that trends across multiple sessions: \
a pivotal actor reversing a long-standing stance, a first-in-years action, major M&A or a \
big earnings surprise, regulatory regime shifts, or sector-wide rotations — anything that \
changes the multi-week narrative for the asset.
  When torn between "immediate" and "days" for a genuine market-moving catalyst, choose the \
LONGER horizon: exiting a real trend after minutes forfeits most of the move. Example: a \
long-time accumulator selling an asset for the FIRST time in years is "days" (regime change), \
NOT "immediate".
- RESULTS ARE NOT CATALYSTS: a headline that merely REPORTS a move that already happened \
("X surges 19%", "BTC drops 5.5%", "shares slide after earnings", "stock jumps on deal") is \
a RESULT — the move is already in the price, the edge is gone. Score these "none". Only the \
ORIGINAL causal event (the earnings, the filing, the signed deal) is tradable, and only fresh.

- FOLLOW-UPS / SAME EVENT: if the item recaps, explains, or is the downstream consequence of \
an event that already occurred earlier, set is_stale=true. We act ONLY on the first, fresh \
appearance of a catalyst — never re-trade the same story or its aftermath.

- WARS / ONGOING CONFLICTS: news about an ongoing, back-and-forth military conflict (strikes, \
threats, troop moves, Strait of Hormuz tension, etc.) is mostly noise with no durable edge — \
score it LOW. Treat it as actionable ONLY on a genuine regime-defining development: a signed \
ceasefire / peace deal, assassination of a head of state, a full blockade, or an unambiguous \
major escalation. See the regime context for the current state of any conflict.

- CROSS-LISTINGS: an exchange listing a tradfi STOCK as a crypto perp/derivative (e.g. \
"Binance Futures lists <stock>") has ~no effect on the underlying — score "none". (This is \
different from a company's own primary exchange listing/IPO.)

- IPOs / PRE-IPO: fresh IPO debuts and pre-IPO / newly-public names are speculative and \
illiquid; the opening pop is largely priced and noisy. Be conservative — usually "none" unless \
there is a distinct major catalyst beyond the listing/IPO itself. Do NOT infer "long" merely \
because a stock IPO'd or opened above its IPO price (that pop is a RESULT). Judge direction by \
the move versus the relevant reference: an IPO/debut that opens or trades BELOW recent \
indications/expectations is bearish, not bullish.

- EXPECTED / ALREADY-EXECUTED ACTIONS: actions the market already expects or that already \
happened are priced in and do NOT move the asset — score "none"/low. Examples: a company's \
routine/announced-in-advance treasury purchases (e.g. a regularly scheduled BTC buy that has \
already occurred), or a disclosure of holdings ACCUMULATED long ago (e.g. a firm revealing a \
large BTC stake bought years ago). Only a SURPRISE change in behavior is a catalyst (a \
first-ever sale, an unexpected halt, a shock-sized or unscheduled move).

- CORPORATE BTC TREASURY MOVES: a company BUYING bitcoin for its treasury — even a large, \
fresh, multi-billion-dollar purchase (e.g. MicroStrategy/Strategy) — or a company DISCLOSING / \
revealing BTC it already holds (e.g. in an IPO/S-1 filing) is NOT a tradable catalyst for BTC. \
The buyer's demand is already in the market and a disclosure creates no new demand, so these \
have little durable price effect. Score BTC "none"/low REGARDLESS of dollar size or how "fresh" \
the headline looks. The ONLY corporate-treasury BTC catalyst is a SURPRISE SELL by a PIVOTAL, \
market-defining mega-holder reversing a famous long-held stance (e.g. MicroStrategy/Strategy/ \
Saylor, who had never sold and hold a meaningful share of supply). It must be that caliber of \
actor reversing course — NOT a minor/niche "digital-asset-treasury" (DAT) company, a small fund, \
or merely a "prominent advocate", and NOT a routine FILING disclosing a past sale. A small sale \
(e.g. tens of millions), or a sale by any non-mega holder, does NOT move BTC — score "none"/low. \
Weigh the SELLER'S true market weight and the sale's materiality, not just that it is a "first" \
sale. This does NOT apply to genuine BTC catalysts (major regulation, ETF/macro flows).

- subject_relation: "direct" if the news is primarily ABOUT the asset you picked or its \
issuer/company — INCLUDING when the headline names the COMPANY rather than the ticker (e.g. \
"SpaceX" for SPCX, "Alphabet" for GOOGL). "derived" if you INFERRED this ticker as a 2nd-order \
beneficiary/supplier/counterparty of news about a DIFFERENT entity (e.g. "Alphabet raises \
capex" -> NVDA). When unsure, choose "derived".

- Only act on the asset the news is ABOUT, not assets merely mentioned in passing.
- Prefer returning direction="none" when uncertain. A missed trade is cheaper than a bad one.

Return your answer ONLY by calling the submit_analysis tool."""


ANALYSIS_TOOL = {
    "name": "submit_analysis",
    "description": "Return the structured trading analysis of the news item.",
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Exact ticker symbol from the tradable lists (e.g. MRVL, BTC). Empty string if none applies.",
            },
            "asset_class": {
                "type": "string",
                "enum": ["equity", "index", "commodity", "crypto", "none"],
            },
            "direction": {"type": "string", "enum": ["long", "short", "none"]},
            "confidence": {
                "type": "number",
                "description": "0..1 confidence that acting now is positive expected value.",
            },
            "time_sensitivity": {
                "type": "string",
                "enum": ["immediate", "hours", "days", "none"],
            },
            "is_stale": {
                "type": "boolean",
                "description": "true if old / already priced-in / rumor / recycled.",
            },
            "rationale": {"type": "string", "description": "<=240 chars explaining the call."},
            "related_tickers": {
                "type": "array",
                "description": "The primary ticker plus AT MOST ONE secondary tradable symbol "
                "materially affected by this news (primary first), for display only — the "
                "trade uses the fields above. Empty array if none. At most 2 entries.",
                "items": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "asset_class": {
                            "type": "string",
                            "enum": ["equity", "index", "commodity", "crypto"],
                        },
                    },
                    "required": ["ticker", "asset_class"],
                    "additionalProperties": False,
                },
            },
            "subject_relation": {
                "type": "string",
                "enum": ["direct", "derived"],
                "description": "\"direct\" = the news is primarily ABOUT this asset or its "
                "issuer/company, even when it names the company rather than the ticker (e.g. "
                "\"SpaceX\" for SPCX, \"Alphabet\" for GOOGL). \"derived\" = you INFERRED this "
                "ticker as a 2nd-order beneficiary/supplier/counterparty of news about a "
                "DIFFERENT entity (e.g. \"Alphabet raises capex\" -> NVDA). When unsure, \"derived\".",
            },
        },
        "required": [
            "ticker", "asset_class", "direction", "confidence",
            "time_sensitivity", "is_stale", "rationale", "related_tickers",
            "subject_relation",
        ],
        # Structured-outputs compatible: with strict_tool enabled the API guarantees the
        # arguments match this schema exactly (valid enums, no missing/extra fields).
        "additionalProperties": False,
    },
}


CONFIRM_INSTRUCTIONS = """You are the risk-side SKEPTIC for an automated news-trading bot \
on Hyperliquid. A first-pass analyst has proposed a trade off a breaking headline and it \
has passed every mechanical risk check; thousands of dollars are about to be deployed. \
Your job is to try to KILL the trade before it happens.

Argue the strongest case that this trade is WRONG before you score it. Attack it from \
every angle:
- PRICED IN: the market tape is provided — if the asset already repriced (on this news or \
into it), the edge is gone. A headline reporting a move that already happened is a RESULT, \
not a catalyst.
- WRONG TICKER / WRONG ASSET: is the chosen symbol really the directly-affected asset, or \
a 2nd-order inference dressed up as a direct call?
- ROUTINE DRESSED AS NEWS: expected/scheduled actions, recycled stories, follow-ups, \
opinions, or large-but-routine numbers are not catalysts.
- REGIME MISMATCH: does the proposed direction fight the prevailing regime/climate without \
a strong enough reason?
- CROWDED / CHASED: the measured pre-news move is provided — entering after the crowd on a \
short-horizon trade is buying someone else's exit.

Then give your INDEPENDENT verdict:
- agree_direction: false if the trade direction itself is wrong or untradeable (this VETOES \
the trade). A veto that prevents a bad trade is a success, not a failure.
- confidence (0..1): YOUR independent probability that entering NOW is positive expected \
value — same definition and calibration as the first pass; do not anchor on the analyst's \
number. Genuine regime-change catalysts (a pivotal actor reversing a famous long-held \
stance, confirmed M&A, a major regulatory shift) deserve high confidence even after a pop — \
do not veto those for ordinary uncertainty. Mediocre, chased, or priced-in setups deserve \
low confidence even when the direction is defensible.
- risk: one sentence — the single strongest reason this trade could lose.

Return your verdict ONLY by calling the confirm_trade tool."""


CONFIRM_TOOL = {
    "name": "confirm_trade",
    "description": "Return the skeptic's independent verdict on the proposed trade.",
    "input_schema": {
        "type": "object",
        "properties": {
            "agree_direction": {
                "type": "boolean",
                "description": "false = veto: the proposed direction is wrong or untradeable.",
            },
            "confidence": {
                "type": "number",
                "description": "0..1 independent probability that entering NOW is +EV.",
            },
            "risk": {
                "type": "string",
                "description": "<=200 chars: the strongest reason this trade could lose.",
            },
        },
        "required": ["agree_direction", "confidence", "risk"],
        "additionalProperties": False,
    },
}


def build_confirm_user_text(item: NewsItem, analysis, market_context: str = "",
                            pre_move_pct: float | None = None,
                            age_seconds: float | None = None) -> str:
    """The skeptic's evidence: the headline, the first-pass verdict, the tape, and the
    deterministic already-moved measurement."""
    raw = item.body if (item.source or "").startswith("Telegram") else item.text
    age = item.age_seconds if age_seconds is None else age_seconds
    lines = [
        f"NEWS: {clean_news_text(raw)}",
        f"AGE_SECONDS: {age:.0f}",
        "",
        ("FIRST-PASS ANALYSIS (do not anchor on its confidence): "
         f"ticker={analysis.ticker} direction={analysis.direction} "
         f"confidence={analysis.confidence:.2f} time_sensitivity={analysis.time_sensitivity}"),
        f"RATIONALE: {analysis.rationale}",
    ]
    if market_context.strip():
        lines += ["", market_context.strip()]
    if pre_move_pct is not None:
        lines += ["", (f"PRE-NEWS MOVE: the market has already moved {pre_move_pct:+.2%} "
                       f"IN the proposed direction between just before this news and the "
                       f"would-be entry (positive = the repricing happened without us).")]
    return "\n".join(lines)


def _cache_control(ttl: str) -> dict:
    """cache_control for a stable prompt block. "5m" (the API default, written at 1.25x) is
    emitted without a ttl field; "1h" (written at 2x, read at 0.1x) keeps the prefix warm
    across the long gaps between news items, which is when 5m caches just expire unread."""
    if ttl and ttl != "5m":
        return {"type": "ephemeral", "ttl": ttl}
    return {"type": "ephemeral"}


def build_system_blocks(equity_symbols: list[str], crypto_symbols: list[str],
                        regime_context: str = "", recent_catalysts: str = "",
                        cache_ttl: str = "1h", exemplars: str = "",
                        context_bridge: str = "") -> list[dict]:
    """System prompt as content blocks. The stable prefix (instructions, exemplars, bridge,
    regime, universe — ordered most- to least-stable so a regime refresh re-pays only its own
    block) is cached; the volatile recent-catalysts memory is appended UNCACHED last so it
    never invalidates the cached prefix."""
    universe_text = (
        "TRADABLE EQUITY / INDEX / COMMODITY SYMBOLS (trade.xyz):\n"
        + (", ".join(equity_symbols) if equity_symbols else "(none enabled)")
        + "\n\nTRADABLE CRYPTO PERP SYMBOLS (Hyperliquid):\n"
        + (", ".join(crypto_symbols) if crypto_symbols else "(none enabled)")
    )
    cc = _cache_control(cache_ttl)
    blocks = [{"type": "text", "text": INSTRUCTIONS, "cache_control": dict(cc)}]
    if exemplars.strip():
        blocks.append({
            "type": "text",
            "text": "WORKED EXAMPLES (hand-curated calibration anchors from past trades — "
                    "match the JUDGMENT shown here, not the specific tickers/phrasings):\n"
                    + exemplars.strip(),
            "cache_control": dict(cc),
        })
    if context_bridge.strip():
        blocks.append({
            "type": "text",
            "text": "BACKGROUND SINCE YOUR TRAINING DATA ENDS (dated reference facts bridging "
                    "your knowledge cutoff to the present — use them to ground what is genuinely "
                    "NEW vs old/priced-in and to keep reasoning factually current; this is "
                    "background, not a trade signal, and the REGIME block below is more current "
                    "where they overlap):\n" + context_bridge.strip(),
            "cache_control": dict(cc),
        })
    if regime_context.strip():
        blocks.append({
            "type": "text",
            "text": "CURRENT MARKET / GEOPOLITICAL REGIME (backdrop for judging each headline; "
                    "weight news against this prevailing context):\n" + regime_context.strip(),
            "cache_control": dict(cc),
        })
    blocks.append({"type": "text", "text": universe_text, "cache_control": dict(cc)})
    if recent_catalysts.strip():
        blocks.append({
            "type": "text",
            "text": "RECENT CATALYSTS ALREADY TRADED (do NOT re-trade the same event or its "
                    "aftermath — if this headline is the same event resurfacing, a follow-up, or "
                    "a downstream consequence, set is_stale=true):\n" + recent_catalysts.strip(),
        })
    return blocks


def build_user_text(item: NewsItem, age_seconds: float | None = None) -> str:
    age = item.age_seconds if age_seconds is None else age_seconds
    # Our feeds are interchangeable aggregators of the same wire, so the source/channel name
    # must NOT sway the call — omit it. For Telegram the "title" is just the channel name, so
    # the news is in the body; for other feeds use title+body. Normalize so the same wire from
    # two aggregators yields identical input (hence identical confidence).
    raw = item.body if (item.source or "").startswith("Telegram") else item.text
    lines = [
        f"NEWS: {clean_news_text(raw)}",
        f"AGE_SECONDS: {age:.0f}",
    ]
    if item.coin_hint or item.symbol_hints:
        hints = ", ".join(filter(None, [item.coin_hint, *item.symbol_hints]))
        lines.append(f"AUTO_DETECTED_CRYPTO_HINTS: {hints}")
    if item.is_retweet:
        lines.append("NOTE: this is a retweet")
    if item.is_reply:
        lines.append("NOTE: this is a reply")
    return "\n".join(lines)
