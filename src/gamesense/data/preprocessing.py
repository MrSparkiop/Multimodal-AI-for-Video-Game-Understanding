"""Cleaning and normalisation of the raw Steam metadata."""

from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import CONFIG, GENRES, DatasetConfig, label_column, label_columns
from ..utils import get_logger

__all__ = [
    "GENRE_SYNONYMS",
    "GENRE_LEAK_TERMS",
    "ADULT_GENRES",
    "ADULT_KEYWORDS",
    "ADULT_KEYWORDS_EXACT",
    "ADULT_NOTE_PATTERNS",
    "adult_content_mask",
    "TEXT_COLUMNS",
    "CleaningReport",
    "normalize_genre",
    "normalize_genres",
    "clean_description",
    "title_variants",
    "remove_title_mentions",
    "mask_genre_terms",
    "word_count",
    "ascii_ratio",
    "encode_labels",
    "decode_labels",
    "label_matrix",
    "is_eligible",
    "add_text_variants",
    "clean_games",
    "positive_counts",
    "class_weights",
    "image_average_hash",
]

LOGGER = get_logger("gamesense.data.preprocessing")

#: Raw Steam genre string (lower-cased, punctuation-normalised) -> canonical
#: label.  Anything not listed here is simply not a prediction target.
GENRE_SYNONYMS: dict[str, str] = {
    "action": "Action",
    "adventure": "Adventure",
    "casual": "Casual",
    "rpg": "RPG",
    "role playing": "RPG",
    "role-playing": "RPG",
    "roleplaying": "RPG",
    "racing": "Racing",
    "race": "Racing",
    "simulation": "Simulation",
    "simulations": "Simulation",
    "sim": "Simulation",
    "sports": "Sports",
    "sport": "Sports",
    "strategy": "Strategy",
}

#: Explicit genre vocabulary used by the *optional* keyword-masking ablation.
#: Mapping is canonical label -> surface forms that give the label away.
GENRE_LEAK_TERMS: dict[str, tuple[str, ...]] = {
    "Action": ("action", "action-packed", "beat em up", "beat 'em up", "hack and slash",
               "hack-and-slash", "shooter", "fps", "first-person shooter", "run and gun"),
    "Adventure": ("adventure", "point-and-click", "point and click", "visual novel",
                  "walking simulator", "metroidvania"),
    "Casual": ("casual", "hyper-casual", "idle game", "clicker", "match-3", "match 3",
               "hidden object", "jigsaw"),
    "RPG": ("rpg", "rpgs", "role-playing", "role playing", "roleplaying", "jrpg", "crpg",
            "arpg", "roguelike", "roguelite", "dungeon crawler"),
    "Racing": ("racing", "race", "races", "racer", "kart", "drift", "rally", "motorsport",
               "time trial"),
    "Simulation": ("simulation", "simulator", "sim", "tycoon", "management game",
                   "city builder", "farming sim"),
    "Sports": ("sports", "sport", "football", "soccer", "basketball", "baseball", "golf",
               "tennis", "hockey", "cricket", "bowling", "boxing"),
    "Strategy": ("strategy", "strategic", "rts", "real-time strategy", "turn-based",
                 "turn based", "4x", "tower defense", "tower defence", "grand strategy",
                 "tactics", "tactical"),
}

#: Steam genre strings that mark sexual content.
ADULT_GENRES: frozenset[str] = frozenset({"nudity", "sexual content"})

# : Terms that identify sexually explicit games in a title or short description.
ADULT_KEYWORDS: tuple[str, ...] = (
    # Stems: matched at a word start, suffixes allowed ("nude" -> "nudes",
    # "masturbat" -> "masturbating", "prostitut" -> "prostitution").
    "hentai", "nudity", "nude", "naked", "sexy", "sexual", "sex scene", "erotic",
    "eroge", "porn", "xxx", "adult", "uncensored", "futanari", "yaoi", "ecchi",
    "lewd", "bdsm", "fetish", "milf", "ahegao", "netorare", "brothel",
    "stripper", "stripping", "striptease", "topless", "lingerie", "orgasm",
    "masturbat", "boobs", "busty", "breast", "succubus", "lust", "semen",
    "incest", "nympho", "prostitut", "seduc", "aphrodisiac", "libido",
    "cumshot", "creampie", "bukkake", "doujin", "oppai", "onahole", "camgirl",
    "onlyfans", "panties", "undress", "horny", "bimbo", "titjob", "blowjob",
    "handjob", "threesome", "explicit", "fanservice", "harem", "waifu",
    "nsfw", "hookup",
)

#: Terms that must match as a whole word, because a prefix match would create
#: obvious false positives ("anal" in "analysis", "futa" in "futanari").
ADULT_KEYWORDS_EXACT: tuple[str, ...] = (
    "anal", "futa", "orgy", "slut", "whore", "hooker", "dildo", "vibrator",
    "r18", "r-18", "18\\+", "\\+18", "x-rated", "h-game", "hgame",
)

#: Patterns screened against Steam's free-text ``notes`` content advisory.
ADULT_NOTE_PATTERNS: tuple[str, ...] = (
    "nudity", "sexual", "adult content", "adults only", "explicit sex",
    "uncensored", "hentai", "eroge", "pornograph", "mature content",
    "sexual content", "sex scenes", "erotic",
)

#: The three description variants produced by :func:`clean_games`.
TEXT_COLUMNS: dict[str, str] = {
    "original": "description_clean",
    "no_title": "description_notitle",
    "masked": "description_masked",
}

# Tokens that must never be treated as a removable "title fragment": they are
# too generic, so deleting them would mangle unrelated sentences.
_GENERIC_TITLE_TOKENS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "of", "and", "or", "in", "on", "at", "to", "for", "with",
        "game", "games", "simulator", "simulation", "story", "stories", "adventure",
        "adventures", "world", "worlds", "war", "wars", "quest", "quests", "hero",
        "heroes", "legend", "legends", "dark", "light", "life", "time", "final",
        "last", "new", "lost", "escape", "puzzle", "vr", "edition", "remastered",
        "deluxe", "definitive", "collection", "demo", "episode", "chapter", "part",
        "ii", "iii", "iv", "vi", "vii", "viii", "ix", "xi", "xii",
    }
)

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_BBCODE_RE = re.compile(r"\[/?[a-zA-Z][^\]]{0,40}\]")
_WS_RE = re.compile(r"\s+")
_REPEATED_PUNCT_RE = re.compile(r"([!?.,;:*_~^=+#\-])\1{2,}")
_NON_TEXT_RE = re.compile(r"[•●▪ ​﻿]")
_ABOUT_PREFIX_RE = re.compile(
    r"^\s*(about\s+(the\s+)?(this\s+)?game|about|overview|description|game\s+description)\s*[:\-–]?\s*",
    flags=re.IGNORECASE,
)
_PLACEHOLDER = "this game"


# --------------------------------------------------------------------------- #
# Genre normalisation
# --------------------------------------------------------------------------- #
def _genre_key(raw: str) -> str:
    """Normalise a raw genre string for synonym lookup."""
    text = unicodedata.normalize("NFKD", str(raw)).strip().lower()
    text = text.replace("&", "and")
    text = re.sub(r"[^a-z0-9\- ]+", " ", text)
    return _WS_RE.sub(" ", text).strip()


def normalize_genre(raw: str) -> str | None:
    """Map one raw Steam genre string onto a canonical label.

    Returns ``None`` when the string is not one of the target genres.
    """
    if raw is None:
        return None
    key = _genre_key(raw)
    if not key:
        return None
    canonical = GENRE_SYNONYMS.get(key)
    if canonical is None:
        return None
    return canonical if canonical in GENRES else None


def normalize_genres(raw_genres: Iterable[str] | None) -> list[str]:
    """Normalise a list of raw genres, de-duplicating and keeping label order."""
    if raw_genres is None:
        return []
    found: set[str] = set()
    for raw in raw_genres:
        canonical = normalize_genre(raw)
        if canonical is not None:
            found.add(canonical)
    return [genre for genre in GENRES if genre in found]


# --------------------------------------------------------------------------- #
# Text cleaning
# --------------------------------------------------------------------------- #
def clean_description(text: str | float | None) -> str:
    """Clean a raw store description for Transformer input."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    raw = str(text)
    raw = html.unescape(raw)
    raw = _HTML_TAG_RE.sub(" ", raw)
    raw = _BBCODE_RE.sub(" ", raw)
    raw = _URL_RE.sub(" ", raw)
    raw = _EMAIL_RE.sub(" ", raw)
    raw = unicodedata.normalize("NFKC", raw)
    raw = _NON_TEXT_RE.sub(" ", raw)
    raw = raw.replace("’", "'").replace("‘", "'")
    raw = raw.replace("“", '"').replace("”", '"')
    raw = _REPEATED_PUNCT_RE.sub(r"\1", raw)
    raw = _WS_RE.sub(" ", raw).strip()
    raw = _ABOUT_PREFIX_RE.sub("", raw)
    return raw.strip()


def title_variants(title: str) -> list[str]:
    """Return the surface forms of *title* worth removing from a description."""
    if not title:
        return []
    base = _WS_RE.sub(" ", str(title)).strip()
    if not base:
        return []

    variants: set[str] = {base}
    # Trademark / edition noise.
    stripped = re.sub(
        r"\s*[®™]\s*|\s*[-–:]\s*(deluxe|definitive|remastered|"
        r"complete|goty|game of the year|enhanced|ultimate|special)\s*edition\s*$",
        " ",
        base,
        flags=re.IGNORECASE,
    ).strip()
    if stripped:
        variants.add(stripped)

    # Parts either side of the usual separators.
    for part in re.split(r"\s*[:–—|/]\s*|\s+-\s+", base):
        part = part.strip(" -:–—|/")
        if len(part) >= 4 and _genre_key(part) not in _GENERIC_TITLE_TOKENS:
            variants.add(part)

    # Punctuation-free rendering ("Half-Life 2" -> "Half Life 2").
    depunct = _WS_RE.sub(" ", re.sub(r"[^\w\s]", " ", base)).strip()
    if len(depunct) >= 4:
        variants.add(depunct)

    # Initialism for titles with three or more informative words.
    words = [w for w in re.findall(r"[A-Za-z0-9]+", base)]
    if len(words) >= 3:
        initials = "".join(w[0] for w in words).upper()
        if len(initials) >= 3:
            variants.add(initials)

    keep = [
        v
        for v in variants
        if len(v) >= 3 and _genre_key(v) not in _GENERIC_TITLE_TOKENS
    ]
    # Longest first so "The Elder Scrolls V: Skyrim" is removed before "Skyrim".
    return sorted(set(keep), key=lambda v: (-len(v), v))


def remove_title_mentions(
    text: str, title: str, *, placeholder: str = _PLACEHOLDER
) -> str:
    """Replace every mention of *title* (and its variants) inside *text*."""
    if not text:
        return ""
    cleaned = text
    for variant in title_variants(title):
        # Exact match on word boundaries.
        exact = r"(?<!\w)" + re.escape(variant) + r"(?!\w)"
        cleaned = re.sub(exact, placeholder, cleaned, flags=re.IGNORECASE)

        # Looser match tolerating different separators between the title words
        # ("Half-Life 2" also catching "Half Life 2" / "Half:Life 2").
        tokens = re.findall(r"[A-Za-z0-9]+", variant)
        if len(tokens) >= 2:
            loose = (
                r"(?<!\w)"
                + r"[\s\W_]{0,3}".join(re.escape(token) for token in tokens)
                + r"(?!\w)"
            )
            cleaned = re.sub(loose, placeholder, cleaned, flags=re.IGNORECASE)

    # Collapse "this game this game" produced by overlapping variants.
    cleaned = re.sub(
        rf"(?:{re.escape(placeholder)})(?:[\s,'’]*{re.escape(placeholder)})+",
        placeholder,
        cleaned,
        flags=re.IGNORECASE,
    )
    return _WS_RE.sub(" ", cleaned).strip()


@lru_cache(maxsize=8)
def _genre_term_pattern(genres: tuple[str, ...]) -> re.Pattern[str]:
    """One compiled alternation covering every leak term for *genres*."""
    terms: set[str] = set()
    for genre in genres:
        terms.update(GENRE_LEAK_TERMS.get(genre, ()))
    alternatives = "|".join(
        re.escape(term) for term in sorted(terms, key=lambda t: (-len(t), t))
    )
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", flags=re.IGNORECASE)


def mask_genre_terms(
    text: str, *, genres: Sequence[str] = GENRES, placeholder: str = "[GENRE]"
) -> str:
    """Mask explicit genre vocabulary in *text* (optional leakage ablation)."""
    if not text:
        return ""
    masked = _genre_term_pattern(tuple(genres)).sub(placeholder, text)
    masked = re.sub(
        rf"(?:{re.escape(placeholder)})(?:[\s\-]+{re.escape(placeholder)})+",
        placeholder,
        masked,
    )
    return _WS_RE.sub(" ", masked).strip()


def word_count(text: str | None) -> int:
    """Number of whitespace-separated tokens in *text*."""
    if not text:
        return 0
    return len(str(text).split())


def ascii_ratio(text: str | None) -> float:
    """Fraction of ASCII characters -- a cheap English-language heuristic."""
    if not text:
        return 0.0
    raw = str(text)
    return sum(1 for ch in raw if ord(ch) < 128) / len(raw)


# --------------------------------------------------------------------------- #
# Label encoding
# --------------------------------------------------------------------------- #
def encode_labels(genres: Iterable[str], *, classes: Sequence[str] = GENRES) -> np.ndarray:
    """Multi-hot encode an iterable of canonical genre names."""
    index = {genre: i for i, genre in enumerate(classes)}
    vector = np.zeros(len(classes), dtype=np.float32)
    for genre in genres:
        position = index.get(genre)
        if position is not None:
            vector[position] = 1.0
    return vector


def decode_labels(
    vector: Sequence[float] | np.ndarray,
    *,
    classes: Sequence[str] = GENRES,
    threshold: float = 0.5,
) -> list[str]:
    """Inverse of :func:`encode_labels`."""
    array = np.asarray(vector, dtype=np.float32).ravel()
    if array.shape[0] != len(classes):
        raise ValueError(f"expected {len(classes)} scores, got {array.shape[0]}")
    return [genre for genre, score in zip(classes, array) if score >= threshold]


def label_matrix(frame: pd.DataFrame, *, classes: Sequence[str] = GENRES) -> np.ndarray:
    """Extract the ``(n_samples, n_classes)`` multi-hot matrix from a dataframe."""
    columns = [label_column(genre) for genre in classes]
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"dataframe is missing label columns: {missing}")
    return frame[columns].to_numpy(dtype=np.float32)


def positive_counts(frame: pd.DataFrame, *, classes: Sequence[str] = GENRES) -> dict[str, int]:
    """Per-class number of positive rows."""
    matrix = label_matrix(frame, classes=classes)
    return {genre: int(matrix[:, i].sum()) for i, genre in enumerate(classes)}


def class_weights(
    labels: np.ndarray, *, clip: float | None = None, eps: float = 1.0
) -> np.ndarray:
    """Compute ``pos_weight`` values for :class:`torch.nn.BCEWithLogitsLoss`."""
    matrix = np.asarray(labels, dtype=np.float64)
    positives = matrix.sum(axis=0)
    negatives = matrix.shape[0] - positives
    weights = (negatives + eps) / (positives + eps)
    if clip is not None:
        weights = np.clip(weights, 1.0 / clip if clip > 0 else 0.0, clip)
    return weights.astype(np.float32)


# --------------------------------------------------------------------------- #
# Eligibility + full cleaning pipeline
# --------------------------------------------------------------------------- #
@dataclass
class CleaningReport:
    """Structured account of everything the cleaning pipeline dropped."""

    rows_in: int = 0
    rows_out: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    genre_counts_raw: dict[str, int] = field(default_factory=dict)
    genre_counts_kept: dict[str, int] = field(default_factory=dict)
    unmapped_genres: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def drop(self, reason: str, count: int) -> None:
        if count:
            self.dropped[reason] = self.dropped.get(reason, 0) + int(count)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "rows_dropped_total": self.rows_in - self.rows_out,
            "dropped_by_reason": dict(sorted(self.dropped.items(), key=lambda kv: -kv[1])),
            "genre_counts_raw_top": dict(
                sorted(self.genre_counts_raw.items(), key=lambda kv: -kv[1])[:30]
            ),
            "genre_counts_kept": self.genre_counts_kept,
            "unmapped_genres_top": dict(
                sorted(self.unmapped_genres.items(), key=lambda kv: -kv[1])[:30]
            ),
            "notes": self.notes,
        }


def is_eligible(
    row: pd.Series, *, dataset: DatasetConfig | None = None
) -> tuple[bool, str]:
    """Check a single cleaned row, returning ``(ok, reason_if_not)``.

    Returns ``(ok, reason)``; *reason* is empty when the row is eligible.
    """
    cfg = dataset or CONFIG.dataset
    description = row.get("description_clean", "")
    if not str(row.get("name", "")).strip():
        return False, "missing_title"
    if not description:
        return False, "missing_description"
    words = word_count(description)
    if words < cfg.min_description_words:
        return False, "description_too_short"
    if words > cfg.max_description_words:
        return False, "description_too_long"
    if ascii_ratio(description) < cfg.min_ascii_ratio:
        return False, "non_english_description"
    if not row.get("genres_norm"):
        return False, "no_target_genre"
    if int(row.get("n_screenshots", 0)) < 1:
        return False, "no_screenshot"
    return True, ""


def _as_regex(term: str) -> str:
    """Escape a term unless it already contains an intentional regex escape."""
    return term if "\\" in term else re.escape(term)


@lru_cache(maxsize=4)
def _adult_keyword_pattern() -> re.Pattern[str]:
    """One alternation over the stem terms and the whole-word terms."""
    stems = "|".join(_as_regex(t) for t in sorted(ADULT_KEYWORDS, key=lambda t: (-len(t), t)))
    exact = "|".join(
        _as_regex(t) for t in sorted(ADULT_KEYWORDS_EXACT, key=lambda t: (-len(t), t))
    )
    return re.compile(
        rf"(?<!\w)(?:{stems})|(?<!\w)(?:{exact})(?!\w)", flags=re.IGNORECASE
    )


@lru_cache(maxsize=4)
def _adult_note_pattern() -> re.Pattern[str]:
    return re.compile("|".join(ADULT_NOTE_PATTERNS), flags=re.IGNORECASE)


def adult_content_mask(frame: pd.DataFrame) -> pd.Series:
    """Flag rows that are sexually explicit or adult-only.

    Combines four signals: genre tags, ``required_age``, the ``notes`` advisory and a keyword
    screen.
    """
    index = frame.index
    flags = pd.Series(False, index=index)

    if "genres_list" in frame.columns:
        flags |= frame["genres_list"].map(
            lambda genres: any(_genre_key(g) in ADULT_GENRES for g in genres)
        )
    elif "genres" in frame.columns:
        flags |= frame["genres"].map(
            lambda genres: any(
                _genre_key(g) in ADULT_GENRES
                for g in (genres if isinstance(genres, (list, tuple, np.ndarray)) else [])
            )
        )

    if "required_age" in frame.columns:
        flags |= pd.to_numeric(frame["required_age"], errors="coerce").fillna(0) >= 18

    if "notes" in frame.columns:
        flags |= frame["notes"].fillna("").astype(str).str.contains(_adult_note_pattern())

    keyword = _adult_keyword_pattern()
    for column in ("name", "short_description", "description_clean", "description_raw"):
        if column in frame.columns:
            flags |= frame[column].fillna("").astype(str).str.contains(keyword)

    return flags


def _eligibility_reasons(frame: pd.DataFrame, cfg: DatasetConfig) -> pd.Series:
    """Vectorised counterpart of :func:`is_eligible` (empty string = eligible)."""
    description = frame["description_clean"].fillna("")
    words = description.str.split().str.len().fillna(0)
    conditions = [
        (frame["name"].fillna("").str.strip() == "", "missing_title"),
        (description.str.strip() == "", "missing_description"),
        (words < cfg.min_description_words, "description_too_short"),
        (words > cfg.max_description_words, "description_too_long"),
        (description.map(ascii_ratio) < cfg.min_ascii_ratio, "non_english_description"),
        (frame["genres_norm"].map(len) == 0, "no_target_genre"),
        (frame["n_screenshots"].fillna(0) < 1, "no_screenshot"),
    ]
    reasons = pd.Series("", index=frame.index, dtype=object)
    for mask, reason in conditions:
        reasons = reasons.mask((reasons == "") & mask.fillna(True), reason)
    return reasons


def add_text_variants(
    games: pd.DataFrame,
    *,
    dataset: DatasetConfig | None = None,
    min_words: int | None = None,
    classes: Sequence[str] = GENRES,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Add the leakage-controlled description variants to a cleaned frame.

    Returns ``(frame, dropped)``; separated from clean_games so it runs after subsampling.
    """
    cfg = dataset or CONFIG.dataset
    floor = min_words if min_words is not None else max(5, cfg.min_description_words // 2)
    out = games.copy()
    out["description_notitle"] = [
        remove_title_mentions(description, name)
        for description, name in zip(out["description_clean"], out["name"])
    ]
    out["description_masked"] = [
        mask_genre_terms(text, genres=tuple(classes)) for text in out["description_notitle"]
    ]
    before = len(out)
    out = out[out["description_notitle"].map(word_count) >= floor].reset_index(drop=True)
    return out, {"empty_after_title_removal": int(before - len(out))}


def clean_games(
    raw: pd.DataFrame,
    *,
    dataset: DatasetConfig | None = None,
    description_field: str = "short_description",
    require_screenshots: int | None = None,
    compute_text_variants: bool = True,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Run the full game-level cleaning pipeline.

    Returns ``(frame, report)`` with one row per surviving game and a count for every drop
    reason.
    """
    cfg = dataset or CONFIG.dataset
    min_shots = cfg.screenshots_per_game if require_screenshots is None else require_screenshots
    report = CleaningReport(rows_in=int(len(raw)))
    frame = raw.copy()

    # ---- 1. normalise container types ---------------------------------- #
    def _as_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            return [str(v) for v in value if v is not None and str(v).strip()]
        if isinstance(value, float) and np.isnan(value):
            return []
        text = str(value).strip()
        return [text] if text else []

    frame["app_id"] = frame["appID"].astype(str).str.strip()
    frame["name"] = frame["name"].fillna("").astype(str).str.strip()
    frame["genres_list"] = frame["genres"].map(_as_list)
    frame["screenshot_urls_list"] = frame["screenshots"].map(_as_list)
    frame["n_screenshots"] = frame["screenshot_urls_list"].map(len)

    report.genre_counts_raw = dict(
        Counter(g for genres in frame["genres_list"] for g in genres)
    )
    report.unmapped_genres = {
        genre: count
        for genre, count in report.genre_counts_raw.items()
        if normalize_genre(genre) is None
    }

    # ---- 2. duplicates -------------------------------------------------- #
    before = len(frame)
    frame = frame.drop_duplicates(subset=["app_id"], keep="first")
    report.drop("duplicate_app_id", before - len(frame))

    # ---- 3. text cleaning ---------------------------------------------- #
    frame["description_raw"] = frame[description_field].fillna("").astype(str)
    frame["description_clean"] = frame["description_raw"].map(clean_description)
    frame["genres_norm"] = frame["genres_list"].map(normalize_genres)

    # ---- 4. row level eligibility -------------------------------------- #
    reasons = _eligibility_reasons(frame, cfg)
    for reason, count in Counter(reasons[reasons != ""]).items():
        report.drop(str(reason), count)
    frame = frame[reasons == ""].copy()

    # ---- 5. screenshot requirement -------------------------------------- #
    before = len(frame)
    frame = frame[frame["n_screenshots"] >= min_shots].copy()
    report.drop(f"fewer_than_{min_shots}_screenshots", before - len(frame))

    # ---- 6. adult content ----------------------------------------------- #
    # Four independent signals, because Steam's genre tags alone miss almost
    # all of it (see adult_content_mask).
    before = len(frame)
    frame = frame[~adult_content_mask(frame)].copy()
    report.drop("adult_content", before - len(frame))

    # ---- 7. duplicate descriptions ------------------------------------- #
    before = len(frame)
    frame = frame.drop_duplicates(subset=["description_clean"], keep="first")
    report.drop("duplicate_description", before - len(frame))

    # ---- 8. leakage-controlled text variants ---------------------------- #
    if compute_text_variants:
        frame, dropped = add_text_variants(frame, dataset=cfg)
        for reason, count in dropped.items():
            report.drop(reason, count)
    else:
        report.notes.append(
            "description_notitle / description_masked were NOT computed here; "
            "call add_text_variants() after subsampling."
        )

    # ---- 9. label encoding --------------------------------------------- #
    frame["genres"] = frame["genres_norm"].map(lambda gs: "|".join(gs))
    frame["n_genres"] = frame["genres_norm"].map(len)
    encoded = np.stack([encode_labels(gs) for gs in frame["genres_norm"]]) if len(frame) else np.zeros(
        (0, len(GENRES)), dtype=np.float32
    )
    for i, column in enumerate(label_columns()):
        frame[column] = encoded[:, i].astype(np.int8)

    frame["screenshot_urls"] = frame["screenshot_urls_list"].map(lambda urls: "\t".join(urls))
    frame["description_word_count"] = frame["description_clean"].map(word_count)

    keep_columns = [
        column
        for column in (
            "app_id",
            "name",
            "description_raw",
            "description_clean",
            "description_notitle",
            "description_masked",
            "description_word_count",
            "genres",
            "n_genres",
            "n_screenshots",
            "screenshot_urls",
            *label_columns(),
        )
        if column in frame.columns
    ]
    out = frame[keep_columns].reset_index(drop=True)
    report.rows_out = int(len(out))
    report.genre_counts_kept = positive_counts(out) if len(out) else {g: 0 for g in GENRES}
    report.notes.append(
        f"Description field used: '{description_field}'. Titles were replaced by "
        f"'{_PLACEHOLDER}' in 'description_notitle'; explicit genre vocabulary was "
        f"additionally masked in 'description_masked'."
    )
    LOGGER.info(
        "Cleaning: %d raw rows -> %d clean games (%d dropped)",
        report.rows_in,
        report.rows_out,
        report.rows_in - report.rows_out,
    )
    return out, report


# --------------------------------------------------------------------------- #
# Image validation helpers
# --------------------------------------------------------------------------- #
def image_average_hash(path: str | Path, *, hash_size: int = 8) -> str | None:
    """Return a 64-bit average hash (hex) for near-duplicate screenshot detection.

    Returns ``None`` when the file is missing or cannot be decoded.
    """
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as image:
            grey = image.convert("L").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
            pixels = np.asarray(grey, dtype=np.float32)
    except (FileNotFoundError, OSError, UnidentifiedImageError, ValueError):
        return None
    bits = (pixels > pixels.mean()).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:0{hash_size * hash_size // 4}x}"
