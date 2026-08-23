"""Behavioural tests for :mod:`gamesense.data.preprocessing`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gamesense.config import GENRES, label_column, label_columns
from gamesense.data.preprocessing import (
    TEXT_COLUMNS,
    adult_content_mask,
    class_weights,
    clean_description,
    clean_games,
    decode_labels,
    encode_labels,
    image_average_hash,
    label_matrix,
    mask_genre_terms,
    normalize_genre,
    normalize_genres,
    positive_counts,
    remove_title_mentions,
    title_variants,
    word_count,
)

#: Games in ``raw_games_frame`` that must survive cleaning with two screenshots.
EXPECTED_CLEAN_APP_IDS = {"100", "101", "102", "103", "104"}

#: The single-screenshot game, recovered when only one screenshot is required.
ONE_SCREENSHOT_APP_ID = "109"

#: Number of hex characters in an ``image_average_hash`` of the default size.
_DEFAULT_HASH_SIZE = 8


# --------------------------------------------------------------------------- #
# clean_description
# --------------------------------------------------------------------------- #
def test_clean_description_removes_markup_urls_and_boilerplate() -> None:
    """Markup, links, addresses and store boiler-plate must all disappear."""
    raw = (
        "<h2>About This Game</h2>"
        "<p>Rescue the colony &amp; survive! Visit https://store.example.com "
        "or e-mail support@example.com.</p> [b]Bold claim[/b] Amazing!!!!!"
    )
    cleaned = clean_description(raw)

    assert "<" not in cleaned and ">" not in cleaned  # HTML tags
    assert "&amp;" not in cleaned and "&" in cleaned  # entity decoded, not dropped
    assert "http" not in cleaned and "store.example.com" not in cleaned  # URL
    assert "@" not in cleaned  # e-mail address
    assert "[b]" not in cleaned and "[/b]" not in cleaned  # Steam BBCode
    assert "Bold claim" in cleaned  # ... but its content survives
    assert not cleaned.lower().startswith("about")  # boiler-plate prefix
    assert cleaned.startswith("Rescue")
    assert "!!!!" not in cleaned  # runs of punctuation squeezed
    assert "  " not in cleaned and cleaned == cleaned.strip()  # whitespace


def test_clean_description_preserves_case_and_sentence_punctuation() -> None:
    """WordPiece needs natural text: casing and sentence marks must survive."""
    natural = "The Hero Awakens. Can you survive 12 nights? Yes, you can!"
    assert clean_description(natural) == natural


@pytest.mark.parametrize("empty", [None, "", "   ", float("nan"), np.nan])
def test_clean_description_returns_empty_string_for_missing_input(empty) -> None:
    """Missing descriptions must become ``""`` rather than ``"nan"`` or raise."""
    assert clean_description(empty) == ""


# --------------------------------------------------------------------------- #
# title_variants / remove_title_mentions
# --------------------------------------------------------------------------- #
def test_title_variants_covers_full_title_and_colon_subtitle() -> None:
    """A ``"Franchise: Subtitle"`` name must yield both halves worth removing."""
    variants = title_variants("The Elder Scrolls V: Skyrim")
    assert "The Elder Scrolls V: Skyrim" in variants
    assert "Skyrim" in variants
    # Longest first, so the full title is replaced before the subtitle alone.
    assert variants[0] == "The Elder Scrolls V: Skyrim"


def test_title_variants_drops_purely_generic_fragments() -> None:
    """Fragments such as ``"the"`` or ``"game"`` must never become removable."""
    variants = title_variants("The Legend of Adventure")
    assert all(v.lower() not in {"the", "game", "legend", "adventure"} for v in variants)


def test_remove_title_mentions_is_case_insensitive_and_covers_subtitles() -> None:
    """Both the full title and its subtitle go, whatever their casing."""
    text = "The Elder Scrolls V: Skyrim is huge. SKYRIM rules."
    cleaned = remove_title_mentions(text, "The Elder Scrolls V: Skyrim")
    assert "Skyrim" not in cleaned and "SKYRIM" not in cleaned
    assert "Elder Scrolls" not in cleaned
    assert cleaned == "this game is huge. this game rules."


def test_remove_title_mentions_matches_punctuation_variants() -> None:
    """``"Half-Life 2"`` must also catch the unhyphenated spelling."""
    cleaned = remove_title_mentions("Half Life 2 is a shooter in City 17.", "Half-Life 2")
    assert "Half" not in cleaned and "Life" not in cleaned
    assert "City 17" in cleaned  # unrelated proper noun untouched


def test_remove_title_mentions_keeps_the_sentence_usable() -> None:
    """A placeholder (not deletion) keeps the sentence grammatical."""
    cleaned = remove_title_mentions("Skyrim is huge.", "Skyrim")
    assert cleaned.strip() != ""
    assert cleaned.startswith("this game")
    assert cleaned.endswith("is huge.")


def test_remove_title_mentions_collapses_repeated_placeholders() -> None:
    """Overlapping variants must not stutter ``"this game this game"``."""
    cleaned = remove_title_mentions("Skyrim Skyrim Skyrim is huge.", "Skyrim")
    assert cleaned.count("this game") == 1


def test_generic_title_does_not_blank_an_unrelated_description() -> None:
    """A game literally called "The Game" must keep a usable description."""
    description = "The Game is a puzzle about shapes, and the game is hard."
    cleaned = remove_title_mentions(description, "The Game")
    assert word_count(cleaned) >= word_count(description) - 2
    assert "puzzle about shapes" in cleaned


# --------------------------------------------------------------------------- #
# mask_genre_terms
# --------------------------------------------------------------------------- #
def test_mask_genre_terms_masks_explicit_vocabulary_case_insensitively() -> None:
    """Explicit genre words are the leakage this ablation must remove."""
    masked = mask_genre_terms("An RPG about role-playing and RACING.")
    assert "RPG" not in masked and "role-playing" not in masked
    assert "RACING" not in masked and "racing" not in masked.lower()
    assert "[GENRE]" in masked


def test_mask_genre_terms_prefers_multi_word_phrases() -> None:
    """Multi-word phrases must win over their own substrings."""
    masked = mask_genre_terms("A first-person shooter with real-time strategy elements.")
    assert "first-person" not in masked
    assert "real-time" not in masked and "strategy" not in masked
    assert masked == "A [GENRE] with [GENRE] elements."


def test_mask_genre_terms_handles_turn_based_strategy() -> None:
    """``"turn-based strategy"`` is the textbook leakage phrase."""
    masked = mask_genre_terms("A turn-based strategy game about racing.")
    assert "turn-based" not in masked and "strategy" not in masked
    assert "racing" not in masked
    assert masked.count("[GENRE]") >= 2


def test_mask_genre_terms_leaves_non_genre_words_alone() -> None:
    """Ordinary prose must be returned untouched."""
    prose = "Beautiful weather and friendly villagers await beyond the mountains."
    assert mask_genre_terms(prose) == prose


def test_mask_genre_terms_returns_empty_string_for_empty_input() -> None:
    assert mask_genre_terms("") == ""


# --------------------------------------------------------------------------- #
# Genre normalisation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Role-Playing", "RPG"),
        ("role playing", "RPG"),
        ("Sport", "Sports"),
        ("RACING", "Racing"),
        ("Simulations", "Simulation"),
        ("Action", "Action"),
    ],
)
def test_normalize_genre_maps_synonyms(raw: str, expected: str) -> None:
    assert normalize_genre(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["Indie", "Utilities", "Massively Multiplayer", "Free To Play", "Early Access", "", None],
)
def test_normalize_genre_rejects_out_of_space_labels(raw) -> None:
    """Business / software / audience descriptors are not prediction targets."""
    assert normalize_genre(raw) is None


def test_normalize_genres_deduplicates_and_orders_like_genres() -> None:
    """Output order must match :data:`GENRES` so label columns stay aligned."""
    result = normalize_genres(["Sports", "Sport", "Strategy", "Role-Playing", "Indie", "Action"])
    assert result == ["Action", "RPG", "Sports", "Strategy"]
    assert result == [genre for genre in GENRES if genre in set(result)]


def test_normalize_genres_handles_missing_input() -> None:
    assert normalize_genres(None) == []
    assert normalize_genres([]) == []


# --------------------------------------------------------------------------- #
# Label encoding
# --------------------------------------------------------------------------- #
def test_encode_labels_sets_the_right_positions_and_ignores_unknowns() -> None:
    vector = encode_labels(["RPG", "Action", "Totally Made Up"])
    assert vector.shape == (len(GENRES),)
    assert vector[GENRES.index("RPG")] == pytest.approx(1.0)
    assert vector[GENRES.index("Action")] == pytest.approx(1.0)
    assert vector.sum() == pytest.approx(2.0)  # the unknown genre was ignored


def test_encode_decode_labels_round_trip() -> None:
    original = ["Adventure", "Racing", "Strategy"]
    assert decode_labels(encode_labels(original)) == [
        genre for genre in GENRES if genre in set(original)
    ]


def test_decode_labels_rejects_a_wrong_length_vector() -> None:
    with pytest.raises(ValueError, match=str(len(GENRES))):
        decode_labels([1.0, 0.0, 1.0])


def test_label_matrix_shape_and_dtype(synthetic_games: pd.DataFrame) -> None:
    matrix = label_matrix(synthetic_games)
    assert matrix.shape == (len(synthetic_games), len(GENRES))
    assert matrix.dtype == np.float32
    assert set(np.unique(matrix)).issubset({0.0, 1.0})


def test_label_matrix_raises_when_a_label_column_is_missing(
    synthetic_games: pd.DataFrame,
) -> None:
    incomplete = synthetic_games.drop(columns=[label_column(GENRES[0])])
    with pytest.raises(KeyError, match=label_column(GENRES[0])):
        label_matrix(incomplete)


# --------------------------------------------------------------------------- #
# Class weights
# --------------------------------------------------------------------------- #
def _hand_built_labels() -> np.ndarray:
    """Return a 10-row matrix with 2, 5 and 8 positives in its three columns."""
    matrix = np.zeros((10, 3), dtype=np.float64)
    matrix[:2, 0] = 1.0
    matrix[:5, 1] = 1.0
    matrix[:8, 2] = 1.0
    return matrix


def test_class_weights_is_the_negative_to_positive_ratio() -> None:
    """Un-smoothed weights must equal ``negatives / positives`` exactly."""
    weights = class_weights(_hand_built_labels(), eps=0.0)
    assert weights.tolist() == pytest.approx([8 / 2, 5 / 5, 2 / 8])


def test_class_weights_smoothing_matches_the_documented_formula() -> None:
    """With the default ``eps`` the ratio is Laplace-smoothed by one count."""
    weights = class_weights(_hand_built_labels())
    assert weights.tolist() == pytest.approx([(8 + 1) / (2 + 1), 1.0, (2 + 1) / (8 + 1)])


def test_class_weights_respects_the_clip() -> None:
    clip = 2.0
    weights = class_weights(_hand_built_labels(), clip=clip)
    assert weights.max() <= clip
    assert weights.min() >= 1.0 / clip


def test_class_weights_survives_a_class_with_no_positives() -> None:
    """An empty class must yield a finite weight, not ``inf`` or ``nan``."""
    labels = np.zeros((10, len(GENRES)), dtype=np.float64)
    labels[:3, 0] = 1.0  # only the first class has positives
    weights = class_weights(labels)
    assert np.isfinite(weights).all()
    assert (weights > 0).all()


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
def test_clean_games_keeps_exactly_the_eligible_rows(
    clean_games_frame: pd.DataFrame,
) -> None:
    """Duplicates, unusable descriptions and non-games must all be gone."""
    assert set(clean_games_frame["app_id"]) == EXPECTED_CLEAN_APP_IDS
    assert len(clean_games_frame) == len(EXPECTED_CLEAN_APP_IDS)
    assert clean_games_frame["app_id"].is_unique  # the duplicate appID collapsed


def test_clean_games_emits_every_expected_column(clean_games_frame: pd.DataFrame) -> None:
    """The downstream dataset relies on these columns existing by name."""
    for column in (*label_columns(), *TEXT_COLUMNS.values(), "app_id", "name", "genres"):
        assert column in clean_games_frame.columns
    labels = clean_games_frame[label_columns()].to_numpy()
    assert set(np.unique(labels)).issubset({0, 1})
    assert (labels.sum(axis=1) >= 1).all()  # every kept game has a target genre


def test_clean_games_labels_agree_with_the_genre_strings(
    clean_games_frame: pd.DataFrame,
) -> None:
    """``y_*`` columns and the pipe-joined ``genres`` string must not diverge."""
    for row in clean_games_frame.to_dict("records"):
        listed = set(str(row["genres"]).split("|")) - {""}
        flagged = {genre for genre in GENRES if row[label_column(genre)] == 1}
        assert listed == flagged
        assert row["n_genres"] == len(flagged)


def test_cleaning_report_accounts_for_every_dropped_row(cleaning_report) -> None:
    """The report has to add up, otherwise the dataset table in the write-up lies."""
    assert cleaning_report.rows_in == 11
    assert cleaning_report.rows_out == len(EXPECTED_CLEAN_APP_IDS)
    assert cleaning_report.rows_in - cleaning_report.rows_out == sum(
        cleaning_report.dropped.values()
    )
    for reason, count in cleaning_report.dropped.items():
        assert isinstance(reason, str) and reason.strip(), "drop reasons must be named"
        assert count > 0


def test_cleaning_report_names_each_kind_of_rejection(cleaning_report) -> None:
    """The awkward rows in the fixture must be attributed to a real reason."""
    reasons = cleaning_report.dropped
    assert reasons["duplicate_app_id"] == 1
    assert reasons["missing_description"] == 1
    assert reasons["no_target_genre"] == 1  # the software-only row
    assert reasons["fewer_than_2_screenshots"] == 1
    assert reasons["description_too_short"] >= 1
    # "Indie" / "Utilities" are outside the label space and must be reported.
    assert "Indie" in cleaning_report.unmapped_genres


def test_cleaning_report_dict_is_self_consistent(cleaning_report) -> None:
    payload = cleaning_report.as_dict()
    assert payload["rows_dropped_total"] == payload["rows_in"] - payload["rows_out"]
    assert sum(payload["dropped_by_reason"].values()) == payload["rows_dropped_total"]
    assert set(payload["genre_counts_kept"]) == set(GENRES)


def test_cleaning_report_genre_counts_match_the_frame(
    clean_games_frame: pd.DataFrame, cleaning_report
) -> None:
    assert cleaning_report.genre_counts_kept == positive_counts(clean_games_frame)


def test_screenshot_requirement_is_the_only_reason_one_game_was_dropped(
    raw_games_frame: pd.DataFrame,
) -> None:
    """Relaxing ``require_screenshots`` must recover exactly that one game."""
    relaxed, report = clean_games(raw_games_frame, require_screenshots=1)
    assert set(relaxed["app_id"]) == EXPECTED_CLEAN_APP_IDS | {ONE_SCREENSHOT_APP_ID}
    assert "fewer_than_1_screenshots" not in report.dropped


def test_clean_games_survives_a_batch_where_every_row_is_rejected(
    raw_games_frame: pd.DataFrame,
) -> None:
    """An all-rejected batch must yield an empty frame, not raise."""
    rejected_only = raw_games_frame[raw_games_frame["appID"].isin(["105", "106", "107", "108"])]
    frame, report = clean_games(rejected_only, require_screenshots=2)
    assert len(frame) == 0
    assert report.rows_out == 0
    assert set(frame.columns) >= set(label_columns())


# --------------------------------------------------------------------------- #
# Image hashing
# --------------------------------------------------------------------------- #
def _write_split_image(path: Path, *, vertical: bool) -> Path:
    """Write a half-black / half-white JPEG so its average hash is non-trivial."""
    from PIL import Image

    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    if vertical:
        pixels[:, 32:] = 255
    else:
        pixels[32:, :] = 255
    Image.fromarray(pixels).save(path, format="JPEG", quality=95)
    return path


def test_image_average_hash_is_stable_for_identical_images(tmp_path: Path) -> None:
    left = _write_split_image(tmp_path / "a.jpg", vertical=True)
    copy = _write_split_image(tmp_path / "b.jpg", vertical=True)
    digest = image_average_hash(left)
    assert digest is not None
    assert len(digest) == _DEFAULT_HASH_SIZE**2 // 4
    assert digest == image_average_hash(copy)


def test_image_average_hash_separates_different_images(tmp_path: Path) -> None:
    vertical = _write_split_image(tmp_path / "vertical.jpg", vertical=True)
    horizontal = _write_split_image(tmp_path / "horizontal.jpg", vertical=False)
    assert image_average_hash(vertical) != image_average_hash(horizontal)


def test_image_average_hash_returns_none_for_unusable_files(tmp_path: Path) -> None:
    """A corrupt payload and an absent file are both "unusable", not fatal."""
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"this is not a valid JPEG payload")
    assert image_average_hash(corrupt) is None
    assert image_average_hash(tmp_path / "does_not_exist.jpg") is None


# --------------------------------------------------------------------------- #
# Adult-content screening
# --------------------------------------------------------------------------- #
# This project renders real screenshots into a report and into a public Streamlit app, so
# sexually explicit games have to be excluded outright rather than merely down-weighted.
def test_adult_mask_catches_the_genre_tags() -> None:
    frame = pd.DataFrame(
        {
            "name": ["Clean Platformer", "Tagged Game"],
            "short_description": ["A cheerful platformer.", "A visual novel."],
            "genres_list": [["Action", "Indie"], ["Casual", "Sexual Content"]],
        }
    )
    assert adult_content_mask(frame).tolist() == [False, True]


def test_adult_mask_catches_required_age_18() -> None:
    frame = pd.DataFrame(
        {
            "name": ["A", "B"],
            "short_description": ["Race around a track.", "A story game."],
            "required_age": [0, 18],
        }
    )
    assert adult_content_mask(frame).tolist() == [False, True]


def test_adult_mask_screens_the_notes_content_advisory() -> None:
    frame = pd.DataFrame(
        {
            "name": ["A", "B", "C"],
            "short_description": ["Build a city.", "Solve puzzles.", "Fly a plane."],
            "notes": [
                "",
                "This game contains Nudity or Sexual Content.",
                "Contains flashing lights.",
            ],
        }
    )
    assert adult_content_mask(frame).tolist() == [False, True, False]


@pytest.mark.parametrize(
    "text",
    [
        "A hentai puzzle game.",
        "An adult visual novel with explicit scenes.",
        "Uncensored artwork of your favourite characters.",
        "A succubus needs your help in this lewd adventure.",
        "Seduce the cast in this eroge.",
        "An 18+ dating game.",
    ],
)
def test_adult_mask_catches_explicit_descriptions(text: str) -> None:
    frame = pd.DataFrame({"name": ["Untitled"], "short_description": [text]})
    assert bool(adult_content_mask(frame).iloc[0]) is True


@pytest.mark.parametrize(
    "name,text",
    [
        # "Maiden"/"Maids" must not match the "maid" idea, "stripes" must not
        # match "strip", and "analysis" must not match the whole-word "anal".
        ("Mistress of Maidens", "Command a fantasy army across a war-torn kingdom."),
        ("The Zebra-Man!", "Revenge wears stripes in this top-down action game."),
        ("Data Detective", "Perform careful analysis of each crime scene."),
        ("Breakout Legends", "A fast arcade game about breaking bricks."),
        ("Futanari Falls", "A whitewater kayaking simulator."),
    ],
)
def test_adult_mask_does_not_over_filter_ordinary_games(name: str, text: str) -> None:
    """Only the deliberately-explicit fifth case should be flagged."""
    frame = pd.DataFrame({"name": [name], "short_description": [text]})
    flagged = bool(adult_content_mask(frame).iloc[0])
    assert flagged is ("Futanari" in name)


def test_adult_mask_works_when_optional_columns_are_absent() -> None:
    """The screen must degrade gracefully on frames without notes/required_age."""
    frame = pd.DataFrame({"name": ["Kart Racer"], "short_description": ["Race karts."]})
    assert adult_content_mask(frame).tolist() == [False]


def test_clean_games_removes_adult_rows_and_counts_them(
    raw_games_frame: pd.DataFrame,
) -> None:
    explicit = pd.DataFrame(
        [
            {
                "appID": "900",
                "name": "Hentai Puzzle Deluxe",
                "short_description": (
                    "A sliding puzzle game featuring uncensored hentai artwork of "
                    "several characters across many increasingly difficult levels."
                ),
                "genres": ["Casual", "Indie"],
                "screenshots": ["https://example.invalid/900/a.jpg",
                                "https://example.invalid/900/b.jpg"],
            },
            {
                "appID": "901",
                "name": "Ordinary Kart Racer",
                "short_description": (
                    "Race karts around twenty colourful circuits, collect power ups "
                    "and beat your friends in split screen multiplayer races."
                ),
                "genres": ["Racing", "Casual"],
                "screenshots": ["https://example.invalid/901/a.jpg",
                                "https://example.invalid/901/b.jpg"],
            },
        ]
    )
    combined = pd.concat([raw_games_frame, explicit], ignore_index=True)
    frame, report = clean_games(combined, require_screenshots=2)

    assert "901" in set(frame["app_id"]), "an ordinary racing game must survive"
    assert "900" not in set(frame["app_id"]), "an explicit game must be removed"
    assert report.dropped.get("adult_content", 0) >= 1
