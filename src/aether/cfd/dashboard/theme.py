"""One palette, applied once, so every panel looks like the same instrument.

The reference mock-up for this was a black screen with neon cyan monospace and
glowing red tracks -- the visual language of a film about a command centre
rather than of one. It reads as urgent for about a minute and then it is
tiring, and worse, it spends its loudest colour on chrome: if the borders and
the headings are already cyan, a genuinely alarming number has nowhere left to
go.

So the chrome here is quiet on purpose. Slate surfaces, a muted steel accent,
and grey text carry the frame; saturated colour is reserved for things that
have actually happened -- a diverged field, a failed run, a residual that has
stopped moving. On a phone at arm's length that is the difference between
seeing the one red row and scanning ten cyan ones.

Monospace is used for numbers and never for prose. Tabular figures line up
column-wise, which is the whole point for a residual or a coefficient, and are
harder work for a sentence.
"""

from __future__ import annotations

from typing import Any

from nicegui import ui

__all__ = [
    "BAD",
    "GOOD",
    "MUTED",
    "SIDE",
    "SIDE_ICON",
    "STATE_COLOUR",
    "WARN",
    "apply",
    "axis_style",
    "chart_base",
    "side_style",
]

#: Surfaces, from the page up. Three steps is enough to separate a page from a
#: card from a raised strip, and more would start to look like decoration.
_PAGE = "#0e1116"
_SURFACE = "#161a21"
_RAISED = "#1d222b"
_BORDER = "#272d38"

#: The one accent. Steel rather than cyan: legible on slate, and dull enough
#: that a warning beside it still reads as a warning.
_ACCENT = "#5a8ac2"

#: Reserved for meaning, never for chrome.
GOOD = "#4f9d72"
WARN = "#c9973f"
BAD = "#c25f55"
MUTED = "#8b93a3"

#: Which side a panel belongs to, as chrome only.
#:
#: This is the one place the palette had to be argued with. RED wants red and
#: BLUE wants blue, but ``BAD`` is already red and the whole point of the rule
#: above is that a saturated colour means *something happened*. A scarlet RED
#: CELL header would be the loudest thing on a screen that is merely idle, and
#: an actual alert beside it would have nowhere left to go.
#:
#: So the side colours are deliberately low-chroma -- a brick and a steel, each
#: desaturated well below ``BAD`` and ``GOOD`` -- and they are allowed on
#: chrome only: a rail item, a panel's left edge, a section heading, a journal
#: line's gutter. Never on a number, a badge, or a status. Side is *where you
#: are*; colour with meaning is *what has happened*; and those must not be the
#: same signal.
SIDE = {
    "RED": "#a8706a",
    "BLUE": "#6b8fb5",
    "SHARED": "#7d8798",
    "OFFLINE": "#6f7a6a",
}

#: The glyph that names each side in the rail, so the grouping survives both
#: colour-blindness and the collapsed 56 px rail where labels are gone.
SIDE_ICON = {
    "RED": "swords",
    "BLUE": "shield_moon",
    "SHARED": "hub",
    "OFFLINE": "science",
}


def side_style(side: str, *, edge: bool = True) -> str:
    """Inline style marking an element as belonging to ``side``.

    A left edge rather than a fill: it survives on a card that already has a
    background, it costs no contrast against the text on it, and four of them
    stacked down a column read as four groups rather than as four alerts.
    """
    colour = SIDE.get(side, SIDE["SHARED"])
    if not edge:
        return f"color: {colour};"
    return f"border-left: 3px solid {colour};"


#: Quasar colour names by run state, used for the badges in the run list.
#: Registered as custom colours below so the palette lives in one place.
STATE_COLOUR = {
    "queued": "aether-muted",
    "meshing": "aether-warn",
    "adapting": "aether-warn",
    "solving": "aether-accent",
    "done": "aether-good",
    "failed": "aether-bad",
    "stopped": "aether-muted",
}


def apply() -> None:
    """Install the palette and the handful of CSS rules Quasar does not cover.

    Called once per page. ``ui.colors`` sets Quasar's own tokens, so
    ``color=positive`` and friends resolve to these rather than to the
    defaults, and the custom names are what the state badges use.
    """
    ui.dark_mode().enable()
    ui.colors(
        primary=_ACCENT,
        secondary="#4a5568",
        accent=_ACCENT,
        dark=_SURFACE,
        dark_page=_PAGE,
        positive=GOOD,
        negative=BAD,
        warning=WARN,
        info=_ACCENT,
        # Named so a badge can say `color=aether-good` and mean this exact
        # green rather than Quasar's, which is two shades brighter.
        **{
            "aether-good": GOOD,
            "aether-warn": WARN,
            "aether-bad": BAD,
            "aether-muted": "#5b6373",
            "aether-accent": _ACCENT,
            "aether-surface": _SURFACE,
            "aether-raised": _RAISED,
            "aether-red": SIDE["RED"],
            "aether-blue": SIDE["BLUE"],
            "aether-shared": SIDE["SHARED"],
            "aether-offline": SIDE["OFFLINE"],
        },
    )
    ui.add_head_html(
        f"""
        <style>
          body {{ background: {_PAGE}; }}
          .nicegui-card {{
            background: {_SURFACE};
            border: 1px solid {_BORDER};
            box-shadow: none;
          }}
          /* Numbers get tabular figures so columns of residuals line up. */
          .font-mono, .q-table {{
            font-variant-numeric: tabular-nums;
            font-feature-settings: "tnum";
          }}
          .q-table thead th {{
            color: {MUTED};
            font-weight: 500;
            letter-spacing: 0.02em;
          }}
          .q-tab {{ text-transform: none; letter-spacing: 0.01em; }}
          /* The rail: quiet until hovered or selected, and the selected item
             carries its side's colour on the edge rather than as a fill. */
          .ag-rail .q-item {{
            border-radius: 4px;
            min-height: 34px;
            padding: 2px 10px;
            margin: 1px 6px;
            border-left: 3px solid transparent;
          }}
          .ag-rail .q-item:hover {{ background: {_RAISED}; }}
          .ag-rail .q-item.ag-active {{
            background: {_RAISED};
            font-weight: 500;
          }}
          .ag-rail-group {{
            color: {MUTED};
            font-size: 10px;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            padding: 10px 16px 2px;
          }}
          /* A panel that belongs to a side wears it on its leading edge. */
          .ag-sided {{ border-left-width: 3px; border-left-style: solid; }}

          /* --- 3-D scenes, and what it takes to make them work on a phone ---

             `ui.scene` takes a pixel width and height, which is the wrong unit
             for a screen whose size is not known until it is opened. Its
             JavaScript `resize()` reads `clientWidth`/`clientHeight` off the
             wrapper and is already bound to `window.resize`, so giving the
             wrapper a *CSS* size is all it takes for the canvas to follow the
             viewport -- including through an Android orientation change.

             `touch-action: none` is the one that actually decides whether this
             is usable on a phone. Without it the browser claims a one-finger
             drag for page scrolling before three.js ever sees it, so the globe
             cannot be rotated by touch at all: it just scrolls the page behind
             it. With it, OrbitControls gets the gesture -- one finger to orbit,
             two to pinch-zoom and pan.

             The height is viewport-relative with a ceiling: tall enough to be
             worth looking at on a phone held upright, and not so tall on a
             desktop that the controls beside it fall off the bottom. */
          .ag-scene {{
            width: 100%;
            height: min(62vh, 560px);
            min-height: 260px;
            touch-action: none;
            display: block;
          }}
          .ag-scene > canvas {{
            width: 100% !important;
            height: 100% !important;
            display: block;
            touch-action: none;
          }}
          /* The situation display: a framed region rather than a card, so it
             reads as the page's ground truth and not as one panel among many.
             Its header stays legible when it is folded away. */
          .ag-situation {{
            background: {_SURFACE};
            border: 1px solid {_BORDER};
            border-radius: 4px;
          }}
          .ag-situation .q-expansion-item__container > .q-item {{
            min-height: 34px;
            padding: 2px 12px;
          }}
          .ag-situation .q-expansion-item__content {{ padding: 0 8px 8px; }}
          /* A control strip under a display: quiet, dense, out of the way. */
          .ag-toolbar {{
            padding: 2px 4px;
            border-top: 1px solid {_BORDER};
          }}
          .ag-toolbar .q-toggle__label {{ font-size: 11px; color: {MUTED}; }}

          @media (max-width: 700px) {{
            /* Landscape on a phone is short: give the scene most of it rather
               than a fixed fraction that leaves a letterbox. */
            .ag-scene {{ height: min(70vh, 420px); }}
          }}
          ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
          ::-webkit-scrollbar-track {{ background: {_PAGE}; }}
          ::-webkit-scrollbar-thumb {{
            background: {_BORDER};
            border-radius: 4px;
          }}
          ::-webkit-scrollbar-thumb:hover {{ background: #39414f; }}
        </style>
        """
    )


def chart_base() -> dict[str, Any]:
    """Chart-level styling shared by every ECharts panel.

    Kept here rather than repeated at each chart because the alternative is
    what the first version did: every chart carrying its own idea of what a
    gridline looks like, drifting apart one panel at a time.
    """
    return {
        "backgroundColor": "transparent",
        "textStyle": {"color": "#c8cedb"},
        "legend": {"textStyle": {"color": MUTED, "fontSize": 10}},
    }


def axis_style() -> dict[str, Any]:
    """Styling for one axis, to be spread into an ``xAxis`` or ``yAxis``."""
    return {
        "axisLine": {"lineStyle": {"color": _BORDER}},
        "axisLabel": {"color": MUTED, "fontSize": 10},
        "nameTextStyle": {"color": MUTED, "fontSize": 10},
        "splitLine": {"lineStyle": {"color": _BORDER, "type": "dashed"}},
    }
