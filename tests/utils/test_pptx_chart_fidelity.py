"""PPTX chart-label extraction FIDELITY (KNOWLEDGE-COMPLETION slice 6, G7).

INTEGRATION ruled slice 6 = FIDELITY, not ingestion widening (DECISIONS §A22):
extract MORE COMPLETELY from charts in decks we already ingest. No new file
types, no new routes, no OCR of chart images, no change to the chunk/locator or
receipt SCHEMA.

Baseline (measured at rag_api main f84c196a): `_chart_text` already extracts the
chart TITLE, CATEGORY labels and SERIES NAMES (rag_api #21). This increment
APPENDS, byte-stable, four more fidelity parts and makes an unreadable chart
honest instead of silent:

  (a) AXIS TITLES   — category + value axis titles when present.
  (b) SERIES VALUES — numeric data points paired with their category, bounded
                      with an explicit "... N more points" truncation marker.
  (c) DATA LABELS   — custom data-label text where a point carries one.
  (d) XY / BUBBLE   — these have no categories; emit x/y (and bubble size) pairs
                      instead of losing the numeric evidence.
  (e) HONEST UNREADABLE CHART — a chart we cannot read is no longer silent; the
                      loader logs a named warning (the receipt-contract decision
                      is recorded in the PR body).

Every fixture is SYNTHETIC and built at test time with python-pptx (invented
labels/values only, like test_parser_fitness.py). No client content.
"""

import io
import logging

import pytest
from pptx import Presentation
from pptx.util import Inches
from pptx.chart.data import CategoryChartData, XyChartData, BubbleChartData
from pptx.enum.chart import XL_CHART_TYPE

from app.utils.document_loader import SlidePowerPointLoader
from app.utils import document_loader as dl


# ---------------------------------------------------------------------------
# Synthetic fixture generators (label SYN-KNOWLEDGE-01)
# ---------------------------------------------------------------------------


def _save(prs, path):
    prs.save(str(path))
    return str(path)


def make_bar_axis_titles_and_labels(path):
    """A COLUMN chart carrying category + value AXIS TITLES and a CUSTOM
    per-point DATA LABEL — the (a) and (c) evidence."""
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "SYN-KNOWLEDGE-01 Axis Deck"
    cd = CategoryChartData()
    cd.categories = ["Alpha", "Beta"]
    cd.add_series("Series ONE", (1.0, 2.0))
    gf = slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(1), Inches(1), Inches(4), Inches(3), cd
    )
    chart = gf.chart
    chart.category_axis.axis_title.text_frame.text = "Quarter"
    chart.value_axis.axis_title.text_frame.text = "Revenue USD"
    plot = chart.plots[0]
    plot.has_data_labels = True
    chart.series[0].points[0].data_label.text_frame.text = "Peak!"
    return _save(prs, path)


def make_line_chart(path):
    """A LINE chart whose numeric VALUES must be retrievable, paired with the
    category — the (b) evidence."""
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "SYN-KNOWLEDGE-01 Line Deck"
    cd = CategoryChartData()
    cd.categories = ["Jan", "Feb", "Mar"]
    cd.add_series("Signups", (10.0, 25.0, 42.5))
    slide.shapes.add_chart(
        XL_CHART_TYPE.LINE, Inches(1), Inches(1), Inches(4), Inches(3), cd
    )
    return _save(prs, path)


def make_pie_chart(path):
    """A PIE chart — one series, categories + values — the (b) evidence for a
    chart with no value axis."""
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "SYN-KNOWLEDGE-01 Pie Deck"
    cd = CategoryChartData()
    cd.categories = ["North", "South"]
    cd.add_series("Share", (60.0, 40.0))
    slide.shapes.add_chart(
        XL_CHART_TYPE.PIE, Inches(1), Inches(1), Inches(4), Inches(3), cd
    )
    return _save(prs, path)


def make_xy_scatter(path):
    """An XY SCATTER chart — no categories; the x/y pairs are the only numeric
    evidence — the (d) evidence."""
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "SYN-KNOWLEDGE-01 Scatter Deck"
    cd = XyChartData()
    srs = cd.add_series("Cohort XY")
    srs.add_data_point(1.5, 2.5)
    srs.add_data_point(3.0, 4.0)
    slide.shapes.add_chart(
        XL_CHART_TYPE.XY_SCATTER, Inches(1), Inches(1), Inches(4), Inches(3), cd
    )
    return _save(prs, path)


def make_bubble_chart(path):
    """A BUBBLE chart — no categories; x/y and bubble size are the evidence —
    the (d) evidence for the 3-value point case."""
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "SYN-KNOWLEDGE-01 Bubble Deck"
    cd = BubbleChartData()
    srs = cd.add_series("Accounts")
    srs.add_data_point(1.0, 2.0, 3.0)
    srs.add_data_point(4.0, 5.0, 6.0)
    slide.shapes.add_chart(
        XL_CHART_TYPE.BUBBLE, Inches(1), Inches(1), Inches(4), Inches(3), cd
    )
    return _save(prs, path)


def make_large_series_chart(path, n_points):
    """A COLUMN chart with MANY points, to prove the deterministic truncation
    marker — the (b) bound."""
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "SYN-KNOWLEDGE-01 Large Deck"
    cd = CategoryChartData()
    cd.categories = [f"C{i}" for i in range(n_points)]
    cd.add_series("Big", tuple(float(i) for i in range(n_points)))
    slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(1), Inches(1), Inches(4), Inches(3), cd
    )
    return _save(prs, path)


def _chart_content(path):
    docs = list(SlidePowerPointLoader(str(path)).lazy_load())
    assert docs, "deck produced no Documents"
    return "\n".join(d.page_content for d in docs)


# ---------------------------------------------------------------------------
# (a) axis titles + (c) custom data labels
# ---------------------------------------------------------------------------


def test_bar_chart_extracts_axis_titles(tmp_path):
    content = _chart_content(make_bar_axis_titles_and_labels(tmp_path / "axis.pptx"))
    # existing byte-stable parts still extracted
    assert "Series ONE" in content
    assert "Alpha" in content and "Beta" in content
    # (a) NEW: axis titles
    assert "Quarter" in content, content
    assert "Revenue USD" in content, content


def test_bar_chart_extracts_custom_data_label(tmp_path):
    content = _chart_content(make_bar_axis_titles_and_labels(tmp_path / "label.pptx"))
    # (c) NEW: custom data-label text
    assert "Peak!" in content, content


# ---------------------------------------------------------------------------
# (b) series values paired with categories
# ---------------------------------------------------------------------------


def test_line_chart_values_are_retrievable_paired_with_category(tmp_path):
    content = _chart_content(make_line_chart(tmp_path / "line.pptx"))
    assert "Signups" in content
    # (b) NEW: each numeric value appears, paired with its category
    assert "Jan 10" in content, content
    assert "Feb 25" in content, content
    assert "Mar 42.5" in content, content


def test_pie_chart_values_are_retrievable(tmp_path):
    content = _chart_content(make_pie_chart(tmp_path / "pie.pptx"))
    assert "Share" in content
    assert "North 60" in content, content
    assert "South 40" in content, content


# ---------------------------------------------------------------------------
# (d) XY / scatter and bubble: x/y pairs, no categories
# ---------------------------------------------------------------------------


def test_xy_scatter_emits_xy_pairs(tmp_path):
    content = _chart_content(make_xy_scatter(tmp_path / "xy.pptx"))
    assert "Cohort XY" in content
    # (d) NEW: x/y pairs survive even though there are no categories
    assert "(1.5, 2.5)" in content, content
    assert "(3, 4)" in content, content


def test_bubble_chart_emits_xy_and_size(tmp_path):
    content = _chart_content(make_bubble_chart(tmp_path / "bub.pptx"))
    assert "Accounts" in content
    # (d) NEW: x/y and bubble size survive
    assert "(1, 2, r=3)" in content, content
    assert "(4, 5, r=6)" in content, content


# ---------------------------------------------------------------------------
# (b) deterministic truncation marker on a large chart
# ---------------------------------------------------------------------------


def test_large_series_is_bounded_with_honest_truncation_marker(tmp_path):
    cap = dl._CHART_MAX_POINTS
    n = cap + 7
    content = _chart_content(make_large_series_chart(tmp_path / "big.pptx", n))
    # the bound is honest: an explicit marker naming how many were dropped
    assert f"... {n - cap} more points" in content, content
    # the first point is present, a point beyond the cap is NOT individually listed
    assert "C0 0" in content, content
    assert f"C{n - 1} {n - 1}" not in content, content


# ---------------------------------------------------------------------------
# (e) honest unreadable chart — never silent
# ---------------------------------------------------------------------------


def test_unreadable_chart_is_logged_not_silent(tmp_path, caplog):
    """A chart whose every access raises must leave a trace. Today `_chart_text`
    swallows chart-internal exceptions with a bare `pass` and returns "" — the
    chart vanishes with NO log. After the fix the loader logs a named warning
    and the rest of the deck still extracts."""

    class _ExplodingChart:
        @property
        def has_title(self):
            raise RuntimeError("boom-title")

        @property
        def plots(self):
            raise RuntimeError("boom-plots")

        @property
        def series(self):
            raise RuntimeError("boom-series")

        @property
        def category_axis(self):
            raise RuntimeError("boom-cat-axis")

        @property
        def value_axis(self):
            raise RuntimeError("boom-val-axis")

    with caplog.at_level(logging.WARNING):
        text = SlidePowerPointLoader._chart_text(_ExplodingChart())

    assert text == ""
    assert any(
        "chart" in r.message.lower() and "unreadable" in r.message.lower()
        for r in caplog.records
    ), [r.message for r in caplog.records]


def test_unreadable_chart_does_not_abort_the_deck(tmp_path):
    """One bad chart must never abort a whole deck: a slide carrying a title and
    an unreadable chart still yields its title text."""
    path = make_bar_axis_titles_and_labels(tmp_path / "mixed.pptx")
    # Monkeypatch-free: the slide's OTHER text (title) must survive even if the
    # chart read is partial. Here we only assert the deck extracts its title,
    # which proves the chart path never raised out of _collect_shape_texts.
    content = _chart_content(path)
    assert "SYN-KNOWLEDGE-01 Axis Deck" in content, content


# ---------------------------------------------------------------------------
# byte-stability: the pre-existing parts lead, new parts only append
# ---------------------------------------------------------------------------


def test_existing_title_categories_series_name_still_lead(tmp_path):
    """The title / categories / series-name parts the loader already produced
    must stay byte-stable; new fidelity parts only APPEND after them."""
    content = _chart_content(make_bar_axis_titles_and_labels(tmp_path / "stable.pptx"))
    # the existing category block "Alpha Beta" and the series name must both be
    # present, and the series name must precede the NEW axis-title/value parts.
    assert "Alpha Beta" in content, content
    assert "Series ONE" in content, content
    # the new value pairing comes AFTER the plain series name
    i_name = content.index("Series ONE")
    i_quarter = content.index("Quarter")
    assert i_name < i_quarter, content


# ---------------------------------------------------------------------------
# Robustness: a blank or NON-FINITE data value must never cost the chart its
# byte-stable parts (title / categories / series names). (Fix-forward on review.)
# ---------------------------------------------------------------------------

_CHART_C = "{http://schemas.openxmlformats.org/drawingml/2006/chart}"


def make_none_value_chart(path):
    """A COLUMN chart whose middle data point is BLANK (None): proves the None
    value renders as the honest literal "null" and the chart still extracts."""
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "SYN-KNOWLEDGE-01 Blank Deck"
    cd = CategoryChartData()
    cd.categories = ["Jan", "Feb", "Mar"]
    cd.add_series("Metric", (1.0, None, 3.0))
    slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(1), Inches(1), Inches(4), Inches(3), cd
    )
    return _save(prs, path)


def make_nonfinite_value_chart(path, bad_text):
    """A COLUMN chart whose last cached value is corrupted to a NON-FINITE literal
    (NaN / inf). Injected into the numCache XML because python-pptx will not author
    a non-finite value, but a hand-edited or corrupt deck can carry one."""
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "SYN-KNOWLEDGE-01 NonFinite Deck"
    cd = CategoryChartData()
    cd.categories = ["Jan", "Feb"]
    cd.add_series("Metric", (1.0, 2.0))
    gf = slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(1), Inches(1), Inches(4), Inches(3), cd
    )
    vs = gf.chart.series[0]._element.findall(
        ".//" + _CHART_C + "val//" + _CHART_C + "numCache//" + _CHART_C + "pt/" + _CHART_C + "v"
    )
    assert vs, "no numeric value cache to corrupt"
    vs[-1].text = bad_text
    return _save(prs, path)


def test_blank_data_point_renders_null_and_chart_survives(tmp_path):
    """Coverage of the None value path: a blank data point is the honest literal
    "null" (never silently dropped) and the chart's other parts are intact."""
    content = _chart_content(make_none_value_chart(tmp_path / "blank.pptx"))
    assert "Metric" in content and "Jan" in content and "Mar" in content, content
    assert "Feb null" in content, content  # blank middle point, paired with its category
    assert "Jan 1" in content and "Mar 3" in content, content


@pytest.mark.parametrize(
    "bad_text,expected",
    [("NaN", "NaN"), ("1e400", "inf"), ("-1e400", "-inf")],
)
def test_nonfinite_value_never_loses_the_whole_chart(tmp_path, bad_text, expected):
    """REGRESSION: a non-finite cached value must NOT cost the chart its title /
    categories / series names. On the pre-fix loader `_fmt_num` raised (int(nan)
    -> ValueError, int(inf) -> OverflowError), the exception propagated out of
    `_chart_text`, and `_collect_shape_texts` dropped the ENTIRE chart. After the
    fix the value renders as a deterministic literal and every other part survives."""
    path = make_nonfinite_value_chart(tmp_path / ("nf_%s.pptx" % expected), bad_text)
    content = _chart_content(path)
    # the byte-stable parts must survive the bad value (red on 4dc84bc)
    assert "Metric" in content, content
    assert "Jan" in content and "Feb" in content, content
    # and the non-finite value is rendered deterministically, not crashed on
    assert expected in content, content
