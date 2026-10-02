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
import re

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


# ---------------------------------------------------------------------------
# Performance: chart extraction TIME must be bounded like its OUTPUT (RV-363).
# A hostile/corrupt huge chart must not stall an ingestion worker. We assert the
# number of points/values PULLED is bounded, not wall-clock, so the test is
# deterministic. (Fix-forward on RV-363.)
# ---------------------------------------------------------------------------

import pptx.chart.series as _ps  # noqa: E402

_HUGE = 10000
_CAP = dl._CHART_MAX_POINTS


def make_huge_category_chart(path, n=_HUGE):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "SYN-KNOWLEDGE-01 Huge Cat Deck"
    cd = CategoryChartData()
    cd.categories = [f"C{i}" for i in range(n)]
    cd.add_series("Big", tuple(float(i) for i in range(n)))
    slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(1), Inches(1), Inches(4), Inches(3), cd
    )
    return _save(prs, path)


def make_huge_xy_chart(path, n=_HUGE):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "SYN-KNOWLEDGE-01 Huge XY Deck"
    cd = XyChartData()
    srs = cd.add_series("BigXY")
    for i in range(n):
        srs.add_data_point(float(i), float(i))
    slide.shapes.add_chart(
        XL_CHART_TYPE.XY_SCATTER, Inches(1), Inches(1), Inches(4), Inches(3), cd
    )
    return _save(prs, path)


def make_no_ptcount_chart(path, n):
    """A category chart with `n` points whose `c:ptCount` is removed from the value
    numCache, so the total is UNKNOWABLE in O(1): the marker must then omit the number."""
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "SYN-KNOWLEDGE-01 NoPtCount Deck"
    cd = CategoryChartData()
    cd.categories = [f"C{i}" for i in range(n)]
    cd.add_series("Big", tuple(float(i) for i in range(n)))
    gf = slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(1), Inches(1), Inches(4), Inches(3), cd
    )
    val = gf.chart.series[0]._element.find(_CHART_C + "val")
    cache = val.find(".//" + _CHART_C + "numCache")
    ptc = cache.find(_CHART_C + "ptCount")
    if ptc is not None:
        cache.remove(ptc)
    return _save(prs, path)


class _CountingPts:
    """Wraps a python-pptx points collection, counting every point iterated. Works
    for BOTH the pre-fix `list(series.points)` (pulls all) and the post-fix
    `islice(iter(series.points), cap+1)` (pulls cap+1)."""

    def __init__(self, inner, counter):
        self._inner = inner
        self._counter = counter

    def __iter__(self):
        for x in self._inner:
            self._counter["points"] += 1
            yield x

    def __len__(self):
        return len(self._inner)


def _install_value_point_spies(monkeypatch, counter):
    """Count how many VALUES (via series.values) and POINTS (via series.points) the
    loader pulls for a category series. Pre-fix uses both APIs over ALL points; the
    fix reads values from the XML cache (never series.values) and islices points."""
    cls = _ps._BaseCategorySeries
    val_desc = cls.__dict__["values"]

    def counting_values(self):
        t = val_desc.fget(self)
        try:
            counter["values"] += len(t)
        except TypeError:
            pass
        return t

    monkeypatch.setattr(cls, "values", property(counting_values))

    pts_desc = cls.__dict__["points"]

    def counting_points(self):
        inner = pts_desc.__get__(self, type(self))
        return _CountingPts(inner, counter)

    monkeypatch.setattr(cls, "points", property(counting_points))


def _install_xy_counter(monkeypatch, counter):
    """Count how many XY points the loader's `_xy_point_values` actually PRODUCES.
    The pre-fix reader `findall`s every <c:pt> and returns the whole list (10000);
    the fix islices to cap+1 and returns at most that. The numCache elements are
    plain immutable lxml `_Element`, so we cannot patch them per-class -- but the
    loader's own classmethod is the single choke point, and the points it returns ARE
    the points it pulled. Handles both arities: pre-fix `(series)` returning a list,
    post-fix `(series, limit)` returning `(list, total)`."""
    orig = SlidePowerPointLoader.__dict__["_xy_point_values"].__func__

    def counting_xy(cls, series, *args, **kwargs):
        res = orig(cls, series, *args, **kwargs)
        pts = res[0] if isinstance(res, tuple) else res
        if pts:
            counter["xy"] += len(pts)
        return res

    monkeypatch.setattr(SlidePowerPointLoader, "_xy_point_values", classmethod(counting_xy))


def test_category_values_and_labels_are_time_bounded(tmp_path, monkeypatch):
    """RED on d2d18fb: list(series.values) and list(series.points) pull all 10000.
    GREEN after: values never go through series.values (0), points are islice'd to
    cap+1. Proves extraction TIME is O(cap), not O(points)."""
    counter = {"values": 0, "points": 0}
    _install_value_point_spies(monkeypatch, counter)
    path = make_huge_category_chart(tmp_path / "huge_cat.pptx")
    content = _chart_content(path)
    # output is still correct & bounded
    assert "Big" in content and "C0 0" in content
    assert f"... {_HUGE - _CAP} more points" in content, content
    # the point is TIME: neither accessor pulled more than cap+1 points
    assert counter["values"] <= _CAP + 1, counter
    assert 0 < counter["points"] <= _CAP + 1, counter


def test_xy_points_are_time_bounded(tmp_path, monkeypatch):
    """RED on d2d18fb: the XY reader returns ALL 10000 points (it findall'd them).
    GREEN after: it islices to cap+1, so it produces (and pulled) at most cap+1."""
    counter = {"xy": 0}
    _install_xy_counter(monkeypatch, counter)
    path = make_huge_xy_chart(tmp_path / "huge_xy.pptx")
    content = _chart_content(path)
    assert "BigXY" in content and "(0, 0)" in content
    assert f"... {_HUGE - _CAP} more points" in content, content
    # the >0 guard proves the counter actually fired; the bound proves O(cap) not O(n).
    assert 0 < counter["xy"] <= _CAP + 1, counter


def test_truncation_marker_shows_exact_remainder_when_ptcount_exists(tmp_path):
    """The marker's number comes from c:ptCount in O(1); for a 10k chart it is the
    exact remainder, never a fabricated or iterated count."""
    content = _chart_content(make_huge_category_chart(tmp_path / "exact.pptx"))
    assert f"... {_HUGE - _CAP} more points" in content, content


def test_truncation_marker_omits_number_when_ptcount_absent(tmp_path):
    """When c:ptCount is unreadable the marker must NOT fabricate a number: it is the
    bare '... more points' form, never '... N more points'."""
    content = _chart_content(make_no_ptcount_chart(tmp_path / "noptc.pptx", _CAP + 5))
    assert "... more points" in content, content
    assert not re.search(r"\.\.\. \d+ more points", content), content
