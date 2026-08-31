"""L0 audit Batch E — #19 calendar content (SIFMA, TEL_AVIV, RIYADH).

Green-oracle anchors, verified, not guessed (§4). SIFMA is the USD/SOFR calendar — its Saturday-holiday
rule injects phantom fixing dates into SOFR windows if wrong.
"""

from datetime import date

from pricebook_ng.foundation import business_days_between, get_calendar

SIFMA = get_calendar("US_GOVERNMENT_SECURITIES")
TLV = get_calendar("TEL_AVIV")
RIYADH = get_calendar("RIYADH")


# ── #19a — SIFMA: Saturday-holiday → preceding Friday, WITH the New-Year year-boundary exception ──
def test_sifma_july4_saturday_shifts_to_friday_closed() -> None:  # NY-Fed anchor: no SOFR 2020-07-03
    assert not SIFMA.is_business_day(date(2020, 7, 3))  # July-4 (Sat) observed on Friday → CLOSED
    assert not SIFMA.is_business_day(date(2026, 7, 3))  # recurs


def test_sifma_new_year_saturday_stays_open() -> None:  # NY-Fed anchor: 2021-12-31 open
    # Jan-1-2022 (Sat) is NOT shifted back across the year boundary to Dec-31-2021 → OPEN
    assert SIFMA.is_business_day(date(2021, 12, 31))


def test_sifma_juneteenth_not_observed_in_2021_first_close_2022() -> None:  # verified: SIFMA first closed 2022
    assert SIFMA.is_business_day(date(2021, 6, 18))  # 2021: short notice, no SIFMA close → OPEN
    assert not SIFMA.is_business_day(date(2022, 6, 20))  # 2022-06-19 (Sun) → observed Mon → CLOSED


def test_sifma_no_unintended_flips() -> None:
    assert SIFMA.is_business_day(date(2021, 3, 16))  # an ordinary Tuesday, unchanged
    assert not SIFMA.is_business_day(date(2020, 7, 4))  # the Saturday itself (weekend)
    assert SIFMA.is_business_day(date(2020, 7, 6))  # the Monday after, unchanged


def test_sifma_sofr_window_across_july3_2020() -> None:
    # [2020-07-01, 2020-07-08): Wed Thu (Fri=holiday) (Sat Sun) Mon Tue → 4 business days, not 5
    assert business_days_between(date(2020, 7, 1), date(2020, 7, 8), SIFMA) == 4


# ── #19b — TEL_AVIV: drop the fixed Gregorian Hebrew-holiday dates → weekend-only ──
def test_tel_aviv_no_falsely_closed_weekday() -> None:
    assert TLV.is_business_day(date(2025, 9, 25))  # Thu — was falsely closed by a stale fixed date
    assert TLV.is_business_day(date(2025, 4, 14))  # a former Passover fixed-date, now open
    assert not TLV.is_business_day(date(2025, 9, 26))  # Friday — the FRI_SAT weekend


# ── #19c — RIYADH: documented pre-2013 limit (post-2013 FRI_SAT approximation) ──
def test_riyadh_documented_limit() -> None:
    assert RIYADH.is_business_day(date(2015, 6, 25))  # post-2013 Thursday — a business day (correct)
    # KNOWN LIMIT (logged, deferred): pre-2013 the weekend was Thu/Fri, but we model FRI_SAT, so a
    # pre-2013 Thursday is (approximately) a business day — the finding is documented, not fixed here.
    assert RIYADH.is_business_day(date(2010, 6, 24))  # Thu — the documented approximation
