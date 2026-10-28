"""The review page in a real browser (Playwright, headless Chromium) on the batch16 scenario's
results, plus a syntax check of the page's JavaScript."""
import re
import subprocess
import sys

import pytest

import stampede as sd
from conftest import needs_node, needs_playwright
from helpers import SCRIPT, rows, run_sim


@needs_node
def test_page_javascript_parses(tmp_path):
    js = sd.REVIEW_PAGE.split("<script>")[1].split("</script>")[0].replace("ROWS", "[]")
    (tmp_path / "page.js").write_text(js)
    p = subprocess.run(["node", "--check", str(tmp_path / "page.js")], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


@pytest.fixture
def review(tmp_path):
    """batch16 results in tmp_path, a review server on them, a browser page. Yields (page, csv, msgs)."""
    pw_api = pytest.importorskip("playwright.sync_api")
    run_sim("batch16", tmp_path)
    scans = tmp_path / "Scans"
    csv_path = tmp_path / "Scans-review" / "photos.csv"

    def setup(rs):  # a few flagged photos without a stamp, one with an unreadable stamp
        for r in rs:
            k = r["rel"][6:10]
            if k in ("0013", "0016"):
                r.update(status="review", flags="rotation-check no-date")
            if k == "0014":
                r.update(status="review", flags="no-date stamp-unread")
    sd.update_csv(csv_path, setup)
    srv = subprocess.Popen([sys.executable, str(SCRIPT), "review", str(scans), "--no-browser"],
                           stdout=subprocess.PIPE, text=True)
    url = re.search(r"http://\S+/", srv.stdout.readline()).group(0)
    msgs = []
    with pw_api.sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.on("dialog", lambda d: (msgs.append(d.message), d.accept()))
        page.goto(url)
        page.wait_for_selector(".card")
        try:
            yield page, csv_path, msgs
        finally:
            browser.close()
            srv.terminate()


def shown(page):
    return sorted(t.split(" · ")[0][6:10] for t in page.locator(".card .meta").all_inner_texts())


def row(csv_path, k):
    return next(r for rel, r in rows(csv_path.parent.parent / "Scans").items() if rel[6:10] == k)


def card(page, k):
    return page.locator(".card", has_text=f"_{k}.jpg")


def wait(page):
    page.wait_for_timeout(700)


@needs_playwright
def test_flagged_view_ok_and_use_buttons(review):
    page, csv_path, msgs = review
    assert shown(page) == ["0001", "0002", "0006", "0008", "0011", "0013", "0014", "0016"]
    page.wait_for_function("[...document.querySelectorAll('.card img')].every(i => i.complete && i.naturalWidth > 0)",
                           timeout=15000)  # every thumbnail loads
    assert card(page, "0006").locator("button.guess").all_inner_texts() == ["use 2006-11-15", "use 2006-11-19"]
    card(page, "0006").locator("button.guess", has_text="2006-11-19").click()
    wait(page)
    r = row(csv_path, "0006")
    assert (r["date"], r["status"], r["edited"]) == ("2006-11-19", "ok", "date")
    assert "0006" not in shown(page)  # done: leaves the Flagged view
    card(page, "0013").locator("button.ok").click()
    wait(page)
    assert "0013" not in shown(page) and row(csv_path, "0013")["edited"] == "status"


@needs_playwright
def test_no_stamp_bulk_ok_then_date_to_enter(review):
    page, csv_path, msgs = review
    page.click("button[data-f=nostamp]")
    wait(page)
    assert shown(page) == ["0013", "0016"]
    page.click("#okall")
    wait(page)
    assert msgs[-1] == "Mark 2 photos as OK?" and shown(page) == []
    assert row(csv_path, "0013")["status"] == row(csv_path, "0016")["status"] == "ok"
    page.click("button[data-f=needdate]")
    wait(page)
    assert shown(page) == ["0001", "0002", "0006", "0008", "0011", "0014"]


@needs_playwright
def test_bulk_guesses_skip_coin_flips(review):
    page, csv_path, msgs = review
    page.click("#guessall")
    page.wait_for_timeout(1500)
    assert "day/month/year" in msgs[-1] and "left for you to pick" in msgs[-1]
    assert row(csv_path, "0001")["date"] == "2006-03-12" and row(csv_path, "0002")["date"] == "2006-03-12"
    assert row(csv_path, "0008")["date"] == "" and row(csv_path, "0011")["date"] == ""  # several options


@needs_playwright
def test_a_stale_page_cannot_overwrite_newer_scan_results(review):
    page, csv_path, msgs = review

    def scan_changes(rs):  # what a running pass 3 would do after the page was loaded
        for r in rs:
            if r["rel"].endswith("0001.jpg"):
                r.update(flags="landscape-assumed readers-disagree:2006-03-12/2006-12-03")
    sd.update_csv(csv_path, scan_changes)
    page.click("#guessall")
    page.wait_for_timeout(1500)
    assert "not changed" in msgs[-1]
    assert row(csv_path, "0001")["date"] == ""  # refused
    assert row(csv_path, "0002")["date"] == "2006-03-12"


@needs_playwright
def test_date_validation_rotation_skip_unskip(review):
    page, csv_path, msgs = review
    page.click("button[data-f=all]")
    c = card(page, "0013")
    c.locator("input.date").fill("1960-01-01")
    c.locator("input.date").press("Enter")
    wait(page)
    assert "outside 1975-2008" in msgs[-1] and row(csv_path, "0013")["date"] == ""
    card(page, "0014").locator("button.rr").click()
    wait(page)
    r = row(csv_path, "0014")
    assert r["rotate"] == "180" and r["edited"] == "rotate"
    card(page, "0014").locator("button.sk").click()
    wait(page)
    assert row(csv_path, "0014")["status"] == "skip"
    card(page, "0014").locator("button.sk").click()
    wait(page)
    r = row(csv_path, "0014")
    assert r["status"] == "review" and r["edited"] == "rotate"  # un-skip hands it back to the scan


@needs_playwright
def test_security_headers_token_and_cookies(review):
    page, csv_path, msgs = review
    st = page.evaluate("fetch('/api/update',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'})"
                       ".then(r=>r.status)")
    assert st == 403  # no session token
    h = page.evaluate("fetch('/').then(r=>[r.headers.get('x-frame-options'), r.headers.get('content-security-policy')])")
    assert h == ["DENY", "frame-ancestors 'none'"]
    st = page.evaluate("""(async()=>{document.cookie='foo={"a":1}; path=/';
        return (await fetch(document.querySelector('.card img').src)).status})()""")
    assert st == 200  # another localhost app's odd cookie doesn't hide ours
