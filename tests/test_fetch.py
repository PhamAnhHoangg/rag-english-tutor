import httpx
import pytest

from tutor.sources import FetchError, fetch_wikibooks, get_json


def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_html_error_page_raises_clear_error():
    c = client(lambda r: httpx.Response(403, headers={"content-type": "text/html"}, text="<html>Forbidden</html>"))
    with pytest.raises(FetchError, match="HTTP 403 .*Forbidden"):
        get_json(c, "https://x/api", {}, sleep=lambda s: None)


def test_retries_on_429_then_succeeds():
    calls = []

    def handler(r):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429, headers={"retry-after": "1"}, text="slow down")
        return httpx.Response(200, json={"ok": True})

    waits = []
    assert get_json(client(handler), "https://x/api", {}, sleep=waits.append) == {"ok": True}
    assert waits == [1.0, 1.0]


def test_gives_up_after_retries():
    with pytest.raises(FetchError, match="after 2 retries"):
        get_json(client(lambda r: httpx.Response(503)), "https://x/api", {}, retries=2, sleep=lambda s: None)


def test_fetch_wikibooks_skips_failing_page(capsys):
    def handler(r):
        q = r.url.params
        if q.get("list") == "allpages":
            return httpx.Response(200, json={"query": {"allpages": [{"title": "English Grammar/A"}, {"title": "English Grammar/B"}]}})
        if q.get("titles") == "English Grammar/A":
            return httpx.Response(404, headers={"content-type": "text/html"}, text="nope")
        return httpx.Response(200, json={"query": {"pages": [{"title": "English Grammar/B", "extract": "Body."}]}})

    pages = list(fetch_wikibooks(client=client(handler), pause=0))
    assert pages == [("English Grammar/B", "Body.")]
    assert "skipping Wikibooks page 'English Grammar/A'" in capsys.readouterr().err
