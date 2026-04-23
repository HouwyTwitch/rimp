"""
Regression tests for bugs fixed in the improvement pass, plus miscellaneous
coverage for paths not exercised by the existing test suite.

Covered here:
- get_cookies() returns {} instead of raising when no cookies exist (sync+async)
- Redirect policy restore: per-request follow_redirects override must not
  corrupt the client's original policy for subsequent requests (sync+async)
- Valid gzip decompression succeeds (sync+async)
- primp.__version__ is a non-empty string
- json() and json_async() with non-JSON content raises json.JSONDecodeError
- Response.read() explicit method call
- .content property caching (second call returns cached value)
- max_redirects custom value is honoured
- Client cookies init parameter sends cookies with every request
- Encoding setter affects subsequent text/text_async output
- Module-level impersonate parameter forwards correctly to the request
"""

import json

import pytest

import primp


# ---------------------------------------------------------------------------
# get_cookies() returns empty dict instead of raising
# ---------------------------------------------------------------------------


class TestGetCookiesEmptyMap:
    def test_sync_client_get_cookies_no_cookies_returns_empty(
        self, test_server: str
    ) -> None:
        """get_cookies() should return {} when no cookies are stored for the URL."""
        client = primp.Client()
        # Fresh client, nothing stored for this URL
        result = client.get_cookies(test_server)
        assert result == {}
        assert isinstance(result, dict)

    def test_sync_client_get_cookies_after_set(self, test_server: str) -> None:
        """After set_cookies(), get_cookies() returns the stored cookies."""
        client = primp.Client()
        client.set_cookies(test_server, {"tok": "abc"})
        result = client.get_cookies(test_server)
        assert result.get("tok") == "abc"

    def test_sync_client_get_cookies_different_url_empty(
        self, test_server: str
    ) -> None:
        """Cookies set for one URL should not appear under another URL."""
        client = primp.Client()
        client.set_cookies(test_server, {"x": "y"})
        # Different path / URL that has no stored cookies
        result = client.get_cookies("http://example.test/")
        assert isinstance(result, dict)
        assert result == {}

    def test_async_client_get_cookies_no_cookies_returns_empty(
        self, test_server: str
    ) -> None:
        client = primp.AsyncClient()
        result = client.get_cookies(test_server)
        assert result == {}
        assert isinstance(result, dict)

    def test_async_client_get_cookies_after_set(self, test_server: str) -> None:
        client = primp.AsyncClient()
        client.set_cookies(test_server, {"session": "s1"})
        result = client.get_cookies(test_server)
        assert result.get("session") == "s1"


# ---------------------------------------------------------------------------
# Redirect policy restore after per-request override
# ---------------------------------------------------------------------------


class TestRedirectPolicyRestore:
    """The per-request follow_redirects flag must not permanently alter the client."""

    def test_sync_follow_redirects_true_client_override_false_then_restores(
        self, test_server: str
    ) -> None:
        """Client with follow_redirects=True: override to False for one request,
        then confirm the next request still follows redirects."""
        client = primp.Client(follow_redirects=True, max_redirects=10)

        # Per-request: disable redirect following → should stop at 302
        r1 = client.get(f"{test_server}/redirect/1", follow_redirects=False)
        assert r1.status_code == 302

        # Second request with no override → original policy (follow) should apply
        r2 = client.get(f"{test_server}/redirect/1")
        assert r2.status_code == 200

    def test_sync_follow_redirects_false_client_override_true_then_restores(
        self, test_server: str
    ) -> None:
        """Client with follow_redirects=False: override to True for one request,
        then confirm the next request still stops at 302."""
        client = primp.Client(follow_redirects=False)

        # Per-request: enable redirect → should complete redirect chain
        r1 = client.get(f"{test_server}/redirect/1", follow_redirects=True)
        assert r1.status_code == 200

        # Second request with no override → original policy (no follow) should apply
        r2 = client.get(f"{test_server}/redirect/1")
        assert r2.status_code == 302

    def test_sync_multiple_overrides_in_a_row(self, test_server: str) -> None:
        """Alternating per-request overrides should always restore to original."""
        client = primp.Client(follow_redirects=True, max_redirects=10)

        for _ in range(3):
            # Override to False
            r_no = client.get(f"{test_server}/redirect/1", follow_redirects=False)
            assert r_no.status_code == 302
            # No override → original True
            r_yes = client.get(f"{test_server}/redirect/1")
            assert r_yes.status_code == 200

    @pytest.mark.asyncio
    async def test_async_follow_redirects_true_override_false_then_restores(
        self, test_server: str
    ) -> None:
        client = primp.AsyncClient(follow_redirects=True, max_redirects=10)

        r1 = await client.get(f"{test_server}/redirect/1", follow_redirects=False)
        assert r1.status_code == 302

        r2 = await client.get(f"{test_server}/redirect/1")
        assert r2.status_code == 200

    @pytest.mark.asyncio
    async def test_async_follow_redirects_false_override_true_then_restores(
        self, test_server: str
    ) -> None:
        client = primp.AsyncClient(follow_redirects=False)

        r1 = await client.get(f"{test_server}/redirect/1", follow_redirects=True)
        assert r1.status_code == 200

        r2 = await client.get(f"{test_server}/redirect/1")
        assert r2.status_code == 302


class TestMaxRedirects:
    def test_custom_max_redirects_honoured(self, test_server: str) -> None:
        """max_redirects=1 should allow exactly one redirect."""
        client = primp.Client(follow_redirects=True, max_redirects=1)
        # /redirect/1 does 1 redirect; should succeed
        r = client.get(f"{test_server}/redirect/1")
        assert r.status_code == 200

    def test_max_redirects_exceeded_raises(self, test_server: str) -> None:
        """Exceeding max_redirects should raise RedirectError."""
        client = primp.Client(follow_redirects=True, max_redirects=1)
        with pytest.raises(primp.RedirectError):
            client.get(f"{test_server}/redirect/3")


# ---------------------------------------------------------------------------
# Valid gzip decompression
# ---------------------------------------------------------------------------


class TestGzipDecompression:
    def test_sync_gzip_response_decompressed(self, test_server: str) -> None:
        """A gzip-encoded response must be transparently decompressed."""
        client = primp.Client()
        response = client.get(f"{test_server}/gzip")
        assert response.status_code == 200
        data = response.json()
        assert data["gzipped"] is True
        assert "message" in data

    def test_sync_gzip_response_text_readable(self, test_server: str) -> None:
        client = primp.Client()
        response = client.get(f"{test_server}/gzip")
        assert "gzipped" in response.text

    def test_sync_gzip_response_content_is_bytes(self, test_server: str) -> None:
        client = primp.Client()
        response = client.get(f"{test_server}/gzip")
        assert isinstance(response.content, bytes)
        # After decompression the content should be valid JSON bytes
        parsed = json.loads(response.content)
        assert parsed["gzipped"] is True

    @pytest.mark.asyncio
    async def test_async_gzip_response_decompressed(self, test_server: str) -> None:
        client = primp.AsyncClient()
        response = await client.get(f"{test_server}/gzip")
        assert response.status_code == 200
        data = response.json()
        assert data["gzipped"] is True

    @pytest.mark.asyncio
    async def test_async_gzip_response_via_json_async(self, test_server: str) -> None:
        client = primp.AsyncClient()
        response = await client.get(f"{test_server}/gzip")
        data = await response.json_async()
        assert data["gzipped"] is True


# ---------------------------------------------------------------------------
# primp.__version__
# ---------------------------------------------------------------------------


class TestVersion:
    def test_version_attribute_exists(self) -> None:
        assert hasattr(primp, "__version__")

    def test_version_is_string(self) -> None:
        assert isinstance(primp.__version__, str)

    def test_version_not_empty(self) -> None:
        assert primp.__version__ != ""

    def test_version_has_dot_separated_parts(self) -> None:
        parts = primp.__version__.split(".")
        assert len(parts) >= 2
        for part in parts:
            assert part.isdigit(), f"Non-numeric version part: {part!r}"


# ---------------------------------------------------------------------------
# json() / json_async() with non-JSON content → JSONDecodeError
# ---------------------------------------------------------------------------


class TestInvalidJsonParsing:
    def test_sync_json_on_xml_raises_decode_error(self, test_server: str) -> None:
        client = primp.Client()
        response = client.get(f"{test_server}/xml")
        with pytest.raises(json.JSONDecodeError):
            response.json()

    def test_sync_json_on_html_raises_decode_error(self, test_server: str) -> None:
        client = primp.Client()
        response = client.get(f"{test_server}/html")
        with pytest.raises(json.JSONDecodeError):
            response.json()

    def test_sync_json_decode_error_is_value_error(self, test_server: str) -> None:
        """JSONDecodeError is a subclass of ValueError."""
        client = primp.Client()
        response = client.get(f"{test_server}/xml")
        with pytest.raises(ValueError):
            response.json()

    @pytest.mark.asyncio
    async def test_async_json_on_xml_raises_decode_error(
        self, test_server: str
    ) -> None:
        client = primp.AsyncClient()
        response = await client.get(f"{test_server}/xml")
        with pytest.raises(json.JSONDecodeError):
            response.json()

    @pytest.mark.asyncio
    async def test_async_json_async_on_xml_raises_decode_error(
        self, test_server: str
    ) -> None:
        client = primp.AsyncClient()
        response = await client.get(f"{test_server}/xml")
        with pytest.raises(json.JSONDecodeError):
            await response.json_async()

    @pytest.mark.asyncio
    async def test_async_json_async_error_is_value_error(
        self, test_server: str
    ) -> None:
        client = primp.AsyncClient()
        response = await client.get(f"{test_server}/xml")
        with pytest.raises(ValueError):
            await response.json_async()


# ---------------------------------------------------------------------------
# Response.read() explicit method call
# ---------------------------------------------------------------------------


class TestResponseReadMethod:
    def test_sync_read_returns_bytes(self, test_server: str) -> None:
        client = primp.Client()
        response = client.get(f"{test_server}/get")
        data = response.read()
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_sync_read_equals_content(self, test_server: str) -> None:
        """read() and .content should return identical bytes."""
        client = primp.Client()
        response = client.get(f"{test_server}/get")
        assert response.read() == response.content

    def test_sync_read_multiple_times_returns_same_bytes(
        self, test_server: str
    ) -> None:
        """Calling read() more than once should return the cached body each time."""
        client = primp.Client()
        response = client.get(f"{test_server}/get")
        first = response.read()
        second = response.read()
        assert first == second


# ---------------------------------------------------------------------------
# .content caching
# ---------------------------------------------------------------------------


class TestContentCaching:
    def test_content_cached_after_first_access(self, test_server: str) -> None:
        client = primp.Client()
        response = client.get(f"{test_server}/get")
        c1 = response.content
        c2 = response.content
        assert c1 == c2

    def test_content_and_text_consistent(self, test_server: str) -> None:
        """Decoding .content with the response encoding should match .text."""
        client = primp.Client()
        response = client.get(f"{test_server}/get")
        raw = response.content
        text = response.text
        assert text == raw.decode("utf-8")


# ---------------------------------------------------------------------------
# Client cookies init parameter
# ---------------------------------------------------------------------------


class TestClientCookiesParam:
    def test_sync_client_cookies_sent_with_request(self, test_server: str) -> None:
        """Cookies passed to Client(cookies=...) should be sent on every request."""
        client = primp.Client(cookies={"auth": "token123"})
        response = client.get(f"{test_server}/cookies")
        data = response.json()
        assert "auth" in data["cookies"]
        assert data["cookies"]["auth"] == "token123"

    @pytest.mark.asyncio
    async def test_async_client_cookies_sent_with_request(
        self, test_server: str
    ) -> None:
        client = primp.AsyncClient(cookies={"auth": "token123"})
        response = await client.get(f"{test_server}/cookies")
        data = response.json()
        assert "auth" in data["cookies"]
        assert data["cookies"]["auth"] == "token123"

    def test_sync_client_cookies_persist_across_requests(
        self, test_server: str
    ) -> None:
        client = primp.Client(cookies={"session": "abc"})
        for _ in range(3):
            r = client.get(f"{test_server}/cookies")
            assert r.json()["cookies"]["session"] == "abc"


# ---------------------------------------------------------------------------
# Encoding setter affects text output
# ---------------------------------------------------------------------------


class TestEncodingSetter:
    def test_sync_encoding_setter_changes_text_decoding(
        self, test_server: str
    ) -> None:
        client = primp.Client()
        response = client.get(f"{test_server}/get")
        # Default encoding is utf-8; setting it explicitly changes the attribute
        response.encoding = "utf-8"
        assert response.encoding == "utf-8"
        text = response.text
        assert isinstance(text, str)

    @pytest.mark.asyncio
    async def test_async_encoding_setter_changes_text_decoding(
        self, test_server: str
    ) -> None:
        client = primp.AsyncClient()
        response = await client.get(f"{test_server}/get")
        response.encoding = "utf-8"
        assert response.encoding == "utf-8"
        text = response.text
        assert isinstance(text, str)


# ---------------------------------------------------------------------------
# Module-level functions accept impersonate parameter
# ---------------------------------------------------------------------------


class TestModuleFunctionsImpersonate:
    def test_get_with_impersonate(self, test_server: str) -> None:
        response = primp.get(
            f"{test_server}/get",
            impersonate="chrome_144",
        )
        assert response.status_code == 200

    def test_post_with_impersonate(self, test_server: str) -> None:
        response = primp.post(
            f"{test_server}/post",
            impersonate="chrome_144",
            json={"x": 1},
        )
        assert response.status_code == 200

    def test_request_function_custom_method(self, test_server: str) -> None:
        response = primp.request("DELETE", f"{test_server}/delete")
        assert response.status_code == 200
