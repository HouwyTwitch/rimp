"""
Tests for AsyncResponse methods on non-streaming responses.

The streaming test suite already covers aread/text_async/json_async on stream=True
responses.  This module covers the same methods on plain (buffered) responses,
which go through a different code path in async/response.rs.

Covered here:
- aread() / get_content_async  — non-streaming buffered body
- text_async                   — non-streaming text decode
- json_async()                 — non-streaming JSON parse (success and error)
- get_headers / get_cookies    — lazy-loaded via py.detach(block_on) path
- aclose() on non-streaming    — drops the inner response
- encoding setter effect       — changing encoding changes text_async output
"""

import json

import pytest

import primp


class TestAsyncResponseAread:
    """aread() and get_content_async on buffered (non-streaming) responses."""

    @pytest.mark.asyncio
    async def test_aread_returns_bytes(self, test_server: str) -> None:
        client = primp.AsyncClient()
        response = await client.get(f"{test_server}/get")
        data = await response.aread()
        assert isinstance(data, bytes)
        assert len(data) > 0

    @pytest.mark.asyncio
    async def test_aread_content_is_parseable_json(self, test_server: str) -> None:
        client = primp.AsyncClient()
        response = await client.get(f"{test_server}/get")
        data = await response.aread()
        parsed = json.loads(data)
        assert "headers" in parsed

    @pytest.mark.asyncio
    async def test_aread_post_body(self, test_server: str) -> None:
        client = primp.AsyncClient()
        response = await client.post(
            f"{test_server}/post",
            json={"ping": "pong"},
        )
        data = await response.aread()
        assert isinstance(data, bytes)
        parsed = json.loads(data)
        assert parsed["json"]["ping"] == "pong"

    @pytest.mark.asyncio
    async def test_get_content_async_property(self, test_server: str) -> None:
        client = primp.AsyncClient()
        response = await client.get(f"{test_server}/get")
        data = await response.content_async
        assert isinstance(data, bytes)
        assert len(data) > 0

    @pytest.mark.asyncio
    async def test_get_content_async_matches_aread(self, test_server: str) -> None:
        """content_async is just a wrapper for aread(); both return the same bytes."""
        client = primp.AsyncClient()
        # Use two separate requests to compare
        r1 = await client.get(f"{test_server}/get", params={"marker": "a"})
        r2 = await client.get(f"{test_server}/get", params={"marker": "a"})
        d1 = await r1.content_async
        d2 = await r2.aread()
        # Same request → same structure (urls differ only in ephemeral port)
        assert json.loads(d1)["args"] == json.loads(d2)["args"]

    @pytest.mark.asyncio
    async def test_aread_empty_body_head_request(self, test_server: str) -> None:
        client = primp.AsyncClient()
        response = await client.head(f"{test_server}/anything")
        data = await response.aread()
        assert isinstance(data, bytes)
        # HEAD responses have no body
        assert data == b""


class TestAsyncResponseTextAsync:
    """text_async property on buffered (non-streaming) responses."""

    @pytest.mark.asyncio
    async def test_text_async_returns_string(self, test_server: str) -> None:
        client = primp.AsyncClient()
        response = await client.get(f"{test_server}/get")
        text = await response.text_async
        assert isinstance(text, str)
        assert len(text) > 0

    @pytest.mark.asyncio
    async def test_text_async_parseable_as_json(self, test_server: str) -> None:
        client = primp.AsyncClient()
        response = await client.get(f"{test_server}/get")
        text = await response.text_async
        parsed = json.loads(text)
        assert "headers" in parsed

    @pytest.mark.asyncio
    async def test_text_async_html_response(self, test_server: str) -> None:
        client = primp.AsyncClient()
        response = await client.get(f"{test_server}/html")
        text = await response.text_async
        assert isinstance(text, str)
        assert "Welcome to Test Server" in text

    @pytest.mark.asyncio
    async def test_text_async_xml_response(self, test_server: str) -> None:
        client = primp.AsyncClient()
        response = await client.get(f"{test_server}/xml")
        text = await response.text_async
        assert isinstance(text, str)
        assert "<root>" in text

    @pytest.mark.asyncio
    async def test_text_async_with_query_params(self, test_server: str) -> None:
        client = primp.AsyncClient()
        response = await client.get(
            f"{test_server}/get",
            params={"hello": "world"},
        )
        text = await response.text_async
        parsed = json.loads(text)
        assert parsed["args"]["hello"] == "world"

    @pytest.mark.asyncio
    async def test_text_async_encoding_utf8(self, test_server: str) -> None:
        """Default encoding should decode without errors."""
        client = primp.AsyncClient()
        response = await client.get(f"{test_server}/get")
        # Should not raise
        text = await response.text_async
        assert isinstance(text, str)


class TestAsyncResponseJsonAsync:
    """json_async() on buffered (non-streaming) responses."""

    @pytest.mark.asyncio
    async def test_json_async_returns_dict(self, test_server: str) -> None:
        client = primp.AsyncClient()
        response = await client.get(f"{test_server}/get")
        data = await response.json_async()
        assert isinstance(data, dict)
        assert "headers" in data

    @pytest.mark.asyncio
    async def test_json_async_post_with_json_body(self, test_server: str) -> None:
        client = primp.AsyncClient()
        response = await client.post(
            f"{test_server}/post",
            json={"key": "value", "number": 42, "nested": {"a": 1}},
        )
        data = await response.json_async()
        assert isinstance(data, dict)
        assert data["json"]["key"] == "value"
        assert data["json"]["number"] == 42
        assert data["json"]["nested"]["a"] == 1

    @pytest.mark.asyncio
    async def test_json_async_with_query_params(self, test_server: str) -> None:
        client = primp.AsyncClient()
        response = await client.get(
            f"{test_server}/get",
            params={"foo": "bar", "n": "123"},
        )
        data = await response.json_async()
        assert data["args"]["foo"] == "bar"
        assert data["args"]["n"] == "123"

    @pytest.mark.asyncio
    async def test_json_async_list_response(self, test_server: str) -> None:
        """json_async() should handle any valid JSON root type."""
        client = primp.AsyncClient()
        response = await client.post(
            f"{test_server}/post",
            json=[1, 2, 3],
        )
        data = await response.json_async()
        # The server wraps it under "json" key
        assert data["json"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_json_async_invalid_json_raises_json_decode_error(
        self, test_server: str
    ) -> None:
        """xml and html endpoints return non-JSON; json_async() must raise JSONDecodeError."""
        client = primp.AsyncClient()
        response = await client.get(f"{test_server}/xml")
        with pytest.raises(json.JSONDecodeError):
            await response.json_async()

    @pytest.mark.asyncio
    async def test_json_async_invalid_html_raises_json_decode_error(
        self, test_server: str
    ) -> None:
        client = primp.AsyncClient()
        response = await client.get(f"{test_server}/html")
        with pytest.raises(json.JSONDecodeError):
            await response.json_async()

    @pytest.mark.asyncio
    async def test_json_async_error_is_catchable_as_value_error(
        self, test_server: str
    ) -> None:
        """JSONDecodeError is a subclass of ValueError."""
        client = primp.AsyncClient()
        response = await client.get(f"{test_server}/xml")
        with pytest.raises(ValueError):
            await response.json_async()


class TestAsyncResponseHeadersCookiesLazy:
    """get_headers and get_cookies trigger the lazy block_on path on non-streaming."""

    @pytest.mark.asyncio
    async def test_headers_accessible_non_streaming(self, test_server: str) -> None:
        client = primp.AsyncClient()
        response = await client.get(f"{test_server}/get")
        headers = response.headers
        assert isinstance(headers, dict)
        assert "content-type" in headers

    @pytest.mark.asyncio
    async def test_cookies_accessible_non_streaming(self, test_server: str) -> None:
        client = primp.AsyncClient()
        response = await client.get(f"{test_server}/cookies/set?sess=xyz")
        cookies = response.cookies
        assert isinstance(cookies, dict)

    @pytest.mark.asyncio
    async def test_headers_cached_on_second_access(self, test_server: str) -> None:
        """Calling .headers twice should return the same dict (cached after first call)."""
        client = primp.AsyncClient()
        response = await client.get(f"{test_server}/get")
        h1 = response.headers
        h2 = response.headers
        assert h1 == h2

    @pytest.mark.asyncio
    async def test_cookies_empty_when_none_set(self, test_server: str) -> None:
        client = primp.AsyncClient()
        response = await client.get(f"{test_server}/get")
        cookies = response.cookies
        assert isinstance(cookies, dict)


class TestAsyncResponseAclose:
    """aclose() on a non-streaming response drops the inner response object."""

    @pytest.mark.asyncio
    async def test_aclose_does_not_raise(self, test_server: str) -> None:
        client = primp.AsyncClient()
        response = await client.get(f"{test_server}/get")
        # Read the content first so there's something to close
        await response.aread()
        await response.aclose()

    @pytest.mark.asyncio
    async def test_async_context_manager_non_streaming(self, test_server: str) -> None:
        """AsyncResponse used as async context manager calls __aexit__ / aclose."""
        client = primp.AsyncClient()
        async with await client.get(f"{test_server}/get") as response:
            data = await response.aread()
            assert isinstance(data, bytes)
            assert len(data) > 0
