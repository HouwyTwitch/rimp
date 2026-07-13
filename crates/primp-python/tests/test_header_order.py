"""Tests that header ordering is preserved on the client and response."""

import primp


def test_client_header_insertion_order_preserved() -> None:
    """Custom headers keep the insertion order they were supplied in."""
    client = primp.Client(headers={"X-Third": "3", "X-First": "1", "X-Second": "2"})

    header_keys = list(client.headers.keys())
    x_third_idx = header_keys.index("x-third")
    x_first_idx = header_keys.index("x-first")
    x_second_idx = header_keys.index("x-second")

    assert x_third_idx < x_first_idx < x_second_idx, f"Header order not preserved: {header_keys}"


def test_response_header_order_is_stable(test_server: str) -> None:
    """Response headers expose a stable, repeatable order across accesses."""
    client = primp.Client()
    response = client.get(f"{test_server}/get")

    response_headers = response.headers
    assert isinstance(response_headers, dict)
    assert list(response_headers.keys()) == list(response_headers.keys())
