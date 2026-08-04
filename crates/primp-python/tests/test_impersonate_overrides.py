"""Offline tests for the impersonate_overrides mechanism."""

import pytest

import primp

# A representative tls.peet.ws /api/all response (Chrome 151 on Windows).
SAMPLE_API_ALL = {
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36",
    "tls": {
        "peetprint": "GREASE-772-771|2-1.1|GREASE-4588-29-23-24|2308-2309-2310-1027-2052-1025-1283-2053-1281-2054-1537|1|2|GREASE-4865-4866-4867-49195-49199-49196-49200-52393-52392-49171-49172-156-157-47-53|0-10-11-13-16-17613-18-23-27-35-41-43-45-5-51-65037-65281-GREASE-GREASE",
    },
    "http2": {
        "akamai_fingerprint": "1:65536;2:0;4:6291456;6:262144|15663105|0|m,a,s,p",
        "sent_frames": [
            {"frame_type": "SETTINGS", "settings": []},
            {"frame_type": "WINDOW_UPDATE", "increment": 15663105},
            {
                "frame_type": "HEADERS",
                "stream_id": 1,
                "headers": [
                    ":method: GET",
                    ":authority: tls.peet.ws",
                    ":scheme: https",
                    ":path: /api/all",
                    'sec-ch-ua: "Not=A?Brand";v="99", "Google Chrome";v="151", "Chromium";v="151"',
                    "sec-ch-ua-mobile: ?0",
                    'sec-ch-ua-platform: "Windows"',
                    "upgrade-insecure-requests: 1",
                    "user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36",
                    "accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                    "sec-fetch-site: none",
                    "sec-fetch-mode: navigate",
                    "sec-fetch-user: ?1",
                    "sec-fetch-dest: document",
                    "accept-encoding: gzip, deflate, br, zstd",
                    "accept-language: ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
                    "priority: u=0, i",
                ],
                "priority": {"weight": 256, "depends_on": 0, "exclusive": 1},
            },
        ],
    },
}


def test_client_accepts_api_all_response():
    """Passing a /api/all dict should build a Client without error and pick up
    the observed User-Agent."""
    client = primp.Client(
        impersonate="chrome_150",
        impersonate_overrides={"peet_api_response": SAMPLE_API_ALL},
    )
    ua = client.headers["user-agent"]
    assert "Chrome/151.0.0.0" in ua
    assert client.headers["sec-ch-ua-platform"] == '"Windows"'
    assert client.headers["accept-language"].startswith("ru-RU")


def test_explicit_overrides_win_over_peet_response():
    client = primp.Client(
        impersonate="chrome_150",
        impersonate_overrides={
            "peet_api_response": SAMPLE_API_ALL,
            "user_agent": "custom-agent/1.0",
            "headers": {"accept-language": "en-US,en;q=0.9"},
        },
    )
    assert client.headers["user-agent"] == "custom-agent/1.0"
    assert client.headers["accept-language"] == "en-US,en;q=0.9"


def test_api_clean_response_still_works():
    """Backwards compatibility: /api/clean format (flat peetprint+akamai)."""
    api_clean = {
        "peetprint": SAMPLE_API_ALL["tls"]["peetprint"],
        "akamai": SAMPLE_API_ALL["http2"]["akamai_fingerprint"],
    }
    client = primp.Client(
        impersonate="chrome_150",
        impersonate_os="windows",
        impersonate_overrides={"peet_api_response": api_clean},
    )
    # Base profile's UA (Chrome 150) is unchanged since /api/clean carries none.
    assert "Chrome/150" in client.headers["user-agent"]


def test_standalone_peetprint_and_akamai_strings():
    client = primp.Client(
        impersonate="chrome_150",
        impersonate_overrides={
            "peetprint": SAMPLE_API_ALL["tls"]["peetprint"],
            "akamai": SAMPLE_API_ALL["http2"]["akamai_fingerprint"],
            "user_agent": "Mozilla/5.0 ... Chrome/151.0.0.0 Safari/537.36",
        },
    )
    assert "Chrome/151.0.0.0" in client.headers["user-agent"]


def test_purely_manual_overrides():
    client = primp.Client(
        impersonate="chrome_150",
        impersonate_overrides={
            "signature_algorithms": [0x0904, 0x0905, 0x0906, 0x0403],
            "http2_settings": [(1, 65536), (2, 0), (4, 6291456), (6, 262144)],
            "http2_pseudo_order": "masp",
            "http2_headers_priority": (255, 0, True),
            "headers_order": ["sec-ch-ua", "user-agent", "accept-encoding"],
        },
    )
    # Nothing to assert about wire behaviour without hitting a real fingerprint
    # endpoint; smoke test verifies construction succeeds.
    assert client is not None


@pytest.mark.parametrize("bad", ["not-a-dict", 42, ["a", "b"]])
def test_invalid_impersonate_overrides_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        primp.Client(impersonate_overrides=bad)  # type: ignore[arg-type]
