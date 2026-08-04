use std::collections::BTreeMap;
use std::sync::Once;

use anyhow::{anyhow, Result};
pub use primp::imp::{Impersonate, ImpersonateOS};
use primp::ProfileOverrides;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList, PyString, PyTuple};
use rand::prelude::*;

/// Available OS impersonation options.
pub const IMPERSONATEOS_LIST: &[ImpersonateOS] = &[
    ImpersonateOS::Android,
    ImpersonateOS::IOS,
    ImpersonateOS::Linux,
    ImpersonateOS::MacOS,
    ImpersonateOS::Windows,
];

/// One-time flags for warnings
static IMPERSONATE_WARNING: Once = Once::new();
static IMPERSONATE_OS_WARNING: Once = Once::new();

/// Select a random element from a slice.
pub fn get_random_element<T>(slice: &[T]) -> &T {
    slice.choose(&mut rand::rng()).unwrap()
}

/// Parse a string into an Impersonate variant.
pub fn parse_impersonate(s: &str) -> Result<Impersonate> {
    match s {
        // Chrome variants
        "chrome_144" => Ok(Impersonate::ChromeV144),
        "chrome_145" => Ok(Impersonate::ChromeV145),
        "chrome_146" => Ok(Impersonate::ChromeV146),
        "chrome_147" => Ok(Impersonate::ChromeV147),
        "chrome_148" => Ok(Impersonate::ChromeV148),
        "chrome_150" => Ok(Impersonate::ChromeV150),
        "chrome_150_0_7871_187" => Ok(Impersonate::ChromeV150_0_7871_187),
        "chrome" => Ok(Impersonate::Chrome),
        // Edge variants
        "edge_144" => Ok(Impersonate::EdgeV144),
        "edge_145" => Ok(Impersonate::EdgeV145),
        "edge_146" => Ok(Impersonate::EdgeV146),
        "edge_147" => Ok(Impersonate::EdgeV147),
        "edge_148" => Ok(Impersonate::EdgeV148),
        "edge" => Ok(Impersonate::Edge),
        // Opera variants
        "opera_126" => Ok(Impersonate::OperaV126),
        "opera_127" => Ok(Impersonate::OperaV127),
        "opera_128" => Ok(Impersonate::OperaV128),
        "opera_129" => Ok(Impersonate::OperaV129),
        "opera_130" => Ok(Impersonate::OperaV130),
        "opera_131" => Ok(Impersonate::OperaV131),
        "opera" => Ok(Impersonate::Opera),
        // Safari variants
        "safari_18.5" => Ok(Impersonate::SafariV18_5),
        "safari_26" => Ok(Impersonate::SafariV26),
        "safari_26.3" => Ok(Impersonate::SafariV26_3),
        "safari" => Ok(Impersonate::Safari),
        // Firefox variants
        "firefox_140" => Ok(Impersonate::FirefoxV140),
        "firefox_146" => Ok(Impersonate::FirefoxV146),
        "firefox_147" => Ok(Impersonate::FirefoxV147),
        "firefox_148" => Ok(Impersonate::FirefoxV148),
        "firefox" => Ok(Impersonate::Firefox),
        // Random selection
        "random" => Ok(Impersonate::Random),
        _ => Err(anyhow!("Invalid impersonate: {:?}", s)),
    }
}

/// Parse a string into an ImpersonateOS variant.
pub fn parse_impersonate_os(s: &str) -> Result<ImpersonateOS> {
    match s {
        "android" => Ok(ImpersonateOS::Android),
        "ios" => Ok(ImpersonateOS::IOS),
        "linux" => Ok(ImpersonateOS::Linux),
        "macos" => Ok(ImpersonateOS::MacOS),
        "windows" => Ok(ImpersonateOS::Windows),
        "random" => Ok(*get_random_element(IMPERSONATEOS_LIST)),
        _ => Err(anyhow!("Invalid impersonate_os: {:?}", s)),
    }
}

/// Parse a string into an Impersonate variant with fallback to random value.
/// If the provided value doesn't exist, logs a one-time warning and returns a random valid value.
pub fn parse_impersonate_with_fallback(s: &str) -> Impersonate {
    parse_impersonate(s).unwrap_or_else(|_| {
        IMPERSONATE_WARNING.call_once(|| {
            tracing::warn!("Impersonate '{}' does not exist, using 'random'", s);
        });
        Impersonate::Random
    })
}

/// Parse a string into an ImpersonateOS variant with fallback to random value.
/// If the provided value doesn't exist, logs a one-time warning and returns a random valid value.
pub fn parse_impersonate_os_with_fallback(s: &str) -> ImpersonateOS {
    parse_impersonate_os(s).unwrap_or_else(|_| {
        IMPERSONATE_OS_WARNING.call_once(|| {
            tracing::warn!("Impersonate OS '{}' does not exist, using 'random'", s);
        });
        *get_random_element(IMPERSONATEOS_LIST)
    })
}

/// Convert a Python dict into a [`ProfileOverrides`].
///
/// Recognised keys (all optional):
///
/// TLS: `cipher_suites: list[int|str]`, `signature_algorithms: list[int|str]`,
///      `named_groups: list[int|str]`, `extension_order_seed: int`.
///
/// HTTP-level: `user_agent: str`, `sec_ch_ua: str`, `sec_ch_ua_platform: str`,
///      `sec_ch_ua_mobile: str`, `headers: dict[str, str]`,
///      `headers_order: list[str]`.
///
/// HTTP/2: `http2_settings: list[(int, int)]`, `http2_pseudo_order: str|list[str]`
///      (e.g. `"masp"` or `["m", "a", "s", "p"]`),
///      `http2_headers_priority: tuple[int, int, bool] | None`,
///      `http2_initial_stream_window_size: int`,
///      `http2_initial_connection_window_size: int`,
///      `http2_header_table_size: int`, `http2_max_header_list_size: int`.
///
/// Peet.ws convenience: `peet_api_response: dict` (full `/api/clean` JSON),
///      or the two strings `peetprint: str` + `akamai: str`. Explicit keys
///      set alongside these win.
pub fn parse_profile_overrides(dict: &Bound<'_, PyDict>) -> PyResult<ProfileOverrides> {
    let mut ov = ProfileOverrides::default();

    // First: seed from peet.ws payload if present.
    let (peetprint, akamai) = extract_peet_strings(dict)?;
    if let (Some(pp), Some(ak)) = (peetprint.as_deref(), akamai.as_deref()) {
        ov = ProfileOverrides::from_peet_api(pp, ak)
            .map_err(|e| PyValueError::new_err(format!("invalid peet fingerprint: {e}")))?;
    }

    if let Some(v) = get_str(dict, "user_agent")? {
        ov.user_agent = Some(v);
    }
    if let Some(v) = get_str(dict, "sec_ch_ua")? {
        ov.sec_ch_ua = Some(v);
    }
    if let Some(v) = get_str(dict, "sec_ch_ua_platform")? {
        ov.sec_ch_ua_platform = Some(v);
    }
    if let Some(v) = get_str(dict, "sec_ch_ua_mobile")? {
        ov.sec_ch_ua_mobile = Some(v);
    }

    if let Some(item) = dict.get_item("headers")? {
        let h: BTreeMap<String, String> = item.extract()?;
        ov.extra_headers.extend(h);
    }
    if let Some(item) = dict.get_item("headers_order")? {
        let list: Vec<String> = item.extract()?;
        ov.headers_order = Some(list);
    }

    if let Some(v) = get_u16_list(dict, "cipher_suites")? {
        ov.cipher_suites = Some(v);
    }
    if let Some(v) = get_u16_list(dict, "signature_algorithms")? {
        ov.signature_algorithms = Some(v);
    }
    if let Some(v) = get_u16_list(dict, "named_groups")? {
        ov.named_groups = Some(v);
    }
    if let Some(item) = dict.get_item("extension_order_seed")? {
        ov.extension_order_seed = Some(item.extract::<u16>()?);
    }

    if let Some(item) = dict.get_item("http2_settings")? {
        let list = item
            .cast::<PyList>()
            .map_err(|_| PyValueError::new_err("http2_settings must be a list"))?;
        let mut out = Vec::with_capacity(list.len());
        for entry in list.iter() {
            let (id, value): (u16, u32) = entry.extract()?;
            out.push((id, value));
        }
        ov.http2_settings = Some(out);
    }
    if let Some(item) = dict.get_item("http2_pseudo_order")? {
        let chars: Vec<char> = if let Ok(s) = item.cast::<PyString>() {
            s.to_cow()?.chars().collect()
        } else {
            let list = item
                .cast::<PyList>()
                .map_err(|_| PyValueError::new_err("http2_pseudo_order must be str or list"))?;
            let mut chars = Vec::with_capacity(list.len());
            for entry in list.iter() {
                let s: String = entry.extract()?;
                if let Some(c) = s.chars().next() {
                    chars.push(c);
                }
            }
            chars
        };
        ov.http2_pseudo_order = Some(chars);
    }
    if let Some(item) = dict.get_item("http2_headers_priority")? {
        if item.is_none() {
            ov.http2_headers_priority = Some(None);
        } else {
            let tup = item
                .cast::<PyTuple>()
                .map_err(|_| PyValueError::new_err("http2_headers_priority must be a tuple"))?;
            if tup.len() != 3 {
                return Err(PyValueError::new_err(
                    "http2_headers_priority must be (weight, dep, exclusive)",
                ));
            }
            let weight: u8 = tup.get_item(0)?.extract()?;
            let dep: u32 = tup.get_item(1)?.extract()?;
            let exclusive: bool = tup
                .get_item(2)?
                .cast::<PyBool>()
                .map_err(|_| PyValueError::new_err("http2_headers_priority[2] must be bool"))?
                .is_true();
            ov.http2_headers_priority = Some(Some((weight, dep, exclusive)));
        }
    }
    for (dict_key, field) in [
        (
            "http2_initial_stream_window_size",
            &mut ov.http2_initial_stream_window_size,
        ),
        (
            "http2_initial_connection_window_size",
            &mut ov.http2_initial_connection_window_size,
        ),
        ("http2_header_table_size", &mut ov.http2_header_table_size),
        ("http2_max_header_list_size", &mut ov.http2_max_header_list_size),
    ] {
        if let Some(item) = dict.get_item(dict_key)? {
            *field = Some(item.extract::<u32>()?);
        }
    }

    Ok(ov)
}

fn extract_peet_strings(dict: &Bound<'_, PyDict>) -> PyResult<(Option<String>, Option<String>)> {
    if let Some(resp) = dict.get_item("peet_api_response")? {
        let resp = resp
            .cast::<PyDict>()
            .map_err(|_| PyValueError::new_err("peet_api_response must be a dict"))?;
        let pp = get_str(resp, "peetprint")?;
        let ak = get_str(resp, "akamai")?;
        if pp.is_some() && ak.is_some() {
            return Ok((pp, ak));
        }
    }
    Ok((get_str(dict, "peetprint")?, get_str(dict, "akamai")?))
}

fn get_str(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
    match dict.get_item(key)? {
        Some(item) => Ok(Some(item.extract::<String>()?)),
        None => Ok(None),
    }
}

/// Accept a list of ints, or a list of strings such as "0x0904" / "2308".
/// Symbolic cipher/algorithm names are not resolved here — pass numeric IDs,
/// or strings prefixed with `0x` for hex.
fn get_u16_list(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<Vec<u16>>> {
    let Some(item) = dict.get_item(key)? else {
        return Ok(None);
    };
    let list = item
        .cast::<PyList>()
        .map_err(|_| PyValueError::new_err(format!("{key} must be a list")))?;
    let mut out = Vec::with_capacity(list.len());
    for entry in list.iter() {
        if let Ok(n) = entry.extract::<u16>() {
            out.push(n);
            continue;
        }
        let s: String = entry.extract()?;
        let s = s.trim();
        let n = if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
            u16::from_str_radix(hex, 16)
                .map_err(|_| PyValueError::new_err(format!("invalid hex u16: {s:?}")))?
        } else {
            s.parse::<u16>()
                .map_err(|_| PyValueError::new_err(format!("invalid u16: {s:?}")))?
        };
        out.push(n);
    }
    Ok(Some(out))
}
