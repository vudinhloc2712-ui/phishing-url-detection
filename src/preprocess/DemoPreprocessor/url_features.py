"""Chuyển một URL thô thành 87 thuộc tính của Phishing URL Dataset (dùng cho DEMO).

Gồm: (1) trích xuất 87 đặc trưng (code gốc của nhóm), (2) điền giá trị dự phòng cho các cột
không tính được (mục 7b) để không còn NaN trước khi đưa vào DemoPreprocessor.
"""
from __future__ import annotations

import argparse
import ipaddress
import math
import os
import re
import socket
import sys
from datetime import datetime, timezone
from typing import Iterable
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# 1. Danh sách cột / Column layout (đúng thứ tự trong dataset_phishing.csv)
# --------------------------------------------------------------------------- #

URL_FEATURES = [
    "length_url", "length_hostname", "ip", "nb_dots", "nb_hyphens", "nb_at",
    "nb_qm", "nb_and", "nb_or", "nb_eq", "nb_underscore", "nb_tilde",
    "nb_percent", "nb_slash", "nb_star", "nb_colon", "nb_comma",
    "nb_semicolumn", "nb_dollar", "nb_space", "nb_www", "nb_com", "nb_dslash",
    "http_in_path", "https_token", "ratio_digits_url", "ratio_digits_host",
    "punycode", "port", "tld_in_path", "tld_in_subdomain", "abnormal_subdomain",
    "nb_subdomains", "prefix_suffix", "random_domain", "shortening_service",
    "path_extension",
    # (nb_redirection, nb_external_redirection nằm ở nhóm HTML)
    "length_words_raw", "char_repeat", "shortest_words_raw",
    "shortest_word_host", "shortest_word_path", "longest_words_raw",
    "longest_word_host", "longest_word_path", "avg_words_raw", "avg_word_host",
    "avg_word_path", "phish_hints", "domain_in_brand", "brand_in_subdomain",
    "brand_in_path", "suspecious_tld", "statistical_report",
]

REDIRECTION_FEATURES = ["nb_redirection", "nb_external_redirection"]

CONTENT_FEATURES = [
    "nb_hyperlinks", "ratio_intHyperlinks", "ratio_extHyperlinks",
    "ratio_nullHyperlinks", "nb_extCSS", "ratio_intRedirection",
    "ratio_extRedirection", "ratio_intErrors", "ratio_extErrors", "login_form",
    "external_favicon", "links_in_tags", "submit_email", "ratio_intMedia",
    "ratio_extMedia", "sfh", "iframe", "popup_window", "safe_anchor",
    "onmouseover", "right_clic", "empty_title", "domain_in_title",
    "domain_with_copyright",
]

EXTERNAL_FEATURES = [
    "whois_registered_domain", "domain_registration_length", "domain_age",
    "web_traffic", "dns_record", "google_index", "page_rank",
]

HTML_FEATURES = REDIRECTION_FEATURES + CONTENT_FEATURES

# Thứ tự gốc của dataset: redirection chen giữa path_extension và length_words_raw
ALL_FEATURES = (
    URL_FEATURES[: URL_FEATURES.index("path_extension") + 1]
    + REDIRECTION_FEATURES
    + URL_FEATURES[URL_FEATURES.index("path_extension") + 1:]
    + CONTENT_FEATURES
    + EXTERNAL_FEATURES
)
assert len(URL_FEATURES) == 54 and len(HTML_FEATURES) == 26
assert len(EXTERNAL_FEATURES) == 7 and len(ALL_FEATURES) == 87

LEVELS = {"url": 1, "html": 2, "full": 3}


def feature_columns(level: str = "url") -> list[str]:
    """Danh sách cột theo thứ tự dataset cho `level` đã chọn."""
    if level not in LEVELS:
        raise ValueError(f"level phải thuộc {list(LEVELS)}, nhận '{level}'")
    keep = set(URL_FEATURES)
    if LEVELS[level] >= 2:
        keep |= set(HTML_FEATURES)
    if LEVELS[level] >= 3:
        keep |= set(EXTERNAL_FEATURES)
    return [c for c in ALL_FEATURES if c in keep]


# --------------------------------------------------------------------------- #
# 2. Hằng số tra cứu / Lookup tables
# --------------------------------------------------------------------------- #

SPLIT_RE = re.compile(r"[-./?=@&%:_]")
SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")

PHISH_HINTS = ["wp", "login", "includes", "admin", "content", "site", "images",
               "js", "alibaba", "css", "myaccount", "dropbox", "themes",
               "plugins", "signin", "view"]

SUSPICIOUS_TLDS = {
    "fit", "tk", "gp", "ga", "work", "ml", "date", "wang", "men", "icu",
    "online", "click", "country", "stream", "download", "xin", "racing",
    "jetzt", "bid", "loan", "top", "cf", "gq", "xyz", "cn", "support", "zip",
    "review", "trade", "party", "science", "cricket", "accountant", "faith",
}

# Regex dịch vụ rút gọn link, tìm KIẾM CHUỖI CON trong toàn bộ URL (đúng cách
# dataset gốc làm, nên "t.co" cũng khớp "net.com"). Đã chỉnh theo kết quả --validate.
SHORTENING_RE = re.compile(
    r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|"
    r"is\.gd|cli\.gs|yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|"
    r"su\.pr|twurl\.nl|snipurl\.com|short\.to|BudURL\.com|ping\.fm|post\.ly|"
    r"Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|doiop\.com|short\.ie|"
    r"kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|lnkd\.in|db\.tt|qr\.ae|"
    r"adf\.ly|bitly\.com|cur\.lv|ity\.im|q\.gs|po\.st|bc\.vc|twitthis\.com|"
    r"u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|prettylinkpro\.com|"
    r"scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|"
    r"link\.zip\.net")

# statistical_report: bản gốc còn so địa chỉ IP với danh sách IP hay bị báo cáo
# (cần DNS); ở đây chỉ dùng phần tên miền, không cần mạng.
REPORTED_RE = re.compile(
    r"at\.ua|usa\.cc|baltazarpresentes\.com\.br|pe\.hu|esy\.es|hol\.es|"
    r"sweddy\.com|myjino\.ru|96\.lt|ow\.ly")

# ip: bản gốc tìm IPv4 (theo sau là "/"), IPv4 dạng hex, IPv6 hoặc 7 ký tự hex liền nhau
# ở BẤT KỲ đâu trong URL (nên cột này ~15% bằng 1 chứ không chỉ các URL dùng địa chỉ IP)
_OCT = r"([01]?\d\d?|2[0-4]\d|25[0-5])"
IP_RE = re.compile(
    rf"({_OCT}\.{_OCT}\.{_OCT}\.{_OCT}/)|"
    r"((0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})/)|"
    r"((?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4})|"
    r"[0-9a-fA-F]{7}")  # bản gốc coi 7 ký tự hex liền nhau là "IP" (phát hiện nhờ --diff ip)

# Bộ thương hiệu rút gọn. Bản gốc dùng ~2000 thương hiệu; nạp danh sách đầy đủ
# bằng `--brands brands.txt` (mỗi dòng một thương hiệu, viết thường).
BRANDS = {
    "paypal", "apple", "microsoft", "google", "facebook", "amazon", "netflix",
    "instagram", "twitter", "linkedin", "whatsapp", "yahoo", "ebay", "dropbox",
    "adobe", "outlook", "office365", "onedrive", "icloud", "gmail", "spotify",
    "steam", "alibaba", "aliexpress", "walmart", "target", "bestbuy", "dhl",
    "fedex", "ups", "usps", "chase", "wellsfargo", "bankofamerica", "citibank",
    "hsbc", "barclays", "santander", "americanexpress", "mastercard", "visa",
    "coinbase", "binance", "blockchain", "metamask", "docusign", "zoom",
    "telegram", "tiktok", "snapchat", "pinterest", "reddit", "github",
    "vietcombank", "vietinbank", "techcombank", "bidv", "mbbank", "agribank",
    "momo", "zalopay", "shopee", "lazada", "tiki", "zalo", "vnpay",
}

NULL_LINKS = {"", "#", "#nothing", "#doesnotexist", "#null", "#void",
              "#whatever", "#content", "#skip", "javascript", "javascript:;",
              "javascript::void(0)", "javascript::void(0);",
              "javascript:void(0)", "javascript:void(0);",
              "javascript:void(0);void(0);"}

USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

try:  # tách miền chính xác (public suffix list có sẵn trong gói, không cần mạng)
    import tldextract
    _TLDX = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)
except Exception:  # pragma: no cover
    _TLDX = None


def load_brands(path: str) -> None:
    """Thay bộ thương hiệu mặc định bằng danh sách trong file (mỗi dòng 1 tên)."""
    global BRANDS
    with open(path, encoding="utf-8") as fh:
        BRANDS = {ln.strip().lower() for ln in fh if ln.strip()}


# --------------------------------------------------------------------------- #
# 3. Tiện ích phân tích URL / URL parsing helpers
# --------------------------------------------------------------------------- #

def normalize_url(url: str) -> str:
    """Thêm scheme `http://` nếu URL thô không có scheme."""
    url = str(url).strip()
    return url if SCHEME_RE.match(url) else "http://" + url


def split_host(host: str) -> tuple[str, str, str]:
    """Trả về (subdomain, domain, suffix) của hostname."""
    if _TLDX is not None:
        ext = _TLDX(host)
        if ext.suffix or ext.domain:
            return ext.subdomain, ext.domain, ext.suffix
    if re.fullmatch(r"[\d.]+|\[?[0-9a-f:]+\]?", host):  # host là địa chỉ IP
        return "", host, ""
    parts = host.split(".")
    if len(parts) == 1:
        return "", host, ""
    n = 2 if (len(parts) >= 3 and len(parts[-1]) == 2 and
              parts[-2] in {"co", "com", "org", "net", "gov", "edu", "ac"}) else 1
    suffix = ".".join(parts[-n:])
    domain = parts[-n - 1]
    return ".".join(parts[: -n - 1]), domain, suffix


def split_words(text: str) -> list[str]:
    return [w for w in SPLIT_RE.split(text.lower()) if w]


def _stats(words: list[str]) -> tuple[int, int, float]:
    if not words:
        return 0, 0, 0.0
    lens = [len(w) for w in words]
    return min(lens), max(lens), float(np.mean(lens))


def char_repeat(words: list[str]) -> int:
    """Số lần một ký tự lặp liên tiếp 2/3/4/5 lần trong các từ."""
    total = 0
    for word in words:
        for k in (2, 3, 4, 5):
            for i in range(len(word) - k + 1):
                if len(set(word[i:i + k])) == 1:
                    total += 1
    return total


def looks_random(label: str) -> int:
    """Xấp xỉ `random_domain`: nhãn miền có vẻ do máy sinh ngẫu nhiên.

    Bản gốc dùng từ điển; ở đây dùng heuristic (không nguyên âm, chuỗi phụ âm
    dài, tỉ lệ chữ số cao, entropy cao). Dùng --validate để đo độ khớp.
    """
    label = label.lower()
    if len(label) < 6:
        return 0
    letters = [c for c in label if c.isalpha()]
    if not letters:
        return 1
    digit_ratio = sum(c.isdigit() for c in label) / len(label)
    vowel_ratio = sum(c in "aeiouy" for c in letters) / len(letters)
    longest_cons = max((len(m) for m in re.findall(r"[^aeiouy\d\W_]+", label)),
                       default=0)
    counts = {c: label.count(c) for c in set(label)}
    entropy = -sum(v / len(label) * math.log2(v / len(label))
                   for v in counts.values())
    return int(digit_ratio > 0.3 or vowel_ratio < 0.2 or longest_cons >= 5
               or entropy > 3.6)


# --------------------------------------------------------------------------- #
# 4. Nhóm 1: thuộc tính từ chuỗi URL / URL-string features
# --------------------------------------------------------------------------- #

def url_features(raw_url: str) -> dict[str, float]:
    url = normalize_url(raw_url)
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
    except ValueError:  # ví dụ IPv6 sai định dạng
        parsed, host = urlparse("http://invalid"), ""
    path = parsed.path or ""
    scheme = parsed.scheme.lower()

    subdomain, domain, suffix = split_host(host)
    # Dataset gốc: từ của host = nhãn miền (KHÔNG gồm đuôi .com) + subdomain;
    # "path" = mọi thứ sau dấu "/" đầu tiên, GỒM cả query và fragment.
    rest = url.split("://", 1)[-1]
    slash = rest.find("/")
    path_q = rest[slash + 1:] if slash >= 0 else ""
    words_host = split_words(domain) + split_words(subdomain)
    words_path = split_words(path_q)
    words_raw = words_host + words_path

    f: dict[str, float] = {}
    f["length_url"] = len(url)
    f["length_hostname"] = len(host)
    f["ip"] = int(bool(IP_RE.search(url)))
    for name, ch in [("nb_dots", "."), ("nb_hyphens", "-"), ("nb_at", "@"),
                     ("nb_qm", "?"), ("nb_and", "&"), ("nb_or", "|"),
                     ("nb_eq", "="), ("nb_underscore", "_"), ("nb_tilde", "~"),
                     ("nb_percent", "%"), ("nb_slash", "/"), ("nb_star", "*"),
                     ("nb_colon", ":"), ("nb_comma", ","),
                     ("nb_semicolumn", ";"), ("nb_dollar", "$")]:
        f[name] = url.count(ch)
    f["nb_space"] = url.count(" ") + url.count("%20")
    f["nb_www"] = sum("www" in w for w in words_raw)
    f["nb_com"] = sum("com" in w for w in words_raw)
    f["nb_dslash"] = rest.count("//")  # không tính "//" của scheme
    f["http_in_path"] = path_q.lower().count("http")
    f["https_token"] = 0 if scheme == "https" else 1  # đúng như dataset: 0 = https
    f["ratio_digits_url"] = sum(c.isdigit() for c in url) / len(url)
    f["ratio_digits_host"] = (sum(c.isdigit() for c in host) / len(host)
                              if host else 0.0)
    f["punycode"] = int("xn--" in host)
    f["port"] = int(parsed.port is not None) if _safe_port(parsed) else 0
    tld = suffix.lower()  # đuôi của CHÍNH tên miền này, tìm như chuỗi con
    f["tld_in_path"] = int(bool(tld) and tld in path_q.lower())
    f["tld_in_subdomain"] = int(bool(tld) and tld in subdomain.lower())
    f["abnormal_subdomain"] = int(bool(
        re.search(r"(http[s]?://(w[w]?|\d))([w]?(\d|-))", url)))
    n_dots = url.count(".")
    f["nb_subdomains"] = 1 if n_dots == 1 else 2 if n_dots == 2 else 3
    f["prefix_suffix"] = int(bool(re.search(r"https?://[^\-]+-[^\-]+/", url)))
    f["random_domain"] = looks_random(domain)
    f["shortening_service"] = int(bool(SHORTENING_RE.search(url)))
    f["path_extension"] = int(path.lower().endswith(".txt"))

    f["length_words_raw"] = len(words_raw)
    f["char_repeat"] = char_repeat(words_raw)
    f["shortest_words_raw"], f["longest_words_raw"], f["avg_words_raw"] = _stats(words_raw)
    f["shortest_word_host"], f["longest_word_host"], f["avg_word_host"] = _stats(words_host)
    f["shortest_word_path"], f["longest_word_path"], f["avg_word_path"] = _stats(words_path)

    f["phish_hints"] = sum(url.lower().count(h) for h in PHISH_HINTS)  # đếm trên TOÀN BỘ URL (gồm cả host)
    f["domain_in_brand"] = int(domain in BRANDS)
    sub_labels = set(split_words(subdomain))
    f["brand_in_subdomain"] = int(any(b in sub_labels and b != domain for b in BRANDS))
    f["brand_in_path"] = int(any(b in words_path and b != domain for b in BRANDS))
    f["suspecious_tld"] = int(suffix.split(".")[-1] in SUSPICIOUS_TLDS)  # tên cột gốc viết sai
    f["statistical_report"] = int(bool(REPORTED_RE.search(url)))
    return f


def _safe_port(parsed) -> bool:
    try:
        parsed.port
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------- #
# 5. Nhóm 2: thuộc tính từ nội dung trang / Page-content features
# --------------------------------------------------------------------------- #

def _registered(host: str) -> str:
    _, domain, suffix = split_host(host)
    return f"{domain}.{suffix}".strip(".")


def _link_kind(link: str, base: str, reg: str) -> str:
    """'null' | 'internal' | 'external'."""
    link = (link or "").strip()
    low = link.lower()
    if low in NULL_LINKS or low.startswith(("mailto:", "tel:", "javascript:")):
        return "null"
    host = (urlparse(urljoin(base, link)).hostname or "").lower()
    if not host:
        return "null"
    return "internal" if (host == reg or host.endswith("." + reg)) else "external"


def _link_kind_host(link: str, base: str) -> str:
    """Như _link_kind nhưng 'nội bộ' = CÙNG HOST (bỏ www.), subdomain khác tính là ngoài."""
    link = (link or "").strip()
    low = link.lower()
    if low in NULL_LINKS or low.startswith(("mailto:", "tel:", "javascript:")):
        return "null"
    host = (urlparse(urljoin(base, link)).hostname or "").lower()
    page = (urlparse(base).hostname or "").lower()
    if not host:
        return "null"
    strip = lambda h: h[4:] if h.startswith("www.") else h
    return "internal" if strip(host) == strip(page) else "external"


def _pct(part: int, total: int) -> float:
    return 100.0 * part / total if total else 0.0


def _ratio(part: int, total: int) -> float:
    return part / total if total else 0.0


def html_features(response, session=None, max_links: int = 0,
                  timeout: float = 8.0) -> dict[str, float]:
    """Tính 26 thuộc tính nhóm HTML từ một `requests.Response`."""
    from bs4 import BeautifulSoup

    final_url = response.url
    host = (urlparse(final_url).hostname or "").lower()
    reg = _registered(host)
    _, domain_label, _ = split_host(host)
    html = response.text or ""
    soup = BeautifulSoup(html, "html.parser")
    f: dict[str, float] = {}

    # --- chuyển hướng / redirects
    hist = list(response.history)
    f["nb_redirection"] = len(hist)
    f["nb_external_redirection"] = sum(
        1 for h in hist
        if _registered((urlparse(h.url).hostname or "").lower()) != reg)

    # --- gom liên kết / collect links
    anchors = [t.get("href") for t in soup.find_all("a", href=True)]
    link_tags = soup.find_all("link", href=True)
    link_hrefs = [t["href"] for t in link_tags]
    script_src = [t["src"] for t in soup.find_all("script", src=True)]
    media_src = [t["src"] for t in soup.find_all(
        ["img", "audio", "video", "source", "embed"], src=True)]

    all_links = anchors + link_hrefs + script_src + media_src
    kinds = [_link_kind(l, final_url, reg) for l in all_links]
    n = len(all_links)
    f["nb_hyperlinks"] = n
    f["ratio_intHyperlinks"] = _ratio(kinds.count("internal"), n)
    f["ratio_extHyperlinks"] = _ratio(kinds.count("external"), n)
    f["ratio_nullHyperlinks"] = _ratio(kinds.count("null"), n)

    css = [t for t in link_tags
           if "stylesheet" in " ".join(t.get("rel", [])).lower()]
    f["nb_extCSS"] = sum(_link_kind(t["href"], final_url, reg) == "external" for t in css)

    # --- ratio_*Redirection / ratio_*Errors: cần gửi request tới từng liên kết
    if max_links > 0 and session is not None:
        sample = list(dict.fromkeys(
            urljoin(final_url, l) for l, k in zip(all_links, kinds) if k != "null"
        ))[:max_links]
        n_int_red = n_ext_red = n_int_err = n_ext_err = 0
        for u in sample:
            kind = _link_kind(u, final_url, reg)
            try:
                r = session.head(u, timeout=timeout, allow_redirects=False)
                red, err = 300 <= r.status_code < 400, r.status_code >= 400
            except Exception:
                red, err = False, True
            n_int_red += red and kind == "internal"
            n_ext_red += red and kind == "external"
            n_int_err += err and kind == "internal"
            n_ext_err += err and kind == "external"
        m = len(sample)
        f["ratio_intRedirection"], f["ratio_extRedirection"] = _ratio(n_int_red, m), _ratio(n_ext_red, m)
        f["ratio_intErrors"], f["ratio_extErrors"] = _ratio(n_int_err, m), _ratio(n_ext_err, m)
    else:
        for c in ("ratio_intRedirection", "ratio_extRedirection",
                  "ratio_intErrors", "ratio_extErrors"):
            f[c] = np.nan

    # --- form
    forms = soup.find_all("form")
    actions = [(fm.get("action") or "").strip() for fm in forms]
    bad_action = lambda a: a.lower() in {"", "#", "about:blank", "javascript:void(0)",
                                         "javascript:void(0);", "javascript:;"}
    f["login_form"] = int(any(
        bad_action(a) or _link_kind(a, final_url, reg) == "external"
        for a in actions))
    f["submit_email"] = int(any(a.lower().startswith("mailto:") for a in actions)
                            or "mail()" in html.lower())
    f["sfh"] = int(any(bad_action(a) or _link_kind(a, final_url, reg) == "external"
                       for a in actions))

    # --- favicon
    icons = [t for t in link_tags
             if "icon" in " ".join(t.get("rel", [])).lower()]
    # Đo bằng tune_content.py trên 101 trang: dataset đánh dấu 1 cả khi KHÔNG có favicon
    # (trung bình 0.52 so với dataset 0.59; cách cũ "không có -> 0" chỉ được 0.04).
    f["external_favicon"] = (int(any(_link_kind(t["href"], final_url, reg) == "external" for t in icons))
                             if icons else 1)

    # --- % liên kết nội bộ trong <link>/<script>
    tag_links = link_hrefs + script_src
    tag_kinds = [_link_kind(l, final_url, reg) for l in tag_links]
    internal_tag = tag_kinds.count("internal")
    f["links_in_tags"] = _pct(internal_tag, internal_tag + tag_kinds.count("external"))

    # --- media (đơn vị %, như dataset gốc)
    mk = [_link_kind(l, final_url, reg) for l in media_src]
    mi, me = mk.count("internal"), mk.count("external")
    f["ratio_intMedia"] = _pct(mi, mi + me)
    f["ratio_extMedia"] = _pct(me, mi + me)

    # --- iframe ẩn
    def hidden_iframe(t) -> bool:
        style = (t.get("style") or "").replace(" ", "").lower()
        return (str(t.get("frameborder", "")) == "0"
                or any(s in style for s in ("display:none", "visibility:hidden", "border:none"))
                or str(t.get("width", "")) == "0" or str(t.get("height", "")) == "0")
    f["iframe"] = int(any(hidden_iframe(t) for t in soup.find_all("iframe")))

    # --- JavaScript đáng ngờ
    low = html.lower().replace(" ", "")
    f["popup_window"] = int(bool(re.search(r"prompt\(", html)))
    f["onmouseover"] = int('onmouseover="window.status=' in low)
    f["right_clic"] = int(bool(re.search(r"event\.button==2", low)))

    # --- % anchor AN TOÀN (không phải null, không trỏ sang host khác)
    # Đo bằng tune_content.py: dataset khớp với 100 - %(null + ngoài), nội bộ = cùng host
    # (trung bình 43.8 so với dataset 41.2; cách cũ "% không an toàn" chỉ 23.5).
    ak = [_link_kind_host(a, final_url) for a in anchors]
    f["safe_anchor"] = (100.0 - _pct(ak.count("null") + ak.count("external"), len(ak))) if ak else 0.0

    # --- tiêu đề & bản quyền (đúng quy ước dataset: 1 = KHÔNG khớp)
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
    f["empty_title"] = int(title == "")
    f["domain_in_title"] = int(domain_label.lower() not in title.lower())
    m = re.search("[\u00a9\u2122\u00ae]", html)
    if m:
        around = html[max(0, m.start() - 50): m.start() + 50].lower()
        f["domain_with_copyright"] = int(domain_label.lower() not in around)
    else:
        f["domain_with_copyright"] = 0
    return f


# --------------------------------------------------------------------------- #
# 6. Nhóm 3: dịch vụ bên ngoài / External-service features
# --------------------------------------------------------------------------- #

def external_features(url: str, timeout: float = 8.0) -> dict[str, float]:
    host = (urlparse(normalize_url(url)).hostname or "").lower()
    reg = _registered(host) or host
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    f = {c: np.nan for c in EXTERNAL_FEATURES}

    # WHOIS: whois_registered_domain, domain_registration_length, domain_age
    try:
        import whois
        w = whois.whois(reg)
        names = w.domain_name if isinstance(w.domain_name, list) else [w.domain_name]
        names = [str(n).lower() for n in names if n]
        f["whois_registered_domain"] = 0 if any(n in host for n in names) else 1

        def first_date(v):
            v = v[0] if isinstance(v, list) else v
            return v.replace(tzinfo=None) if isinstance(v, datetime) else None
        created, expires = first_date(w.creation_date), first_date(w.expiration_date)
        f["domain_age"] = (now - created).days if created else -1
        f["domain_registration_length"] = max((expires - now).days, 0) if expires else 0
    except ImportError:
        pass  # chưa cài python-whois -> NaN
    except Exception:  # miền không có bản ghi WHOIS
        f["whois_registered_domain"], f["domain_age"], f["domain_registration_length"] = 1, -1, 0

    # DNS: 0 = có bản ghi, 1 = không phân giải được (đúng quy ước dataset)
    try:
        socket.gethostbyname(host)
        f["dns_record"] = 0
    except OSError:
        f["dns_record"] = 1

    try:
        import requests
        # web_traffic: Alexa đã ngừng (2022) -> dùng hạng Tranco; 0 = không có trong danh sách
        try:
            r = requests.get(f"https://tranco-list.eu/api/ranks/domain/{reg}", timeout=timeout)
            ranks = r.json().get("ranks", []) if r.ok else []
            f["web_traffic"] = ranks[0]["rank"] if ranks else 0
        except Exception:
            pass

        # page_rank: Open PageRank (cần OPR_API_KEY)
        key = os.getenv("OPR_API_KEY")
        if key:
            try:
                r = requests.get("https://openpagerank.com/api/v1.0/getPageRank",
                                 params={"domains[]": reg},
                                 headers={"API-OPR": key}, timeout=timeout)
                f["page_rank"] = r.json()["response"][0]["page_rank_integer"]
            except Exception:
                pass

        # google_index: 0 = có trong chỉ mục, 1 = không (cần Custom Search API)
        gkey, cx = os.getenv("GOOGLE_API_KEY"), os.getenv("GOOGLE_CSE_ID")
        if gkey and cx:
            try:
                r = requests.get("https://www.googleapis.com/customsearch/v1",
                                 params={"key": gkey, "cx": cx, "q": f"site:{reg}"},
                                 timeout=timeout)
                total = int(r.json()["searchInformation"]["totalResults"])
                f["google_index"] = 0 if total > 0 else 1
            except Exception:
                pass
    except ImportError:
        pass
    return f


# --------------------------------------------------------------------------- #
# 7. API chính / Main API
# --------------------------------------------------------------------------- #

def extract_features(url: str, level: str = "url", timeout: float = 8.0,
                     max_links: int = 0, session=None) -> dict[str, float]:
    """Chuyển một URL thô thành dict thuộc tính theo `feature_columns(level)`.

    level="url"  : chỉ chuỗi URL (offline, nhanh).
    level="html" : + tải trang để tính 26 cột nội dung/chuyển hướng.
    level="full" : + WHOIS/DNS/Tranco/PageRank/Google (7 cột).
    max_links    : số liên kết tối đa được gửi HEAD để tính ratio_*Redirection /
                   ratio_*Errors (0 = bỏ qua, các cột này là NaN).
    """
    cols = feature_columns(level)
    feats = url_features(url)

    if LEVELS[level] >= 2:
        feats.update({c: np.nan for c in HTML_FEATURES})
        try:
            import requests
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            sess = session or requests.Session()
            sess.headers.update({"User-Agent": USER_AGENT})
            resp = sess.get(normalize_url(url), timeout=timeout,
                            allow_redirects=True, verify=False)
            feats.update(html_features(resp, sess, max_links, timeout))
        except ImportError:
            print("Cần `pip install requests beautifulsoup4` cho level html/full.",
                  file=sys.stderr)
        except Exception as exc:  # trang chết, timeout, chặn bot...
            print(f"[warn] không tải được {url}: {type(exc).__name__}", file=sys.stderr)

    if LEVELS[level] >= 3:
        feats.update(external_features(url, timeout))

    return {c: feats.get(c, np.nan) for c in cols}


def build_feature_frame(urls: Iterable[str], level: str = "url", **kw) -> pd.DataFrame:
    """Chuyển nhiều URL thành DataFrame (cột `url` + các thuộc tính)."""
    urls = list(urls)
    rows = []
    for i, u in enumerate(urls, 1):
        rows.append(extract_features(u, level=level, **kw))
        if LEVELS[level] > 1 and i % 20 == 0:
            print(f"  {i}/{len(urls)}", file=sys.stderr)
    df = pd.DataFrame(rows, columns=feature_columns(level))
    df.insert(0, "url", urls)
    return df


def load_dataset(path: str, level: str = "url") -> tuple[pd.DataFrame, pd.Series]:
    """Đọc dataset_phishing.csv -> (X, y). y: legitimate=0, phishing=1.

    Chỉ giữ các cột thuộc `level` để X khớp đúng với `extract_features(level)`
    (huấn luyện bằng level="url" thì demo chỉ cần nhập URL, không cần mạng).
    """
    df = pd.read_csv(path)
    y = (df["status"].astype(str).str.lower() == "phishing").astype(int)
    X = df[feature_columns(level)].copy()
    return X, y


# --------------------------------------------------------------------------- #
# 7b. Điền giá trị dự phòng cho cột không tính được / Fallback for missing columns
# --------------------------------------------------------------------------- #
# Khi tải trang / tra WHOIS / gọi API thất bại, các cột tương ứng là NaN. Model không nhận NaN
# nên điền giá trị dự phòng. Các giá trị dưới đây KHÔNG tự bịa: lấy từ chính dataset gốc,
# ở những dòng mà cùng tình huống đã xảy ra:
#   * 1.381 dòng "trang rỗng" (nb_hyperlinks = 0, 1.063 dòng là phishing): mọi cột nội dung
#     bằng 0, riêng empty_title = 1 và domain_in_title = 1.
#   * 1.781 dòng WHOIS thất bại (domain_age = -1): domain_registration_length = 0,
#     whois_registered_domain = 0, web_traffic và page_rank phổ biến nhất = 0, google_index = 1.

FALLBACK_EMPTY_PAGE = {c: 0 for c in HTML_FEATURES}
FALLBACK_EMPTY_PAGE.update({"empty_title": 1, "domain_in_title": 1})

FALLBACK_EXTERNAL = {
    "whois_registered_domain": 0,
    "domain_registration_length": 0,
    "domain_age": -1,        # placeholder "không tra được" của dataset
    "web_traffic": 0,        # 0 = không có trong danh sách xếp hạng
    "dns_record": 0,
    "google_index": 1,       # 1 = không có trong chỉ mục Google
    "page_rank": 0,
}

FALLBACK = {**FALLBACK_EMPTY_PAGE, **FALLBACK_EXTERNAL}


def fill_missing(feats: dict[str, float]) -> tuple[dict[str, float], list[str]]:
    """Thay mọi NaN bằng giá trị dự phòng. Trả về (dict đã điền, danh sách cột đã phải điền)."""
    filled, imputed = dict(feats), []
    for c, v in feats.items():
        if v is None or (isinstance(v, float) and math.isnan(v)):
            if c not in FALLBACK:
                raise ValueError(f"Cột {c} bị NaN nhưng chưa có giá trị dự phòng")
            filled[c] = FALLBACK[c]
            imputed.append(c)
    return filled, imputed


def extract_features_filled(url: str, level: str = "full", **kw) -> tuple[dict[str, float], list[str]]:
    """extract_features() + fill_missing(). Dùng hàm này trong demo.

    Trả về (87 đặc trưng không còn NaN, danh sách cột đã điền dự phòng).
    Danh sách này dài nghĩa là kết quả dự đoán kém tin cậy hơn.
    """
    return fill_missing(extract_features(url, level=level, **kw))


# --------------------------------------------------------------------------- #
# 8. Kiểm tra độ khớp với dataset gốc / Validation against the original CSV
# --------------------------------------------------------------------------- #

def validate(csv_path: str, sample: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Tính lại các cột nhóm URL từ cột `url` rồi so với giá trị trong CSV."""
    df = pd.read_csv(csv_path)
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=seed)
    mine = pd.DataFrame([url_features(u) for u in df["url"]], index=df.index)
    rows = []
    for c in URL_FEATURES:
        if c not in df.columns:
            rows.append((c, np.nan, np.nan, "thiếu trong CSV"))
            continue
        a, b = mine[c].astype(float), df[c].astype(float)
        match = np.isclose(a, b, rtol=1e-3, atol=1e-6).mean()
        rows.append((c, match, float(np.abs(a - b).mean()), ""))
    rep = pd.DataFrame(rows, columns=["feature", "match_rate", "mean_abs_diff", "note"])
    return rep.sort_values("match_rate").reset_index(drop=True)


def diff_feature(csv_path: str, feature: str, n: int = 15, seed: int = 42) -> pd.DataFrame:
    """In các URL mà giá trị tính lại KHÁC giá trị trong CSV (để dò công thức)."""
    if feature not in URL_FEATURES:
        raise ValueError(f"{feature} không thuộc nhóm URL: {URL_FEATURES}")
    df = pd.read_csv(csv_path)
    mine = pd.Series([url_features(u)[feature] for u in df["url"]], index=df.index)
    bad = ~np.isclose(mine.astype(float), df[feature].astype(float), rtol=1e-3, atol=1e-6)
    out = pd.DataFrame({"url": df["url"], "dataset": df[feature], "mine": mine,
                        "status": df["status"]})[bad]
    print(f"{feature}: lệch {int(bad.sum())}/{len(df)} dòng "
          f"(dataset=1 & mine=0: {int(((out.dataset > out.mine)).sum())}, "
          f"dataset=0 & mine=1: {int(((out.dataset < out.mine)).sum())})")
    return out.sample(min(n, len(out)), random_state=seed) if len(out) else out


# --------------------------------------------------------------------------- #
# 9. Dòng lệnh / CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", help="một URL thô")
    ap.add_argument("--input", help="CSV chứa danh sách URL")
    ap.add_argument("--column", default="url", help="tên cột URL trong --input")
    ap.add_argument("--output", help="ghi kết quả ra CSV")
    ap.add_argument("--level", choices=list(LEVELS), default="url")
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--max-links", type=int, default=0,
                    help="số liên kết HEAD-check cho ratio_*Redirection/Errors")
    ap.add_argument("--fillna", type=float, default=None,
                    help="thay NaN bằng giá trị này (vd. 0)")
    ap.add_argument("--brands", help="file danh sách thương hiệu (mỗi dòng một tên)")
    ap.add_argument("--validate", metavar="DATASET_CSV",
                    help="so sánh các cột nhóm URL với dataset_phishing.csv")
    ap.add_argument("--diff", metavar="FEATURE",
                    help="(cùng --validate) in các URL lệch của một cột, vd. --diff ip")
    ap.add_argument("--sample", type=int, default=3000,
                    help="số dòng lấy mẫu khi --validate (0 = tất cả)")
    args = ap.parse_args(argv)

    if args.brands:
        load_brands(args.brands)

    if args.validate and args.diff:
        with pd.option_context("display.max_colwidth", 90, "display.width", 200):
            print(diff_feature(args.validate, args.diff).to_string(index=False))
        return 0

    if args.validate:
        rep = validate(args.validate, args.sample)
        with pd.option_context("display.max_rows", None, "display.width", 100):
            print(rep.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"\nTrung bình match_rate: {rep['match_rate'].mean():.3f}")
        return 0

    if args.url:
        urls = [args.url]
    elif args.input:
        urls = pd.read_csv(args.input)[args.column].dropna().astype(str).tolist()
    else:
        ap.error("cần một trong --url, --input hoặc --validate")

    out = build_feature_frame(urls, args.level, timeout=args.timeout,
                              max_links=args.max_links)
    if args.fillna is not None:
        out = out.fillna(args.fillna)
    if args.output:
        out.to_csv(args.output, index=False)
        print(f"Đã ghi {len(out)} dòng x {out.shape[1] - 1} thuộc tính -> {args.output}")
    else:
        with pd.option_context("display.max_rows", None):
            print(out.drop(columns="url").T.to_string(header=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())