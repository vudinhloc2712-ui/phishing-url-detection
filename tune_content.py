"""
Dò định nghĩa đúng cho các cột nội dung trang bằng cách thử nhiều cách tính và so với dataset gốc.

Chạy từ thư mục gốc project (lần đầu cần mạng, các lần sau đọc từ cache nên chạy offline, nhanh):
    python tune_content.py                 # 60 URL mỗi lớp
    python tune_content.py --n 100         # nhiều hơn -> kết luận chắc hơn

Cách hoạt động:
  1. Lấy ngẫu nhiên URL trong dataset, tải HTML lưu vào thư mục cache_pages/ (chỉ tải 1 lần).
  2. Với mỗi cột còn nghi ngờ, tính theo nhiều cách (variant) trên cùng HTML đã lưu.
  3. So từng variant với giá trị dataset, in agree% (cao hơn = giống dataset hơn).
Lưu ý: trang đã đổi kể từ 2020 nên agree% không bao giờ 100%; hãy nhìn variant nào TÁCH HẲN khỏi các variant khác.
"""
import argparse
import json
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.preprocess.DemoPreprocessor.url_features import (
    NULL_LINKS, USER_AGENT, _registered, normalize_url, split_host)

# ---------------------------------------------------------------- tải & cache
def fetch(url, timeout):
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    r = s.get(normalize_url(url), timeout=timeout, allow_redirects=True, verify=False)
    return {"url": url, "final_url": r.url, "html": r.text or "", "n_history": len(r.history)}


def load_pages(df, cache_dir, timeout, workers=8):
    cache_dir.mkdir(exist_ok=True)

    def one(item):
        idx, url = item
        f = cache_dir / f"{idx}.json"
        if f.exists():
            return idx, json.loads(f.read_text(encoding="utf-8"))
        try:
            page = fetch(url, timeout)
        except Exception:
            page = None
        f.write_text(json.dumps(page), encoding="utf-8")
        return idx, page

    with ThreadPoolExecutor(workers) as ex:
        res = dict(ex.map(one, df["url"].items()))
    return {i: p for i, p in res.items() if p and p["html"].strip()}

# ---------------------------------------------------------------- phân loại link
SCOPES = ["registered", "host", "domain_in_link"]


def _host(u):
    return (urlparse(u).hostname or "").lower()


def kind(link, base, scope):
    """'null' | 'internal' | 'external' theo 3 cách hiểu 'nội bộ'."""
    link = (link or "").strip()
    low = link.lower()
    if low in NULL_LINKS or low.startswith(("mailto:", "tel:", "javascript:")):
        return "null"
    absu = urljoin(base, link)
    h, ph = _host(absu), _host(base)
    if not h:
        return "null"
    if scope == "registered":
        reg = _registered(ph)
        return "internal" if (h == reg or h.endswith("." + reg)) else "external"
    if scope == "host":
        strip = lambda x: x[4:] if x.startswith("www.") else x
        return "internal" if strip(h) == strip(ph) else "external"
    label = split_host(ph)[1]
    return "internal" if (label and label in absu.lower()) else "external"


def _pct(a, b):
    return 100.0 * a / b if b else 0.0


def _ratio(a, b):
    return a / b if b else 0.0

# ---------------------------------------------------------------- các variant
def collect(soup):
    a = [t.get("href") for t in soup.find_all("a", href=True)]
    link_tags = soup.find_all("link", href=True)
    link = [t["href"] for t in link_tags]
    script = [t["src"] for t in soup.find_all("script", src=True)]
    img = [t["src"] for t in soup.find_all("img", src=True)]
    media = [t["src"] for t in soup.find_all(["img", "audio", "video", "source", "embed"], src=True)]
    iframe = [t["src"] for t in soup.find_all("iframe", src=True)]
    anyattr = [t.get("href") or t.get("src") or t.get("action") for t in soup.find_all(True)
               if t.get("href") or t.get("src") or t.get("action")]
    return dict(a=a, link_tags=link_tags, link=link, script=script, img=img, media=media,
                iframe=iframe, anyattr=anyattr)


TAGSETS = {
    "a": lambda c: c["a"],
    "a+link+script": lambda c: c["a"] + c["link"] + c["script"],
    "a+link+script+media [HIỆN TẠI]": lambda c: c["a"] + c["link"] + c["script"] + c["media"],
    "a+link+script+media+iframe": lambda c: c["a"] + c["link"] + c["script"] + c["media"] + c["iframe"],
    "mọi thẻ có href/src/action": lambda c: c["anyattr"],
}


def build_variants(page):
    from bs4 import BeautifulSoup
    html, base = page["html"], page["final_url"]
    soup = BeautifulSoup(html, "html.parser")
    c = collect(soup)
    V = {}   # cột -> {tên variant: giá trị}

    # nb_hyperlinks
    V["nb_hyperlinks"] = {name: len(fn(c)) for name, fn in TAGSETS.items()}

    # ratio_int/ext Hyperlinks: tagset x scope
    for col, want in (("ratio_extHyperlinks", "external"), ("ratio_intHyperlinks", "internal")):
        V[col] = {}
        for tname, fn in TAGSETS.items():
            links = fn(c)
            for sc in SCOPES:
                ks = [kind(l, base, sc) for l in links]
                cur = " [HIỆN TẠI]" if (tname.endswith("[HIỆN TẠI]") and sc == "registered") else ""
                V[col][f"{tname.replace(' [HIỆN TẠI]', '')} | {sc}{cur}"] = _ratio(ks.count(want), len(links))

    # ratio_ext/int Media (đơn vị %)
    for col, want in (("ratio_extMedia", "external"), ("ratio_intMedia", "internal")):
        V[col] = {}
        for mname, links in (("img+audio+video+source+embed", c["media"]), ("chỉ img", c["img"])):
            for sc in SCOPES:
                ks = [kind(l, base, sc) for l in links]
                mi, me = ks.count("internal"), ks.count("external")
                cur = " [HIỆN TẠI]" if (mname.startswith("img+") and sc == "registered") else ""
                V[col][f"{mname} | {sc} | mẫu=int+ext{cur}"] = _pct(ks.count(want), mi + me)
                V[col][f"{mname} | {sc} | mẫu=tất cả"] = _pct(ks.count(want), len(ks))

    # links_in_tags (% nội bộ trong <link>/<script>)
    V["links_in_tags"] = {}
    for tname, links in (("link+script", c["link"] + c["script"]), ("chỉ link", c["link"])):
        for sc in SCOPES:
            ks = [kind(l, base, sc) for l in links]
            i, e = ks.count("internal"), ks.count("external")
            cur = " [HIỆN TẠI]" if (tname == "link+script" and sc == "registered") else ""
            V["links_in_tags"][f"{tname} | {sc}{cur}"] = _pct(i, i + e)

    # nb_extCSS
    V["nb_extCSS"] = {}
    for sc in SCOPES:
        css = [t for t in c["link_tags"] if "stylesheet" in " ".join(t.get("rel", [])).lower()]
        V["nb_extCSS"][f"rel=stylesheet | {sc}" + (" [HIỆN TẠI]" if sc == "registered" else "")] = \
            sum(kind(t["href"], base, sc) == "external" for t in css)
        css2 = [t for t in c["link_tags"] if ".css" in t["href"].lower()]
        V["nb_extCSS"][f"href chứa .css | {sc}"] = sum(kind(t["href"], base, sc) == "external" for t in css2)

    # external_favicon
    V["external_favicon"] = {}
    icons = [t for t in c["link_tags"] if "icon" in " ".join(t.get("rel", [])).lower()]
    for sc in SCOPES:
        ext = int(any(kind(t["href"], base, sc) == "external" for t in icons))
        cur = " [HIỆN TẠI]" if sc == "registered" else ""
        V["external_favicon"][f"không có favicon->0 | {sc}{cur}"] = ext
        V["external_favicon"][f"không có favicon->1 | {sc}"] = ext if icons else 1

    # safe_anchor (%)
    V["safe_anchor"] = {}
    n = len(c["a"])
    for sc in SCOPES:
        ks = [kind(l, base, sc) for l in c["a"]]
        unsafe = _pct(ks.count("null") + ks.count("external"), n)
        cur = " [HIỆN TẠI]" if sc == "registered" else ""
        V["safe_anchor"][f"% (null+external) | {sc}{cur}"] = unsafe
        V["safe_anchor"][f"100 - % (null+external) | {sc}"] = 100 - unsafe if n else 0.0
    nulls = _pct(sum(kind(l, base, "host") == "null" for l in c["a"]), n)
    V["safe_anchor"]["% chỉ null (#, javascript, mailto)"] = nulls
    V["safe_anchor"]["100 - % chỉ null"] = 100 - nulls if n else 0.0

    # domain_with_copyright
    label = split_host(_host(base))[1].lower()
    low = html.lower()
    m = re.search(r"©|&copy;|™|®", low)
    m2 = re.search(r"©|&copy;|copyright", low)
    around = low[max(0, m.start() - 50): m.start() + 50] if m else ""
    around2 = low[max(0, m2.start() - 50): m2.start() + 50] if m2 else ""
    V["domain_with_copyright"] = {
        "có © và KHÔNG chứa domain ->1 [HIỆN TẠI]": int(bool(m) and label not in around),
        "có © và chứa domain ->1": int(bool(m) and label in around),
        "có ©/copyright và chứa domain ->1": int(bool(m2) and label in around2),
        "KHÔNG (có ©/copyright và chứa domain) ->1": int(not (bool(m2) and label in around2)),
    }
    return V

# ---------------------------------------------------------------- so sánh
TOL = {"nb_hyperlinks": 2, "nb_extCSS": 0, "ratio_extHyperlinks": 0.03, "ratio_intHyperlinks": 0.03,
       "ratio_extMedia": 3, "ratio_intMedia": 3, "links_in_tags": 3, "safe_anchor": 3,
       "external_favicon": 0, "domain_with_copyright": 0}


def agree(a, b, col):
    return np.abs(a - b) <= np.maximum(0.1 * np.maximum(np.abs(a), np.abs(b)), TOL[col])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="số URL mỗi lớp")
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--csv", default="data/raw/dataset_phishing.csv")
    ap.add_argument("--cache", default="cache_pages")
    ap.add_argument("--top", type=int, default=5, help="số variant tốt nhất in ra cho mỗi cột")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    df = df[df["nb_hyperlinks"] > 0]            # chỉ trang có nội dung lúc thu thập (2020)
    sample = pd.concat([df[df.status == "legitimate"].sample(min(a.n, (df.status == "legitimate").sum()), random_state=11),
                        df[df.status == "phishing"].sample(min(a.n, (df.status == "phishing").sum()), random_state=11)])
    print(f"Tải/đọc cache {len(sample)} trang ...")
    pages = load_pages(sample, Path(a.cache), a.timeout)
    print(f"Còn sống và có nội dung: {len(pages)}/{len(sample)} trang\n")
    if len(pages) < 10:
        print("Quá ít trang để kết luận - kiểm tra mạng hoặc tăng --n")
        return

    allv = {i: build_variants(p) for i, p in pages.items()}
    idx = list(pages)
    for col in TOL:
        ds = sample.loc[idx, col].astype(float).values
        rows = []
        for vname in allv[idx[0]][col]:
            mine = np.array([allv[i][col][vname] for i in idx], dtype=float)
            rows.append((vname, agree(ds, mine, col).mean() * 100, mine.mean()))
        rows.sort(key=lambda r: -r[1])
        print(f"== {col}   (dataset: mean={ds.mean():.2f}, n={len(idx)})")
        shown = rows[: a.top] + [r for r in rows[a.top:] if "[HIỆN TẠI]" in r[0]]
        for name, ag, mean in shown:
            print(f"   agree {ag:5.1f}%  mean={mean:8.2f}   {name}")
        print()


if __name__ == "__main__":
    main()