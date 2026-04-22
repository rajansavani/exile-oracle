from __future__ import annotations
from cProfile import label
import json
import re
import time
from pathlib import Path
import requests
from tqdm import tqdm
from src.config import settings


# CONSTANTS

# base URL of the PoE Wiki's MediaWiki API
WIKI_API = "https://www.poewiki.net/w/api.php"

# identify ourselves with a user agent when making requests to the wiki
USER_AGENT = (
    f"{settings.app_name}/{settings.app_version} "
    "(educational RAG project; contact: https://github.com/rajansavani)"
)

# delay between requests to avoid rate-limiting and to be polite to the wiki
REQUEST_DELAY_SECONDS = 0.5

# where to save the scraped plain-text files
RAW_DIR = Path("data/raw")

# which categories to pull (keys are just labels, values are exact category names on the wiki)
TARGET_CATEGORIES: dict[str, str] = {
    # skills 
    "skill_gems": "Skill gems",                                  # 265
    "support_gems": "Support gems",                              # 149

    # passive tree
    "keystones": "Keystone passive skills",                      # 114
    "notable_passives": "Notable passive skills",                # 1056

    # ascendancies
    "ascendancy_classes": "Ascendancy classes",                  # 25
    "ascendancy_notables": "Ascendancy notable passive skills",  # 302
    "ascendancy_basics": "Ascendancy basic passive skills",      # 161

    # mechanics
    "damage": "Damage",                                          # 8
    "damage_types": "Damage types",                              # 5
    "damage_sources": "Damage sources",                          # 4
    "item_mechanics": "Item mechanics",                          # 21
    "item_modifiers": "Item modifiers",                          # 41

    # uniques
    "unique_amulets": "Unique amulets",                          # 105
    "unique_belts": "Unique belts",                              # 78
    "unique_body_armours": "Unique body armours",                # 132
    "unique_boots": "Unique boots",                              # 87
    "unique_bows": "Unique bows",                                # 39
    "unique_claws": "Unique claws",                              # 24
    "unique_daggers": "Unique daggers",                          # 13
    "unique_gloves": "Unique gloves",                            # 99
    "unique_helmets": "Unique helmets",                          # 131
    "unique_axes": "Unique axes",                                # 3
    "unique_abyss_jewels": "Unique abyss jewels",                # 5
    "unique_contracts": "Unique contracts",                      # 6
    "unique_fishing_rods": "Unique fishing rods",                # 4
    "unique_flasks": "Unique flasks",                            # 3
    "unique_hybrid_flasks": "Unique hybrid flasks",              # 3
    "unique_idols": "Unique idols",                              # 40
    "unique_item_pieces": "Unique item pieces",                  # 26

    # bosses
    "bosses": "Boss monsters",                                   # 355

    # patch notes
    "patch_notes": "Versions",                                   # 778
}

# hard ceiling on pages per category, capping for now to avoid runaway scraping in v1
MAX_PAGES_PER_CATEGORY = 1200


# small MediaWiki API client with session reuse and rate limiting
class WikiClient:
    def __init__(self, api_url: str = WIKI_API, delay: float = REQUEST_DELAY_SECONDS):
        self.api_url = api_url
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
    
    # get the MediaWiki API with JSON response and delay, retries for transient errors
    def get(self, params: dict, max_retries: int = 5) -> dict:
        params = {**params, "format": "json", "formatversion": "2"}

        for attempt in range(max_retries):
            try:
                response = self.session.get(self.api_url, params=params, timeout=30)
                # 5xx and 429 are retryable; everything else (200, 404, etc.) returns immediately
                if response.status_code >= 500 or response.status_code == 429:
                    # raise inside the try so the except block handles the backoff
                    response.raise_for_status()
                response.raise_for_status()
                time.sleep(self.delay)
                return response.json()
            except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
                is_last_attempt = attempt == max_retries - 1
                if is_last_attempt:
                    # out of retries, raise the error to be handled by the caller
                    raise
                backoff = 2 ** attempt  # 1, 2, 4, 8, 16 seconds
                # use tqdm.write so we don't corrupt progress bars if one is active
                try:
                    tqdm.write(f"    ~ transient error ({type(e).__name__}): retry {attempt + 1}/{max_retries} in {backoff}s")
                except Exception:
                    print(f"    ~ transient error ({type(e).__name__}): retry {attempt + 1}/{max_retries} in {backoff}s")
                time.sleep(backoff)

        raise RuntimeError("exhausted retries without return or raise")
    

# category enumeration

# return all subcategory names of `category` 
def list_subcategories(client: WikiClient, category: str) -> list[str]:
    subcats: list[str] = []
    cmcontinue: str | None = None

    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmnamespace": "14",  # namespace 14 = category pages
            "cmlimit": "500",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = client.get(params)
        for m in data.get("query", {}).get("categorymembers", []):
            # strip category prefix from title
            title = m["title"]
            if title.startswith("Category:"):
                subcats.append(title[len("Category:"):])
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
    return subcats


# return all page titles in the given category, up to the specified limit
def list_category_pages(client: WikiClient, category: str, limit: int) -> list[str]:
    titles: list[str] = []
    cmcontinue: str | None = None

    while len(titles) < limit:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmnamespace": "0", # namespace 0 = main article namespace (skips category, talk, file, etc.)
            "cmlimit": "500", # max allowed by MediaWiki API
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        
        data = client.get(params)
        members = data.get("query", {}).get("categorymembers", [])
        titles.extend(m["title"] for m in members)

        # MediaWiki tells us if there's more via a "continue" token
        # if absent we've gone through everything
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break

    return titles[:limit]

# return all page titles in the given category and its subcategories (deduplicated)
def list_category_pages_recursive(
    client: WikiClient,
    category: str,
    limit: int,
    max_depth: int = 2,
) -> list[str]:
    seen: set[str] = set()
    # visit stack: (category_name, depth_remaining)
    to_visit: list[tuple[str, int]] = [(category, max_depth)]

    while to_visit and len(seen) < limit:
        cat, depth = to_visit.pop(0)
        # pull direct article pages from this category
        for title in list_category_pages(client, cat, limit - len(seen)):
            seen.add(title)
            if len(seen) >= limit:
                break
        # recurse into subcategories if we haven't hit our depth cap
        if depth > 0:
            try:
                to_visit.extend((sub, depth - 1) for sub in list_subcategories(client, cat))
            except Exception as e:
                print(f"    ! subcat listing failed for {cat!r}: {e}")

    return list(seen)[:limit]

# page content fetching
# return the plain-text content of a single wiki page
def fetch_page_text(client: WikiClient, title: str) -> str | None:
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",         # plain text instead of HTML
        "exsectionformat": "plain", # no Section headers, just prose
        "titles": title,
        "redirects": "1",
    }
    data = client.get(params)
    pages = data.get("query", {}).get("pages", [])

    if not pages:
        return None

    # with formatversion=2, pages is a list of dicts with a title + extract
    page = pages[0]
    if page.get("missing"):
        return None
    extract = page.get("extract", "").strip()
    return extract or None

# filesystem helpers

# convert a wiki page title into a safe filename
def safe_filename(title: str) -> str:
    # replace any sequence of illegal characters or whitespace with a single underscore
    cleaned = re.sub(r'[\\/:*?"<>|\s]+', "_", title)

    # trim leading/trailing underscores and limit length
    return cleaned.strip("_")[:150]

# write a page's text to disk and return the path written
def write_page(category_label: str, title: str, text: str) -> Path:
    out_dir = RAW_DIR / category_label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{safe_filename(title)}.txt"
    # append the title as the first line so downstream processing can use it as metadata if needed
    out_path.write_text(f"# {title}\n\n{text}", encoding="utf-8")
    return out_path

# main ingestion function: fetch pages from each target category and save to data/raw/
def ingest(categories: dict[str, str] = TARGET_CATEGORIES, max_per_category: int = MAX_PAGES_PER_CATEGORY) -> dict[str, int]:
    client = WikiClient()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, int] = {}

    for label, category in categories.items():
        print(f"\n=== Category: {category} (label: {label}) ===")
        try:
            titles = list_category_pages_recursive(client, category, max_per_category)
        except requests.HTTPError as e:
            print(f"Failed to list pages for category '{category}': {e}")
            summary[label] = 0
            continue

        print(f"  Found {len(titles)} pages. Fetching...")
        written = 0
        for title in tqdm(titles, desc=f"  {label}", unit="page"):
            try:
                text = fetch_page_text(client, title)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                # log any other failure and move on
                tqdm.write(f"    ! {title}: {type(e).__name__}: {e}")
                continue
            if not text:
                continue
            write_page(label, title, text)
            written += 1
        summary[label] = written

    # manifest so we know what's on the disk without having to read the filesystem
    manifest = {
        "source": "poewiki.net",
        "api": WIKI_API,
        "categories": summary,
        "total_pages": sum(summary.values()),
    }
    (RAW_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return summary

if __name__ == "__main__":
    print(f"Ingesting to {RAW_DIR.resolve()}")
    summary = ingest()
    print("\n=== Done ===")
    for label, count in summary.items():
        print(f"  {label:20s} {count:>5d} pages")
    print(f"  {'TOTAL':20s} {sum(summary.values()):>5d} pages")