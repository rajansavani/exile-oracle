# diagnostic tool: list real categories on poewiki.net matching prefixes

from __future__ import annotations
import argparse
from src.ingest import WikiClient

# return (category_name, page_count) for categories matching prefix
def list_categories_with_prefix(client: WikiClient, prefix: str, limit: int = 50) -> list[tuple[str, int]]:
    params = {
        "action": "query",
        "list": "allcategories",
        "acprefix": prefix,
        "aclimit": str(limit),
        "acprop": "size",   # include size (page count) in response
    }
    data = client.get(params)
    results = []
    for cat in data.get("query", {}).get("allcategories", []):
        name = cat.get("category") or cat.get("*") or ""
        size = cat.get("size", cat.get("pages", 0))
        results.append((name, size))
    return results

# return number of pages in a specific category (main namespace only)
def count_category_members(client: WikiClient, category: str) -> int:
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmnamespace": "0",
        "cmlimit": "500",
    }
    data = client.get(params)
    return len(data.get("query", {}).get("categorymembers", []))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", nargs="?", default=None, help="Category name prefix to search")
    parser.add_argument("--members", help="Count members of this exact category")
    args = parser.parse_args()

    client = WikiClient()

    if args.members:
        count = count_category_members(client, args.members)
        print(f"Category:{args.members}  ->  {count} pages (main namespace)")
        return

    # if no prefix given, use some common ones as default
    prefixes = [args.prefix] if args.prefix else [
        "Skill", "Support", "Keystone", "Ascendancy",
        "Unique", "Item", "Boss", "Status", "Damage", "Version",
    ]

    for prefix in prefixes:
        print(f"\n--- Categories starting with {prefix!r} ---")
        results = list_categories_with_prefix(client, prefix)
        if not results:
            print("  (none found)")
            continue
        for name, size in results:
            print(f"  {size:>5}  Category:{name}")


if __name__ == "__main__":
    main()