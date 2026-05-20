import argparse
import csv
import os
import re
import time
from pathlib import Path
from urllib.parse import unquote

import requests
from streamlit import success


API_URL = "https://commons.wikimedia.org/w/api.php"

HEADERS = {
    "User-Agent": "car-body-classifier-dataset/1.0 (educational student project; local dataset collection)"
}


def safe_filename(name: str) -> str:
    name = unquote(name)
    name = name.replace("File:", "")
    name = re.sub(r"[^\w\-. ]+", "_", name)
    name = name.strip().replace(" ", "_")
    return name


def api_get(params):
    params["format"] = "json"
    response = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()


def get_category_members(category, member_type):
    """
    member_type: 'file' or 'subcat'
    """
    members = []
    cont = {}

    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmtype": member_type,
            "cmlimit": "500",
            **cont,
        }

        data = api_get(params)
        members.extend(data.get("query", {}).get("categorymembers", []))

        if "continue" not in data:
            break

        cont = data["continue"]

    return members


def get_file_info(file_title):
    params = {
        "action": "query",
        "titles": file_title,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
    }

    data = api_get(params)
    pages = data.get("query", {}).get("pages", {})

    for page in pages.values():
        info = page.get("imageinfo", [])
        if not info:
            return None
        return info[0]

    return None


def download_file(url, output_path, max_retries=8):
    wait_time = 10

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, stream=True, timeout=60)

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")

                if retry_after and retry_after.isdigit():
                    sleep_time = int(retry_after)
                else:
                    sleep_time = wait_time

                print(f"429 Too Many Requests. {sleep_time} saniye bekleniyor...")
                time.sleep(sleep_time)
                wait_time *= 2
                continue

            response.raise_for_status()

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            return True

        except Exception as e:
            print(f"Download retry {attempt + 1}/{max_retries}: {e}")
            time.sleep(wait_time)
            wait_time *= 2

    return False


def collect_files(category, recursive=False, max_depth=1, current_depth=0, seen_categories=None):
    if seen_categories is None:
        seen_categories = set()

    if category in seen_categories:
        return []

    seen_categories.add(category)

    files = get_category_members(category, "file")

    if recursive and current_depth < max_depth:
        subcats = get_category_members(category, "subcat")
        for subcat in subcats:
            subcat_title = subcat["title"].replace("Category:", "")
            files.extend(
                collect_files(
                    subcat_title,
                    recursive=True,
                    max_depth=max_depth,
                    current_depth=current_depth + 1,
                    seen_categories=seen_categories,
                )
            )

    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, help="Commons category name without Category:")
    parser.add_argument("--out", required=True, help="Output folder")
    parser.add_argument("--limit", type=int, default=500, help="Maximum number of files to download")
    parser.add_argument("--prefix", default="IMG", help="Filename prefix")
    parser.add_argument("--recursive", action="store_true", help="Also scan subcategories")
    parser.add_argument("--max-depth", type=int, default=1, help="Subcategory depth")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = collect_files(
        args.category,
        recursive=args.recursive,
        max_depth=args.max_depth,
    )

    print(f"Found {len(files)} files in category search.")

    metadata_path = out_dir / "source_metadata.csv"

    downloaded = 0

    with open(metadata_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["local_filename", "commons_title", "source_url", "license", "artist"])

        for item in files:
            if downloaded >= args.limit:
                break

            title = item["title"]

            try:
                info = get_file_info(title)
                if not info or "url" not in info:
                    continue

                url = info["url"]
                ext = os.path.splitext(url.split("?")[0])[1].lower()

                if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
                    continue

                local_name = f"{args.prefix}_{downloaded + 1:04d}{ext}"
                output_path = out_dir / local_name

                success = download_file(url, output_path)

                if not success:
                    print(f"Failed after retries: {title}")
                    continue

                meta = info.get("extmetadata", {})
                license_name = meta.get("LicenseShortName", {}).get("value", "")
                artist = meta.get("Artist", {}).get("value", "")

                writer.writerow([local_name, title, url, license_name, artist])

                downloaded += 1
                print(f"[{downloaded}] Downloaded: {local_name}")

                time.sleep(3)

            except Exception as e:
                print(f"Skipped {title}: {e}")

    print(f"Done. Downloaded {downloaded} files.")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()