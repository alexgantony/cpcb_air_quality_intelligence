"""Fetch the latest CPCB real-time air quality snapshot from data.gov.in
and save it untouched as data/raw/api/YYYY-MM-DD_HHMM.csv.gz (India time).

Run from anywhere:  uv run src/collect.py
"""

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# --- Settings -----------------------------------------------------------------
load_dotenv()  # reads .env locally; does nothing on GitHub Actions
API_KEY = os.environ["DATA_GOV_API_KEY"]
URL = "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"

PAGE_SIZE = 999  # API rejects limit >= 1000
MIN_PAGE_SIZE = 100  # smallest page size to fall back to
MAX_ATTEMPTS = 3  # tries per page
MAX_PASSES = 3  # full passes allowed to fill in records that paging missed
RETRY_WAIT = 20  # seconds between tries
EMPTY_WAIT = 180  # seconds to wait when the dataset is empty (being refreshed)
EMPTY_ATTEMPTS = 2  # one wait of 3 min: the next scheduled run picks it up otherwise
TIMEOUT = (10, 120)  # (connect, read) seconds

# Optional extra query parameters asking the API for a fixed sort order,
# e.g. {"sort[station]": "asc"}. Left empty until confirmed in a browser:
# an unsupported parameter could make every request fail.
SORT_PARAMS: dict = {}

# One reading = one pollutant at one station. Used to find duplicates.
KEY_COLS = ["state", "city", "station", "pollutant_id"]

# Some government servers reject the default "python-requests" User-Agent,
# so identify as a normal browser.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

# Fixed UTC+5:30 offset. ZoneInfo("Asia/Kolkata") would need the extra
# tzdata package on Windows, and India has no daylight saving anyway.
IST = timezone(timedelta(hours=5, minutes=30))

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "raw" / "api"

API_TIME_FORMAT = "%d-%m-%Y %H:%M:%S"  # how last_update arrives, e.g. 24-09-2026 17:00:00
FILE_TIME_FORMAT = "%Y-%m-%d_%H%M"  # how snapshot files are named, e.g. 2026-09-24_1700


# --- Functions ----------------------------------------------------------------
class EmptyDatasetError(RuntimeError):
    """The API answered 200 OK but the whole dataset is empty.

    data.gov.in does this for several minutes while it reloads the hourly data.
    Retrying quickly or shrinking the page size won't help: we have to wait.
    """


def fetch_page(offset: int, limit: int) -> dict:
    """Request one page of records, retrying on bad status codes and network errors."""
    params = {"api-key": API_KEY, "format": "json", "limit": limit, "offset": offset}
    params.update(SORT_PARAMS)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            r = requests.get(URL, params=params, headers=HEADERS, timeout=TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                if int(data.get("total") or 0) == 0:
                    raise EmptyDatasetError("API returned an empty dataset")
                return data
            print(f"  offset {offset}, limit {limit}, attempt {attempt}: HTTP {r.status_code}")
        except (requests.exceptions.RequestException, ValueError) as e:
            # ValueError: response was not valid JSON (e.g. an HTML error page)
            print(f"  offset {offset}, limit {limit}, attempt {attempt}: {type(e).__name__}")

        if attempt < MAX_ATTEMPTS:
            time.sleep(RETRY_WAIT)

    raise RuntimeError(f"API failed {MAX_ATTEMPTS} times at offset {offset}")


def fetch_pass(page_size: int) -> tuple[list, int, int]:
    """Page through the API once. Returns (records, total, page size used).

    If a page keeps failing, the page size is halved (999 -> 499 -> 249 -> 124)
    because large pages are the usual cause of 502 errors from this API.
    """
    records = []
    total = None

    while total is None or len(records) < total:
        offset = len(records)  # next record we still need
        try:
            data = fetch_page(offset, page_size)
        except EmptyDatasetError:
            raise  # not a page-size problem: let fetch_all wait it out
        except RuntimeError:
            if page_size // 2 < MIN_PAGE_SIZE:
                raise  # already at the smallest size: give up
            page_size //= 2
            print(f"  reducing page size to {page_size}")
            continue

        total = int(data["total"])
        page = data.get("records", [])

        if not page:  # safety stop: never loop forever on an empty page
            break

        records.extend(page)
        print(f"  got {len(records)}/{total} records")

    return records, total, page_size


def fetch_pass_when_ready(page_size: int) -> tuple[list, int, int]:
    """Run fetch_pass, waiting and retrying while the dataset is being refreshed."""
    for attempt in range(1, EMPTY_ATTEMPTS + 1):
        try:
            return fetch_pass(page_size)
        except EmptyDatasetError:
            if attempt == EMPTY_ATTEMPTS:
                raise
            print(f"  dataset empty (API refreshing), waiting {EMPTY_WAIT // 60} min "
                  f"[{attempt}/{EMPTY_ATTEMPTS - 1}]")
            time.sleep(EMPTY_WAIT)
    raise EmptyDatasetError("unreachable")


def fetch_all() -> pd.DataFrame:
    """Collect every record, repairing gaps caused by unstable paging.

    The API does not return records in a fixed order, so one page can repeat
    a record from another page while a different record is skipped. Each pass
    is merged into a dict keyed by station + pollutant, which removes the
    repeats; extra passes run until every record has been seen.
    """
    readings = {}  # (state, city, station, pollutant_id) -> record
    page_size = PAGE_SIZE
    total = 0

    for pass_no in range(1, MAX_PASSES + 1):
        print(f"Pass {pass_no}")
        records, total, page_size = fetch_pass_when_ready(page_size)

        for rec in records:
            readings[tuple(rec[c] for c in KEY_COLS)] = rec

        missing = total - len(readings)
        print(f"  unique readings: {len(readings)}/{total}")
        if missing <= 0:
            break
        if pass_no < MAX_PASSES:
            print(f"  {missing} readings missing, running another pass")

    df = pd.DataFrame(list(readings.values()))

    if df.empty:
        raise RuntimeError("API returned 0 records")
    if len(df) < total:
        # Save what we have rather than losing the whole hour.
        print(f"  warning: {total - len(df)} readings still missing after {MAX_PASSES} passes")
    if df["last_update"].nunique() > 1:
        # Happens if CPCB updates mid-run. Data is still valid, so save it
        # and let the cleaning step deal with it.
        print(f"  warning: {df['last_update'].nunique()} different last_update values")

    return df


def snapshot_path(last_update: str) -> Path:
    """Path a snapshot taken at this reading time is saved to.

    Files are named after the reading's own hour (last_update), not the time we
    fetched it, so the same hour always maps to the same file and collecting it
    twice is impossible.
    """
    reading_time = datetime.strptime(last_update, API_TIME_FORMAT)
    return OUT_DIR / f"{reading_time:{FILE_TIME_FORMAT}}.csv.gz"


def peek_reading_time() -> str | None:
    """Ask for a single record to learn which hour the API is currently serving.

    Lets a run skip the full download when that hour is already saved. Returns
    None if the cheap check fails for any reason, in which case the caller just
    does the full fetch.
    """
    try:
        return fetch_page(0, 1)["records"][0]["last_update"]
    except (EmptyDatasetError, RuntimeError, LookupError):
        return None


def main() -> None:
    fetched_at = datetime.now(IST)
    print(f"Fetching CPCB data at {fetched_at:%Y-%m-%d %H:%M} IST")

    # Cheap check first: one record tells us the hour on offer.
    last_update = peek_reading_time()
    if last_update is not None and snapshot_path(last_update).exists():
        print(f"  {last_update} already collected, skipping")
        return

    df = fetch_all()

    out_path = snapshot_path(df["last_update"].mode()[0])
    if out_path.exists():  # peek failed earlier, or the hour changed mid-run
        print(f"  {out_path.name} already collected, skipping")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)  # .gz extension -> compressed automatically

    print(f"Saved {len(df)} rows to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
