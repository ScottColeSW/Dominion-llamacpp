"""Pre-production image fetch pipeline (Phase 2, Revision 23).

Standalone script, not imported by the running engine. It queries the
Wikimedia Commons API for a real, freely licensed photo per question and
writes the results to image_library.json next to this file, keyed by
"DomainName|||image_prompt" so content.py can patch them onto the matching
Question at import time (see _apply_image_library there).

Run it per-domain during development so results can be spot-checked before
committing to the full library:

    python3 fetch_images.py --domain Cats
    python3 fetch_images.py --domain Cats --limit 15
    python3 fetch_images.py --all --limit 15

Nothing here touches game logic or the content.py question text; a domain
with no entry in image_library.json just keeps showing its plain-text
image_prompt exactly as Phase 1 always has, so this is safe to run
incrementally, one domain (or one small batch) at a time. Results are saved
incrementally and a question already present in image_library.json is
skipped on a re-run, so an interrupted or --limit-capped pass can simply be
re-invoked to pick up where it left off rather than re-querying everything.
"""
from __future__ import annotations
import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# Wikimedia file titles are frequently non-ASCII (accented names, non-Latin
# scripts); Windows' default console codepage (cp1252) can't encode a lot of
# that and crashes mid-run on a plain print(). Reconfigure stdout to UTF-8 so
# a long --all run doesn't die partway through on a title Windows can't
# print.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.content import DOMAIN_LIBRARY, DOMAINS_BY_NAME  # noqa: E402

LIBRARY_PATH = os.path.join(os.path.dirname(__file__), "image_library.json")
API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "TheAgentGameProto/1.0 (pre-production image research; contact: scott)"

# Optional vision-verification step (--vision): a local Ollama vision model
# double-checks a candidate photo actually shows the answer before it's
# accepted, rather than trusting Commons' loose title-text search ranking
# alone (see _title_confirms_relevance below, which this supplements, not
# replaces). Off by default -- the script works exactly as it always has
# without llava pulled.
# 127.0.0.1, not "localhost" -- see ollama_agent.py for why: "localhost" can
# resolve to the IPv6 loopback first on this machine, and Ollama only
# listens on IPv4, so that connection hangs in SYN_SENT for minutes instead
# of failing fast, defeating VISION_TIMEOUT below entirely.
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
VISION_MODEL = "llava:7b"
VISION_TIMEOUT = 90.0

DOMAIN_HINT = {
    "Cats": "cat",
    "Dogs": "dog",
    "Pizza": "pizza",
    "Dinosaurs": "dinosaur",
    "Ocean Animals": "ocean sea",
    "Fair Grounds": "carnival fair",
    "Autumn": "autumn fall",
    "Birthdays": "birthday party",
    "Bicycles": "bicycle",
    "Farm Animals": "farm",
    "Ice Cream": "ice cream",
    "Rainbows": "rainbow",
    "Camping": "camping",
    "Sports Balls": "ball sport",
    "Board Games": "board game",
}

QUERY_OVERRIDES = {
    ("Sports Balls", "five"): "basketball team five players",
    ("Sports Balls", "eleven"): "soccer team players",
    ("Sports Balls", "six"): "volleyball team players",
    ("Sports Balls", "108"): "baseball stitches",
    ("Sports Balls", "strike"): "bowling strike pins",
    ("Sports Balls", "spare"): "bowling pins",
    ("Sports Balls", "oval"): "american football",
    ("Sports Balls", "orange"): "basketball",
    ("Sports Balls", "leather"): "leather baseball",
    ("Sports Balls", "ball size"): "soccer ball size",
    ("Sports Balls", "composite leather"): "basketball texture",
    ("Sports Balls", "hexagons and pentagons"): "soccer ball pattern",
    ("Rainbows", "the sun"): "sun sky",
    ("Rainbows", "gold"): "pot of gold rainbow",
    ("Rainbows", "arch"): "stone arch",
    ("Rainbows", "halo"): "sun halo optical phenomenon",
    ("Rainbows", "red light"): "red light",
    ("Rainbows", "violet light"): "violet light",
    ("Board Games", "go"): "go board game stones",
    ("Board Games", "war"): "playing cards war game",
    ("Board Games", "risk"): "risk board game map",
    ("Board Games", "spinner"): "board game spinner",
    ("Board Games", "sand timer"): "hourglass timer",
    ("Dinosaurs", "arms"): "tyrannosaurus rex arms",
    ("Dinosaurs", "plate"): "stegosaurus plates",
    ("Dinosaurs", "clutch"): "dinosaur eggs nest",
    ("Autumn", "rust"): "rust color autumn leaves",
    ("Autumn", "crunch"): "autumn leaves ground",
    ("Camping", "fry"): "campfire cooking",
    ("Farm Animals", "kid"): "baby goat",
    ("Ice Cream", "lid"): "ice cream cup lid",
    ("Bicycles", "hold it up"): "bicycle kickstand",
    ("Cats", "litter"): "litter of kittens newborn",
    ("Cats", "whiskers"): "cat whiskers close up",
    ("Cats", "bell"): "cat collar bell",
    ("Cats", "suckle"): "kitten nursing mother",
    ("Cats", "chirp"): "cat chattering bird window",
    ("Cats", "head bump"): "cat headbutt affection",
    ("Cats", "tiger"): "bengal tiger",
    ("Cats", "cat kiss"): "cat slow blink eyes",
    ("Cats", "ears"): "cat pointy ears photograph",
    ("Cats", "lily"): "lily flower white",
}

# Answers where hand review found Commons has no correctly-licensed real
# photo worth showing at all (typically a copyrighted character/brand whose
# name collides with an unrelated but well-documented subject on Commons).
# These are permanently skipped rather than retried with ever-more overrides;
# the question just falls back to its plain-text prompt, same as any other
# miss.
PERMANENT_SKIP = {
    ("Cats", "garfield"),  # collides hard with President James Garfield's
                           # papers/memorabilia; the cartoon cat itself is
                           # copyrighted and Commons has no free art of it.
}

BAD_TITLE_WORDS = [
    "flag of", "logo", "seal of", "coat of arms", "diagram", "map of",
    "icon", "wikipedia", "wikimedia", "commons logo", ".svg",
]
GOOD_EXTENSIONS = (".jpg", ".jpeg", ".png")


def build_query(domain: str, answer: str) -> str:
    override = QUERY_OVERRIDES.get((domain, answer))
    if override:
        return override
    hint = DOMAIN_HINT.get(domain, "")
    if hint and hint.split()[0].lower() in answer.lower():
        return answer
    return f"{answer} {hint}".strip()


def search_commons(query: str, limit: int = 6, max_retries: int = 4) -> list:
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": "6",
        "gsrlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size",
        "iiurlwidth": "700",
        "format": "json",
    }
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    backoff = 4.0
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            return list(pages.values())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                print(f"    ... rate limited, backing off {backoff:.0f}s")
                time.sleep(backoff)
                backoff *= 2
                continue
            print(f"    ! request failed for {query!r}: {e}")
            return []
        except Exception as e:
            print(f"    ! request failed for {query!r}: {e}")
            return []
    return []


def _title_confirms_relevance(title: str, query: str) -> bool:
    """Commons' internal search ranking is loose: a query like 'jingle ball
    cat' can surface an unrelated museum print because the ranker matched on
    something else entirely. Require that at least one substantial word
    (4+ letters, so short connector words like 'the'/'a' don't count) from
    the query actually appears in the result's own file title, so a result
    that isn't really about what we asked for gets rejected instead of
    silently accepted."""
    title_l = title.lower()
    words = [w for w in re.findall(r"[a-z']+", query.lower()) if len(w) >= 4]
    if not words:
        return True  # nothing substantial to check against, don't block
    return any(w in title_l for w in words)


def verify_with_vision(url: str, domain_name: str, answer: str) -> bool:
    """Downloads a candidate photo and asks a local vision model whether it
    actually shows the answer -- AND whether it's simple and clear enough
    for someone to recognize at a glance, not just technically on-topic.
    This library is meant to be plain and easy to read during a live duel,
    not an art-museum piece, so a cluttered, busy, or multi-subject photo
    should fail this even if it does contain the right thing somewhere in
    frame. Any failure (download error, Ollama unreachable, unparseable
    reply) counts as a rejection, not a pass -- here in pre-production, a
    false negative (falls back to the next candidate, or the plain text
    prompt) is always safer than a false positive (a wrong or confusing
    photo shown live on the board)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=15) as resp:
            image_bytes = resp.read()
    except (urllib.error.URLError, OSError):
        return False
    body = json.dumps({
        "model": VISION_MODEL,
        "prompt": f"Does this image clearly and simply show '{answer}' ({domain_name} "
                  f"context), in a way an average person could recognize at a glance? "
                  f"Reject it if it's cluttered, busy, abstract, has multiple competing "
                  f"subjects, or would take more than a second or two to make sense of. "
                  f"Reply with just YES or NO.",
        "images": [base64.b64encode(image_bytes).decode("ascii")],
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=VISION_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return False
    return (data.get("response") or "").strip().upper().startswith("YES")


def pick_best(pages: list, query: str = "", verify=None):
    for page in pages:
        title = page.get("title", "")
        info = (page.get("imageinfo") or [None])[0]
        if not info:
            continue
        if any(bad in title.lower() for bad in BAD_TITLE_WORDS):
            continue
        if query and not _title_confirms_relevance(title, query):
            continue
        url = info.get("thumburl") or info.get("url")
        if not url or not url.lower().endswith(GOOD_EXTENSIONS):
            continue
        width = info.get("thumbwidth") or info.get("width") or 0
        height = info.get("thumbheight") or info.get("height") or 0
        if width < 250 or height < 180:
            continue
        if verify is not None and not verify(url):
            continue
        meta = info.get("extmetadata", {})
        artist = re.sub("<[^>]+>", "", meta.get("Artist", {}).get("value", "")).strip()
        license_short = meta.get("LicenseShortName", {}).get("value", "")
        credit_bits = [b for b in [artist, license_short] if b]
        credit = " / ".join(credit_bits) if credit_bits else "Wikimedia Commons"
        return {"url": url, "credit": f"{credit} (Wikimedia Commons)", "title": title}
    return None


def load_library() -> dict:
    if os.path.exists(LIBRARY_PATH):
        with open(LIBRARY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_library(library: dict) -> None:
    """Atomic write: a kill mid-write (this script gets timed out and
    terminated externally fairly often) must never leave image_library.json
    truncated or empty, since that would silently wipe out every entry
    fetched so far. Write to a sibling temp file first, then os.replace it
    into place; a rename is atomic, so the file on disk is always either
    the old complete version or the new complete version, never a partial
    write caught mid-flight."""
    tmp_path = LIBRARY_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(library, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, LIBRARY_PATH)


def fetch_domain(domain_name, library, delay=1.2, limit=None, use_vision=False):
    domain = DOMAINS_BY_NAME[domain_name]
    found, missed, skipped = 0, 0, 0
    fetched_this_call = 0
    for q in domain.questions:
        key = f"{domain_name}|||{q.image_prompt}"
        if key in library:
            skipped += 1
            continue
        if limit is not None and fetched_this_call >= limit:
            break
        if (domain_name, q.answer) in PERMANENT_SKIP:
            missed += 1
            fetched_this_call += 1
            print(f"    SKIP {q.answer!r:32s} (no reliable free photo exists)")
            save_library(library)
            continue
        query = build_query(domain_name, q.answer)
        pages = search_commons(query)
        verify = (lambda u, a=q.answer: verify_with_vision(u, domain_name, a)) if use_vision else None
        result = pick_best(pages, query, verify=verify)
        fetched_this_call += 1
        if result:
            library[key] = {"url": result["url"], "credit": result["credit"]}
            found += 1
            print(f"    OK   {q.answer!r:32s} -> {result['title']}")
        else:
            missed += 1
            print(f"    MISS {q.answer!r:32s} (query: {query!r})")
        save_library(library)  # save after every single item so a killed/timed-out
        # call never loses progress already made within it
        time.sleep(delay)
    remaining = len(domain.questions) - skipped - found - missed
    print(f"  {domain_name}: {found} found, {missed} missed, {skipped} cached, {remaining} left of {len(domain.questions)}")
    return library


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", help="Fetch just one domain by name")
    parser.add_argument("--all", action="store_true", help="Fetch every domain")
    parser.add_argument("--limit", type=int, default=None,
                         help="Cap how many not-yet-cached questions to fetch this call, "
                              "so a long run can be split across several short calls")
    parser.add_argument("--vision", action="store_true",
                         help="Verify each candidate photo with a local llava vision model "
                              "(`ollama pull llava:7b`) before accepting it, instead of "
                              "trusting Commons' title-text search ranking alone. Slower "
                              "(one extra local model call per candidate) but catches "
                              "photos that only matched on title text, not content.")
    args = parser.parse_args()

    library = load_library()

    if args.domain:
        names = [args.domain]
    elif args.all:
        names = [d.name for d in DOMAIN_LIBRARY]
    else:
        parser.error("pass --domain NAME or --all")
        return

    for name in names:
        print(f"Fetching {name}...")
        library = fetch_domain(name, library, limit=args.limit, use_vision=args.vision)
        save_library(library)

    print(f"\nWrote {len(library)} total image entries to {LIBRARY_PATH}")


if __name__ == "__main__":
    main()
