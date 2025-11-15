#!/usr/bin/env python3
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

import requests
import trafilatura
import yaml
from openai import OpenAI


def setup_logging():
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        stream=sys.stderr,
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def env(name, default=None, required=False):
    value = os.getenv(name, default)
    if required and not value:
        logging.error("Missing required environment variable %s", name)
        sys.exit(1)
    return value


def strip_html(html_text: str) -> str:
    import html as html_mod
    import re

    if not html_text:
        return ""
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html_text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_mod.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_lines_generic(text: str) -> str:
    """Drop obvious menu/boilerplate lines: tiny, punctuation-only, etc."""
    if not text:
        return ""
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        # Short lines are usually navigation labels etc.
        if len(s) <= 3:
            continue
        # Lines that are mostly punctuation / bullets
        if all(c in "-•|·.,:;_/\\[]" for c in s):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def looks_useful(text: str) -> bool:
    """Heuristic to decide whether content is worth sending to the model."""
    if not text:
        return False
    # Minimum length: be strict; we prefer missing stuff to junk
    if len(text) < 300:
        return False
    # Require at least a few sentence boundaries
    sentence_like = text.count(".") + text.count("!") + text.count("?")
    if sentence_like < 3:
        return False
    return True


def load_yaml_config(path: str | None) -> tuple[dict, dict]:
    if not path:
        logging.info("No FEED_CONFIG_PATH provided; using empty YAML config")
        return {}, {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logging.warning(
            "YAML config file not found at %s; proceeding with defaults", path
        )
        return {}, {}
    except Exception as exc:
        logging.warning("Failed to load YAML config from %s: %s", path, exc)
        return {}, {}

    if not isinstance(data, dict):
        logging.warning("YAML config root is not a mapping; ignoring")
        return {}, {}

    global_cfg = data.get("global") or {}
    feeds_cfg = data.get("feeds") or {}
    if not isinstance(global_cfg, dict):
        global_cfg = {}
    if not isinstance(feeds_cfg, dict):
        feeds_cfg = {}

    logging.info(
        "Loaded YAML config: %d feed entries, global keys: %s",
        len(feeds_cfg),
        ", ".join(global_cfg.keys()) or "(none)",
    )
    return global_cfg, feeds_cfg


def authenticate_greader(
    base_url: str, username: str, api_password: str, timeout: int = 20
) -> str:
    url = base_url.rstrip("/") + "/accounts/ClientLogin"
    logging.info("Authenticating to FreshRSS Google Reader API as %s", username)
    resp = requests.post(
        url,
        data={"Email": username, "Passwd": api_password},
        timeout=timeout,
    )
    if resp.status_code != 200:
        logging.error("Authentication failed with HTTP %s", resp.status_code)
        sys.exit(1)
    auth_token = None
    for line in resp.text.splitlines():
        if line.startswith("Auth="):
            auth_token = line.split("=", 1)[1].strip()
            break
    if not auth_token:
        logging.error("Authentication response did not contain Auth token")
        sys.exit(1)
    logging.info("Authentication successful")
    return auth_token


def fetch_subscriptions(base_url: str, auth_token: str, timeout: int = 20) -> dict:
    url = base_url.rstrip("/") + "/reader/api/0/subscription/list"
    params = {"output": "json"}
    headers = {"Authorization": "GoogleLogin auth=" + auth_token}
    logging.info("Fetching subscriptions")
    resp = requests.get(url, headers=headers, params=params, timeout=timeout)
    if resp.status_code != 200:
        logging.error("Failed to fetch subscriptions, HTTP %s", resp.status_code)
        sys.exit(1)
    data = resp.json()
    subs = data.get("subscriptions", [])
    logging.info("Fetched %d subscriptions", len(subs))
    sub_map = {}
    for s in subs:
        sid = s.get("id")
        if sid:
            sub_map[sid] = s
    return sub_map


def fetch_unread_items(
    base_url: str, auth_token: str, max_items: int, timeout: int = 30
) -> list:
    url = base_url.rstrip("/") + "/reader/api/0/stream/contents/reading-list"
    params = {
        "n": str(max_items),
        "xt": "user/-/state/com.google/read",
        "output": "json",
    }
    headers = {"Authorization": "GoogleLogin auth=" + auth_token}
    logging.info("Fetching up to %d unread items", max_items)
    resp = requests.get(url, headers=headers, params=params, timeout=timeout)
    if resp.status_code != 200:
        logging.error("Failed to fetch unread items, HTTP %s", resp.status_code)
        sys.exit(1)
    data = resp.json()
    items = data.get("items", [])
    logging.info("Fetched %d unread items", len(items))
    return items


def extract_main_text(url: str) -> str | None:
    """Use trafilatura as aggressively as possible, biasing toward precision."""
    if not url:
        return None
    try:
        downloaded = trafilatura.fetch_url(url)
    except Exception as exc:
        logging.debug("trafilatura fetch_url failed for %s: %s", url, exc)
        downloaded = None

    text = None
    if downloaded:
        try:
            text = trafilatura.extract(
                filecontent=downloaded,
                include_comments=False,
                include_formatting=False,
                include_links=False,
                favor_precision=True,
            )
        except Exception as exc:
            logging.debug("trafilatura extract(content) failed for %s: %s", url, exc)

    if not text:
        try:
            text = trafilatura.extract(
                filecontent=None,
                url=url,
                include_comments=False,
                include_formatting=False,
                include_links=False,
                favor_precision=True,
            )
        except Exception as exc:
            logging.debug("trafilatura extract(url=...) failed for %s: %s", url, exc)

    if text:
        cleaned = text.strip()
        if cleaned:
            return cleaned
    return None


def normalize_item(
    item: dict,
    subscriptions: dict,
    fetch_full_content: bool,
    article_max_chars: int,
) -> dict:
    origin = item.get("origin") or {}
    stream_id = origin.get("streamId")
    subscription = subscriptions.get(stream_id, {}) if stream_id else {}

    # Feed URL: prefer subscription["url"] (canonical RSS/Atom URL)
    feed_url = subscription.get("url") or ""

    # Basic article fields
    summary_html = ""
    summary_obj = item.get("summary") or {}
    if isinstance(summary_obj, dict):
        summary_html = summary_obj.get("content") or ""
    if not summary_html:
        content_obj = item.get("content") or {}
        if isinstance(content_obj, dict):
            summary_html = content_obj.get("content") or ""

    alternate = item.get("alternate") or []
    url = ""
    if alternate and isinstance(alternate, list):
        first_alt = alternate[0] or {}
        url = first_alt.get("href") or ""

    published_ts = item.get("published") or 0
    try:
        published_iso = datetime.fromtimestamp(
            int(published_ts), tz=timezone.utc
        ).isoformat()
    except Exception:
        published_iso = ""

    content_text = ""

    # 1) Prefer full-page extraction
    if fetch_full_content and url:
        content_text = extract_main_text(url) or ""

    # 2) Fallback: cleaned feed HTML, but only if non-trivial
    if not content_text:
        raw = strip_html(summary_html)
        if len(raw) >= 200:  # be strict: prefer missing over junk
            content_text = raw
        else:
            content_text = ""

    # 3) Generic boilerplate cleaning
    content_text = clean_lines_generic(content_text)

    # 4) Hard length cap
    if content_text and article_max_chars > 0 and len(content_text) > article_max_chars:
        content_text = content_text[:article_max_chars]

    return {
        "item_id": item.get("id") or "",
        "feed_url": feed_url,
        "title": item.get("title") or "",
        "published": published_iso,
        "content": content_text,
    }


def apply_feed_stripping(content_text: str, feed_cfg: dict) -> str:
    """Feed-specific stripping rules from YAML (strip_contains)."""
    if not content_text:
        return ""
    strip_contains = feed_cfg.get("strip_contains") or []
    if not strip_contains:
        return content_text

    lowered_patterns = [
        pattern.lower()
        for pattern in strip_contains
        if isinstance(pattern, str) and pattern
    ]
    if not lowered_patterns:
        return content_text

    lines = content_text.splitlines()
    kept_lines = []
    for raw_line in lines:
        lowered_line = raw_line.lower()
        if any(pattern in lowered_line for pattern in lowered_patterns):
            continue
        kept_lines.append(raw_line)
    return "\n".join(kept_lines)


def build_summarization_payload_and_item_ids(
    items: list,
    subscriptions: dict,
    feeds_cfg: dict,
    summary_goal: str,
    fetch_full_content: bool,
    article_max_chars: int,
) -> tuple[str, list[str]]:
    feeds_by_url: dict[str, dict[str, Any]] = {}

    for item in items:
        norm = normalize_item(
            item, subscriptions, fetch_full_content, article_max_chars
        )
        item_id = norm.get("item_id") or ""
        feed_url = norm.get("feed_url") or ""
        title = norm.get("title") or ""
        content = norm.get("content") or ""

        if not feed_url:
            logging.debug("Skipping item without feed_url (id=%s)", item_id)
            continue

        feed_cfg = feeds_cfg.get(feed_url, {}) if isinstance(feeds_cfg, dict) else {}

        # Feed-specific stripping
        content = apply_feed_stripping(content, feed_cfg)
        content = clean_lines_generic(content)  # one more pass after strip

        # Strict usefulness check
        if not looks_useful(content):
            logging.debug(
                "Skipping low-value content for feed=%s id=%s", feed_url, item_id
            )
            continue

        if not (title or content):
            logging.debug("Skipping item without title/content (id=%s)", item_id)
            continue

        priority = int(feed_cfg.get("priority", 0) or 0)
        description = str(feed_cfg.get("description", "") or "")
        representation = str(feed_cfg.get("representation", "") or "")

        feed_data = feeds_by_url.get(feed_url)
        if not feed_data:
            feed_data = {
                "priority": priority,
                "description": description,
                "representation": representation,
                "articles": [],
                "item_ids": [],
            }
            feeds_by_url[feed_url] = feed_data

        feed_data["articles"].append(
            {
                "title": title,
                "content": content,
                "published": norm.get("published") or "",
            }
        )
        if item_id:
            feed_data["item_ids"].append(item_id)

    payload: dict[str, Any]

    if not feeds_by_url:
        logging.info("No items with usable content for summarization after filtering")
        payload = {
            "summary_goal": summary_goal,
            "feeds": [],
        }
        return json.dumps(payload, ensure_ascii=False), []

    # Sort articles per feed by published time
    for feed_data in feeds_by_url.values():
        feed_data["articles"].sort(key=lambda a: a.get("published") or "")

    # Prepare payload sorted by priority (descending)
    sorted_feeds: list[dict[str, Any]] = sorted(
        feeds_by_url.values(),
        key=lambda f: f["priority"],
        reverse=True,
    )

    payload_feeds: list[dict[str, Any]] = [
        {
            "priority": f["priority"],
            "description": f["description"],
            "representation": f["representation"],
            "articles": f["articles"],
        }
        for f in sorted_feeds
    ]

    payload = {
        "summary_goal": summary_goal,
        "feeds": payload_feeds,
    }

    # Unique list of item IDs to mark as read
    item_ids: list[str] = []
    seen: set[str] = set()
    for f in feeds_by_url.values():
        for iid in f["item_ids"]:
            if iid and iid not in seen:
                seen.add(iid)
                item_ids.append(iid)

    payload_json = json.dumps(payload, ensure_ascii=False)
    logging.info(
        "Built summarization payload: %d feeds, %d items, %d chars JSON",
        len(sorted_feeds),
        len(item_ids),
        len(payload_json),
    )

    return payload_json, item_ids


def call_openai(
    model: str,
    payload_json: str,
    max_output_tokens: int,
    model_instructions: str | None,
) -> str:
    client = OpenAI()
    logging.info("Calling OpenAI model %s via Responses API", model)

    base_instructions = (
        model_instructions.strip()
        if model_instructions and model_instructions.strip()
        else (
            "You generate a single, elegantly written daily digest for the user.\n"
            "You receive JSON containing:\n"
            "- summary_goal: how the user wants the digest to feel "
            "and what to prioritize.\n"
            "- feeds: an array where each feed has:\n"
            "  - priority (higher = more important to the user)\n"
            "  - description (what this feed usually covers)\n"
            "  - representation (how the user wants this feed to show up "
            "in the digest)\n"
            "  - articles: with title, main text content, and publication time.\n\n"
            "Use the per-feed priority, description, and representation fields "
            "as strong signals of the user's intent."
        )
    )

    structure_instructions = (
        "\n\nWrite a single, unified summary organized by themes and importance, "
        "not by source.\n"
        "- Do not create sections named after feeds.\n"
        "- Focus on what is maximally relevant and high-signal for the user.\n"
        "- Use clear headings and short paragraphs where helpful.\n"
        "- Return only the inner HTML for a single <article> element "
        "(no <html> or <body>)."
    )

    instructions = base_instructions + structure_instructions

    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=(
            "Here is the JSON input for today's unread items. "
            "Write the best possible article for this user.\n\n" + payload_json
        ),
        max_output_tokens=max_output_tokens,
        temperature=0.3,
    )
    html_fragment = response.output_text or ""
    return html_fragment.strip()


def ensure_article_wrapper(inner_html: str) -> str:
    text = inner_html.strip()
    if "<article" in text.lower():
        return text
    return f"<article>\n{text}\n</article>"


def format_output(article_html: str, output_format: str) -> str:
    output_format = (output_format or "article").lower()
    if output_format not in ("text", "html_page", "article"):
        logging.warning(
            "Unknown OUTPUT_FORMAT %r, defaulting to 'article'", output_format
        )
        output_format = "article"

    if output_format == "article":
        return ensure_article_wrapper(article_html)

    if output_format == "html_page":
        article_block = ensure_article_wrapper(article_html)
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n'
            "<head>\n"
            '<meta charset="utf-8">\n'
            "<title>Daily Feed Synthesis</title>\n"
            "</head>\n"
            "<body>\n"
            f"{article_block}\n"
            "</body>\n"
            "</html>\n"
        )

    # Plain text
    article_block = ensure_article_wrapper(article_html)
    text = strip_html(article_block)
    return text + "\n"


def get_token_for_write(
    base_url: str, auth_token: str, timeout: int = 10
) -> str | None:
    url = base_url.rstrip("/") + "/reader/api/0/token"
    headers = {"Authorization": "GoogleLogin auth=" + auth_token}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
    except Exception as exc:
        logging.warning("Failed to fetch write token: %s", exc)
        return None
    if resp.status_code != 200:
        logging.warning("Token request failed with HTTP %s", resp.status_code)
        return None
    token = resp.text.strip()
    if not token:
        logging.warning("Empty token received from token endpoint")
        return None
    logging.info("Obtained write token")
    return token


def mark_items_read(base_url: str, auth_token: str, item_ids: list[str]):
    if not item_ids:
        logging.info("No items to mark as read")
        return

    token = get_token_for_write(base_url, auth_token)
    if not token:
        logging.warning("Cannot mark items read without token")
        return

    url = base_url.rstrip("/") + "/reader/api/0/edit-tag"
    headers = {"Authorization": "GoogleLogin auth=" + auth_token}

    data = [("T", token), ("a", "user/-/state/com.google/read")]
    for iid in item_ids:
        data.append(("i", iid))

    try:
        resp = requests.post(url, headers=headers, data=data, timeout=20)
    except Exception as exc:
        logging.warning("Failed to mark items as read: %s", exc)
        return

    if resp.status_code != 200:
        logging.warning("Mark-read request failed with HTTP %s", resp.status_code)
        return

    logging.info("Marked %d items as read in FreshRSS", len(item_ids))


def main():
    setup_logging()
    logging.info("Starting FreshRSS greader summarizer run")

    base_url = env("FRESHRSS_API_URL", required=True)
    username = env("FRESHRSS_USERNAME", required=True)
    api_password = env("FRESHRSS_API_PASSWORD", required=True)
    feed_config_path = env("FEED_CONFIG_PATH", required=False)

    global_cfg, feeds_cfg = load_yaml_config(feed_config_path)
    summary_goal = global_cfg.get(
        "summary_goal",
        "Write a single, elegantly written synthesis of these unread items, "
        "tailored to what this user cares about.",
    )
    model_instructions = global_cfg.get("model_instructions")

    openai_model = env("OPENAI_MODEL", required=False, default="gpt-4.1-mini")
    max_items = int(env("MAX_ITEMS", required=False, default="200"))
    max_output_tokens = int(env("MAX_OUTPUT_TOKENS", required=False, default="1200"))
    output_format = env("OUTPUT_FORMAT", required=False, default="article")
    fetch_full_content = (
        env("FETCH_FULL_CONTENT", required=False, default="true").lower() == "true"
    )
    article_max_chars = int(env("ARTICLE_MAX_CHARS", required=False, default="4000"))

    try:
        auth_token = authenticate_greader(base_url, username, api_password)
        subscriptions = fetch_subscriptions(base_url, auth_token)
        items = fetch_unread_items(base_url, auth_token, max_items)

        if not items:
            logging.info("No unread items; generating empty output")
            placeholder = (
                "<article><h1>No new items</h1>"
                "<p>You are fully caught up. There are no unread articles for this "
                "period.</p>"
                "</article>"
            )
            sys.stdout.write(format_output(placeholder, output_format))
            return

        payload_json, item_ids = build_summarization_payload_and_item_ids(
            items,
            subscriptions,
            feeds_cfg,
            summary_goal,
            fetch_full_content,
            article_max_chars,
        )

        if not item_ids:
            logging.info(
                "No items with usable content to summarize; skipping OpenAI call"
            )
            placeholder = (
                "<article><h1>Nothing worth summarizing</h1>"
                "<p>Unread items were present, but none had sufficiently useful "
                "content.</p>"
                "</article>"
            )
            sys.stdout.write(format_output(placeholder, output_format))
            return

        article_inner_html = call_openai(
            openai_model, payload_json, max_output_tokens, model_instructions
        )
        final_output = format_output(article_inner_html, output_format)
        sys.stdout.write(final_output)

        # Mark all considered items as read
        mark_items_read(base_url, auth_token, item_ids)

        logging.info("Run completed successfully")

    except Exception as exc:
        logging.exception("Run failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
