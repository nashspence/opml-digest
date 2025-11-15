#!/usr/bin/env python3
import os
import sys
import json
import logging
from datetime import datetime, timezone

import requests
import trafilatura
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
    import re
    import html as html_mod

    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html_text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_mod.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def authenticate_greader(base_url: str, username: str, api_password: str, timeout: int = 20) -> str:
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


def fetch_unread_items(base_url: str, auth_token: str, max_items: int, timeout: int = 30) -> list:
    url = base_url.rstrip("/") + "/reader/api/0/stream/contents/reading-list"
    params = {
        "n": max_items,
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


def fetch_feed_descriptions(base_url: str, auth_token: str, stream_ids: set, timeout: int = 20) -> dict:
    headers = {"Authorization": "GoogleLogin auth=" + auth_token}
    feed_descriptions: dict[str, str] = {}
    if not stream_ids:
        return feed_descriptions
    logging.info("Fetching feed descriptions for %d feeds", len(stream_ids))
    for sid in stream_ids:
        url = base_url.rstrip("/") + "/reader/api/0/stream/contents/" + sid
        params = {"n": 0, "output": "json"}
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout)
        except Exception as exc:
            logging.warning("Error fetching description for %s: %s", sid, exc)
            continue
        if resp.status_code != 200:
            logging.warning("Failed to fetch feed metadata for %s, HTTP %s", sid, resp.status_code)
            continue
        try:
            data = resp.json()
        except Exception as exc:
            logging.warning("Invalid JSON for feed %s: %s", sid, exc)
            continue
        desc = data.get("description")
        if isinstance(desc, str) and desc.strip():
            feed_descriptions[sid] = desc.strip()
    logging.info("Resolved descriptions for %d feeds", len(feed_descriptions))
    return feed_descriptions


def extract_main_text(url: str) -> str | None:
    if not url:
        return None
    try:
        downloaded = trafilatura.fetch_url(url)
    except Exception as exc:
        logging.debug("trafilatura fetch_url failed for %s: %s", url, exc)
        return None
    if not downloaded:
        return None
    try:
        text = trafilatura.extract(downloaded, include_comments=False)
    except Exception as exc:
        logging.debug("trafilatura extract failed for %s: %s", url, exc)
        return None
    if text:
        return text.strip()
    return None


def normalize_item(
    item: dict,
    subscriptions: dict,
    feed_descriptions: dict,
    fetch_full_content: bool,
    article_max_chars: int,
) -> dict:
    origin = item.get("origin") or {}
    stream_id = origin.get("streamId")
    subscription = subscriptions.get(stream_id, {}) if stream_id else {}
    categories = subscription.get("categories") or []
    category_labels = [
        c.get("label") for c in categories if isinstance(c, dict) and c.get("label")
    ]
    feed_title = origin.get("title") or subscription.get("title") or ""
    feed_html_url = (
        origin.get("htmlUrl")
        or subscription.get("htmlUrl")
        or subscription.get("url")
        or ""
    )
    feed_description = (
        feed_descriptions.get(stream_id)
        or subscription.get("description")
        or ""
    )
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
    if fetch_full_content and url:
        content_text = extract_main_text(url) or ""
    if not content_text:
        content_text = strip_html(summary_html) if summary_html else ""
    if content_text and article_max_chars > 0 and len(content_text) > article_max_chars:
        content_text = content_text[:article_max_chars]
    return {
        "id": item.get("id") or "",
        "title": item.get("title") or "",
        "url": url,
        "published": published_iso,
        "content": content_text,
        "feed_stream_id": stream_id or "",
        "feed_title": feed_title,
        "feed_html_url": feed_html_url,
        "feed_description": feed_description,
        "feed_categories": category_labels,
    }


def build_summarization_payload(
    username: str,
    summary_goal: str,
    items: list,
    subscriptions: dict,
    feed_descriptions: dict,
    fetch_full_content: bool,
    article_max_chars: int,
) -> str:
    normalized = [
        normalize_item(i, subscriptions, feed_descriptions, fetch_full_content, article_max_chars)
        for i in items
    ]
    normalized = [n for n in normalized if n["title"] or n["content"]]
    normalized.sort(key=lambda n: n.get("published") or "", reverse=True)
    feed_context: dict[str, dict] = {}
    for n in normalized:
        sid = n["feed_stream_id"]
        if not sid:
            continue
        if sid not in feed_context:
            feed_context[sid] = {
                "id": sid,
                "title": n["feed_title"],
                "html_url": n["feed_html_url"],
                "description": n["feed_description"],
                "categories": n["feed_categories"],
            }
    payload = {
        "user": {
            "username": username,
        },
        "summary_goal": summary_goal,
        "feeds": list(feed_context.values()),
        "articles": normalized,
    }
    payload_json = json.dumps(payload, ensure_ascii=False)
    logging.info(
        "Built summarization payload: %d feeds, %d articles, %d chars JSON",
        len(feed_context),
        len(normalized),
        len(payload_json),
    )
    return payload_json


def call_openai(model: str, payload_json: str, max_output_tokens: int) -> str:
    client = OpenAI()
    logging.info("Calling OpenAI model %s via Responses API", model)
    response = client.responses.create(
        model=model,
        instructions=(
            "You write a single, elegantly written HTML article that synthesizes many RSS items "
            "for one user. The input is JSON with a summary_goal, feed metadata (titles, URLs, "
            "descriptions, categories) and an array of articles (titles, URLs, publication times, "
            "and main content text).\n\n"
            "Use feed titles, descriptions and categories as strong signals of what the user values. "
            "Prioritize what is maximally relevant to them and de-emphasize noise. Do not create "
            "sections per source; instead, weave a unified narrative organized by themes and importance. "
            "Use natural prose and link phrases with <a href=\"...\">…</a> when specific stories are "
            "referenced.\n\n"
            "Return only the inner HTML of a single <article> element (no <html> or <body>)."
        ),
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
        logging.warning("Unknown OUTPUT_FORMAT %r, defaulting to 'article'", output_format)
        output_format = "article"
    if output_format == "article":
        return ensure_article_wrapper(article_html)
    if output_format == "html_page":
        article_block = ensure_article_wrapper(article_html)
        return (
            "<!DOCTYPE html>\n"
            "<html lang=\"en\">\n"
            "<head>\n"
            "<meta charset=\"utf-8\">\n"
            "<title>Daily Feed Synthesis</title>\n"
            "</head>\n"
            "<body>\n"
            f"{article_block}\n"
            "</body>\n"
            "</html>\n"
        )
    article_block = ensure_article_wrapper(article_html)
    text = strip_html(article_block)
    return text + "\n"


def main():
    setup_logging()
    logging.info("Starting FreshRSS greader summarizer run")
    base_url = env("FRESHRSS_API_URL", required=True)
    username = env("FRESHRSS_USERNAME", required=True)
    api_password = env("FRESHRSS_API_PASSWORD", required=True)
    summary_goal = env(
        "SUMMARY_PROMPT",
        required=False,
        default=(
            "Write a single, elegantly written synthesis of these unread items, "
            "tailored to what this user cares about."
        ),
    )
    openai_model = env("OPENAI_MODEL", required=False, default="gpt-4.1-mini")
    max_items = int(env("MAX_ITEMS", required=False, default="200"))
    max_output_tokens = int(env("MAX_OUTPUT_TOKENS", required=False, default="1200"))
    output_format = env("OUTPUT_FORMAT", required=False, default="article")
    fetch_full_content = env("FETCH_FULL_CONTENT", required=False, default="true").lower() == "true"
    article_max_chars = int(env("ARTICLE_MAX_CHARS", required=False, default="4000"))
    try:
        auth_token = authenticate_greader(base_url, username, api_password)
        subscriptions = fetch_subscriptions(base_url, auth_token)
        items = fetch_unread_items(base_url, auth_token, max_items)
        if not items:
            logging.info("No unread items; generating empty output")
            placeholder = (
                "<article><h1>No new items</h1>"
                "<p>You are fully caught up. There are no unread articles for this period.</p>"
                "</article>"
            )
            sys.stdout.write(format_output(placeholder, output_format))
            return
        stream_ids: set[str] = set()
        for i in items:
            origin = i.get("origin") or {}
            sid = origin.get("streamId")
            if sid:
                stream_ids.add(sid)
        feed_descriptions = fetch_feed_descriptions(base_url, auth_token, stream_ids)
        payload_json = build_summarization_payload(
            username,
            summary_goal,
            items,
            subscriptions,
            feed_descriptions,
            fetch_full_content,
            article_max_chars,
        )
        article_inner_html = call_openai(openai_model, payload_json, max_output_tokens)
        final_output = format_output(article_inner_html, output_format)
        sys.stdout.write(final_output)
        logging.info("Run completed successfully")
    except Exception as exc:
        logging.exception("Run failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
