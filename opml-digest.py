#!/usr/bin/env python3
import sys
import os
import json
import logging
from datetime import datetime, timezone
from xml.etree import ElementTree as ET

import feedparser
import trafilatura
from dateutil import parser as dateparser
from openai import OpenAI


def setup_logging():
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        stream=sys.stderr,
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        logging.error("Missing required environment variable: %s", name)
        sys.exit(1)
    return value


def parse_since_date(arg: str) -> datetime:
    dt = dateparser.parse(arg)
    if dt is None:
        logging.error("Could not parse since-date argument: %s", arg)
        sys.exit(1)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def parse_opml(opml_path: str):
    try:
        tree = ET.parse(opml_path)
    except Exception as exc:
        logging.error("Failed to parse OPML file %s: %s", opml_path, exc)
        sys.exit(1)

    root = tree.getroot()
    head = root.find("head")
    if head is None:
        logging.error("OPML must have a <head> element")
        sys.exit(1)

    # Required head fields: prompt, openaiModel, maxOutputTokens
    prompt_elem = head.find("prompt")
    model_elem = head.find("openaiModel")
    tokens_elem = head.find("maxOutputTokens")

    if prompt_elem is None or not (prompt_elem.text and prompt_elem.text.strip()):
        logging.error("OPML <head><prompt>...</prompt></head> is required and must be non-empty")
        sys.exit(1)
    prompt = prompt_elem.text.strip()

    if model_elem is None or not (model_elem.text and model_elem.text.strip()):
        logging.error("OPML <head><openaiModel>...</openaiModel></head> is required and must be non-empty")
        sys.exit(1)
    openai_model = model_elem.text.strip()

    if tokens_elem is None or not (tokens_elem.text and tokens_elem.text.strip()):
        logging.error("OPML <head><maxOutputTokens>...</maxOutputTokens></head> is required and must be non-empty")
        sys.exit(1)
    try:
        max_output_tokens = int(tokens_elem.text.strip())
    except ValueError:
        logging.error("maxOutputTokens must be an integer, got: %s", tokens_elem.text)
        sys.exit(1)

    feeds = []
    for node in root.findall(".//outline"):
        xml_url = node.attrib.get("xmlUrl") or node.attrib.get("xmlURL")
        if not xml_url:
            continue

        title = node.attrib.get("title") or node.attrib.get("text") or ""

        priority_str = node.attrib.get("priority")
        if priority_str is None:
            logging.error("Feed %s is missing required 'priority' attribute", xml_url)
            sys.exit(1)
        try:
            priority = int(priority_str)
        except ValueError:
            logging.error("Feed %s has non-integer 'priority': %s", xml_url, priority_str)
            sys.exit(1)

        description = node.attrib.get("description")
        if description is None:
            logging.error("Feed %s is missing required 'description' attribute", xml_url)
            sys.exit(1)

        representation = node.attrib.get("representation")
        if representation is None:
            logging.error("Feed %s is missing required 'representation' attribute", xml_url)
            sys.exit(1)

        feeds.append(
            {
                "xml_url": xml_url,
                "title": title,
                "priority": priority,
                "description": description,
                "representation": representation,
            }
        )

    if not feeds:
        logging.error("No feeds with xmlUrl found in OPML")
        sys.exit(1)

    logging.info(
        "Loaded %d feeds from OPML; model=%s, max_output_tokens=%d",
        len(feeds),
        openai_model,
        max_output_tokens,
    )

    return prompt, openai_model, max_output_tokens, feeds


def entry_datetime(entry) -> datetime | None:
    t = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if not t:
        return None
    try:
        dt = datetime(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec, tzinfo=timezone.utc)
    except Exception:
        return None
    return dt


def scrape_entry_content(url: str) -> str | None:
    if not url:
        return None
    text = None
    try:
        text = trafilatura.extract(
            url=url,
            include_comments=False,
            include_formatting=False,
            include_links=False,
            favor_precision=True,
        )
    except Exception as exc:
        logging.debug("trafilatura extract(url=...) failed for %s: %s", url, exc)

    if text:
        text = text.strip()
        if text:
            return text
    return None


def fetch_feed_articles(feed_cfg: dict, since_dt: datetime):
    url = feed_cfg["xml_url"]
    logging.info("Fetching feed %s (%s)", feed_cfg.get("title") or "", url)
    parsed = feedparser.parse(url)
    entries = parsed.entries or []
    articles = []

    for entry in entries:
        dt = entry_datetime(entry)
        if not dt or dt <= since_dt:
            continue

        link = getattr(entry, "link", "") or ""
        content = scrape_entry_content(link)
        if not content:
            continue

        title = getattr(entry, "title", "") or ""
        published_iso = dt.isoformat()

        articles.append(
            {
                "title": title,
                "published": published_iso,
                "content": content,
            }
        )

    logging.info(
        "Feed %s produced %d scraped articles after %s",
        feed_cfg.get("title") or "",
        len(articles),
        since_dt.isoformat(),
    )
    return articles


def build_payload(prompt: str, feeds_cfg: list[dict], since_dt: datetime) -> dict:
    feeds_payload = []
    for feed in sorted(feeds_cfg, key=lambda f: f["priority"], reverse=True):
        articles = fetch_feed_articles(feed, since_dt)
        if not articles:
            continue
        feeds_payload.append(
            {
                "title": feed["title"],
                "priority": feed["priority"],
                "description": feed["description"],
                "representation": feed["representation"],
                "articles": articles,
            }
        )

    if not feeds_payload:
        logging.info("No scraped articles after the given date; nothing to send to OpenAI")
        return {}

    payload = {
        "prompt": prompt,
        "since": since_dt.isoformat(),
        "feeds": feeds_payload,
    }
    logging.info(
        "Built payload with %d feeds and total %d articles",
        len(feeds_payload),
        sum(len(f["articles"]) for f in feeds_payload),
    )
    return payload


def call_openai(model: str, payload: dict, max_output_tokens: int) -> str:
    client = OpenAI()
    payload_json = json.dumps(payload, ensure_ascii=False)
    logging.info("Calling OpenAI model %s via Responses API (payload chars=%d)", model, len(payload_json))

    instructions = (
        "You will receive a single JSON object.\n"
        "- The 'prompt' field tells you exactly what to do and how the user wants the output formatted.\n"
        "- The 'feeds' array contains the content you should base your response on.\n"
        "Follow the 'prompt' as closely as possible, using the provided feeds and articles as context."
    )

    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=payload_json,
        max_output_tokens=max_output_tokens,
        temperature=0.3,
    )
    return (response.output_text or "").strip()


def main():
    setup_logging()

    if len(sys.argv) != 3:
        logging.error("Usage: %s /path/to/feeds.opml 2025-11-01", sys.argv[0])
        sys.exit(1)

    opml_path = sys.argv[1]
    since_arg = sys.argv[2]

    if not os.path.exists(opml_path):
        logging.error("OPML file not found: %s", opml_path)
        sys.exit(1)

    since_dt = parse_since_date(since_arg)
    logging.info("Since date: %s", since_dt.isoformat())

    prompt, openai_model, max_output_tokens, feeds_cfg = parse_opml(opml_path)
    payload = build_payload(prompt, feeds_cfg, since_dt)

    if not payload:
        # Nothing new / usable; do not print anything, just exit.
        sys.exit(0)

    openai_api_key = require_env("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = openai_api_key  # ensure client picks it up

    try:
        output = call_openai(openai_model, payload, max_output_tokens)
    except Exception as exc:
        logging.exception("OpenAI call failed: %s", exc)
        sys.exit(1)

    sys.stdout.write(output + "\n")
    sys.stdout.flush()
    logging.info("Done")


if __name__ == "__main__":
    main()