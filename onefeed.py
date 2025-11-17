#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import trafilatura
import yaml
from dateutil import parser as dateparser
from jsonschema import Draft202012Validator
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


def apply_schema_defaults(instance, schema):
    if isinstance(instance, dict) and isinstance(schema, dict):
        for key, subschema in schema.get("properties", {}).items():
            if key in instance:
                instance[key] = apply_schema_defaults(instance[key], subschema)
            elif "default" in subschema:
                instance[key] = subschema["default"]
    elif isinstance(instance, list) and isinstance(schema, dict):
        item_schema = schema.get("items", {})
        for idx, item in enumerate(instance):
            instance[idx] = apply_schema_defaults(item, item_schema)
    return instance


def load_schema() -> dict:
    schema_path = Path(__file__).with_name("config-schema.yaml")
    try:
        with schema_path.open("r", encoding="utf-8") as schema_file:
            return yaml.safe_load(schema_file)
    except FileNotFoundError:
        logging.error("Configuration schema file is missing: %s", schema_path)
    except yaml.YAMLError as exc:
        logging.error("Failed to parse configuration schema: %s", exc)
    sys.exit(1)


def load_example_config() -> str:
    example_path = Path(__file__).with_name("config-example.yaml")
    try:
        return example_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logging.error("Example configuration file is missing: %s", example_path)
    except OSError as exc:
        logging.error(
            "Failed to read example configuration from %s: %s", example_path, exc
        )
    sys.exit(1)


def parse_config(config_path: str):
    schema = load_schema()

    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except FileNotFoundError:
        logging.error("YAML config file not found: %s", config_path)
        sys.exit(1)
    except yaml.YAMLError as exc:
        logging.error("Failed to parse YAML config %s: %s", config_path, exc)
        sys.exit(1)

    if not isinstance(config, dict):
        logging.error("Configuration root must be a mapping")
        sys.exit(1)

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(config), key=lambda e: e.path)
    if errors:
        for error in errors:
            location = ".".join(str(segment) for segment in error.path) or "<root>"
            logging.error("Config validation error at %s: %s", location, error.message)
        sys.exit(1)

    config = apply_schema_defaults(config, schema)

    prompt = config["prompt"].strip()
    openai_model = config["openaiModel"].strip()
    max_output_tokens = int(config["maxOutputTokens"])

    feeds = [
        {
            "url": feed["url"],
            "title": feed.get("title", ""),
            "priority": int(feed["priority"]),
            "description": feed["description"],
            "representation": feed["representation"],
            "includeComments": bool(feed["includeComments"]),
            "maxPosts": (
                int(feed["maxPosts"]) if feed["maxPosts"] is not None else None
            ),
            "maxArticleChars": (
                int(feed["maxArticleChars"])
                if feed["maxArticleChars"] is not None
                else None
            ),
            "maxCommentsChars": (
                int(feed["maxCommentsChars"])
                if feed["maxCommentsChars"] is not None
                else None
            ),
        }
        for feed in config["feeds"]
    ]

    logging.info(
        "Loaded %d feeds from YAML; model=%s, max_output_tokens=%d",
        len(feeds),
        openai_model,
        max_output_tokens,
    )

    return prompt, openai_model, max_output_tokens, feeds


def entry_datetime(entry) -> datetime | None:
    t = getattr(entry, "published_parsed", None) or getattr(
        entry, "updated_parsed", None
    )
    if not t:
        return None
    try:
        dt = datetime(
            t.tm_year,
            t.tm_mon,
            t.tm_mday,
            t.tm_hour,
            t.tm_min,
            t.tm_sec,
            tzinfo=timezone.utc,
        )
    except Exception:
        return None
    return dt


def scrape_entry_content(
    url: str, include_comments: bool
) -> tuple[str | None, str | None]:
    if not url:
        return None, None

    downloaded = None
    try:
        downloaded = trafilatura.fetch_url(url)
    except Exception as exc:
        logging.debug("trafilatura fetch_url failed for %s: %s", url, exc)

    if not downloaded:
        return None, None

    text = None
    try:
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_formatting=False,
            include_links=False,
            favor_precision=True,
        )
    except Exception as exc:
        logging.debug("trafilatura extract failed for %s: %s", url, exc)

    comments_text = None
    if include_comments:
        try:
            doc = trafilatura.bare_extraction(
                downloaded,
                url=url,
                include_comments=True,
                include_formatting=False,
                include_links=False,
                favor_precision=True,
            )
            if doc:
                comments_field = getattr(doc, "comments", None)
                if comments_field:
                    comments_text = comments_field.strip() or None
        except Exception as exc:
            logging.debug("trafilatura bare_extraction failed for %s: %s", url, exc)

    if text:
        text = text.strip()
        if not text:
            text = None

    return text, comments_text


def fetch_feed_articles(feed_cfg: dict, since_dt: datetime):
    url = feed_cfg["url"]
    logging.info("Fetching feed %s (%s)", feed_cfg.get("title") or "", url)
    parsed = feedparser.parse(url)
    entries = parsed.entries or []
    articles: list[dict] = []

    for entry in entries:
        if feed_cfg["maxPosts"] is not None and len(articles) >= feed_cfg["maxPosts"]:
            break

        dt = entry_datetime(entry)
        if not dt or dt <= since_dt:
            continue

        link = getattr(entry, "link", "") or ""
        content, comments = scrape_entry_content(link, feed_cfg["includeComments"])
        if not content:
            continue

        if feed_cfg["maxArticleChars"] is not None:
            content = content[: feed_cfg["maxArticleChars"]]

        title = getattr(entry, "title", "") or ""
        published_iso = dt.isoformat()

        articles.append(
            {
                "title": title,
                "published": published_iso,
                "content": content,
            }
        )

        if feed_cfg["includeComments"] and comments:
            if feed_cfg["maxCommentsChars"] is not None:
                comments = comments[: feed_cfg["maxCommentsChars"]]
            articles[-1]["comments"] = comments

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
        logging.info(
            "No scraped articles after the given date; nothing to send to OpenAI"
        )
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
    logging.info(
        "Calling OpenAI model %s via Responses API (payload chars=%d)",
        model,
        len(payload_json),
    )

    instructions = (
        "You will receive a single JSON object.\n"
        "- The 'prompt' field tells you exactly what to do and how the user wants the "
        "output formatted.\n"
        "- The 'feeds' array contains the content you should base your response on.\n"
        "Follow the 'prompt' as closely as possible, using the provided feeds and "
        "articles as context."
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

    parser = argparse.ArgumentParser(description="Summarise feeds using OpenAI.")
    parser.add_argument("config", nargs="?", help="Path to YAML config")
    parser.add_argument("since", nargs="?", help="ISO date/time from which to scrape")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--write-example",
        metavar="PATH",
        help="Write an example YAML config to PATH and exit",
    )
    group.add_argument(
        "--validate",
        metavar="PATH",
        help="Validate YAML config against the schema and exit",
    )

    args = parser.parse_args()

    if args.write_example:
        example_path = Path(args.write_example)
        example_contents = load_example_config()
        example_path.write_text(example_contents, encoding="utf-8")
        logging.info("Wrote example config to %s", example_path)
        sys.exit(0)

    if args.validate:
        parse_config(args.validate)
        logging.info("Config at %s is valid", args.validate)
        sys.exit(0)

    if not (args.config and args.since):
        parser.error("Usage: onefeed.py /path/to/config.yaml 2025-11-01")

    if not os.path.exists(args.config):
        logging.error("YAML config file not found: %s", args.config)
        sys.exit(1)

    since_dt = parse_since_date(args.since)
    logging.info("Since date: %s", since_dt.isoformat())

    prompt, openai_model, max_output_tokens, feeds_cfg = parse_config(args.config)
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
