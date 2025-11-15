# greader-digest

## Scenario: generate digest for unread items
* Given FreshRSS exposes the Google Reader API at "<api_url>"
* And the user "<username>" authenticates with API password "<password>"
* And unread items exist across multiple feeds
* And SUMMARY_PROMPT is set to "<prompt>"
* And FETCH_FULL_CONTENT is "true"
* And ARTICLE_MAX_CHARS is "<limit>"
* When I run greader-digest with FRESHRSS_API_URL="<api_url>"
* And FRESHRSS_USERNAME="<username>"
* And FRESHRSS_API_PASSWORD="<password>"
* And OPENAI_MODEL="<model>"
* And MAX_ITEMS="<max_items>"
* And MAX_OUTPUT_TOKENS="<max_tokens>"
* Then greader-digest authenticates via /accounts/ClientLogin
* And it fetches subscriptions with /reader/api/0/subscription/list
* And it fetches unread entries with /reader/api/0/stream/contents/reading-list
* And it fetches metadata for each feed referenced by unread items
* And it builds a summarization payload combining feed context and article content
* And it calls the OpenAI Responses API with "<model>"
* And it writes a single <article> HTML fragment to stdout according to OUTPUT_FORMAT

## Scenario: emit placeholder when no unread items exist
* Given FreshRSS reports zero unread entries
* When I run greader-digest
* Then greader-digest writes an <article> stating there are no new items
