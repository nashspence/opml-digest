# opml-digest

## Scenario: Summarize OPML feeds with OpenAI
Given an OPML file whose head includes <prompt>, <openaiModel>, and <maxOutputTokens>
And each feed outline includes xmlUrl, priority, description, and representation attributes
And the OPENAI_API_KEY environment variable is set
When I run `python opml-digest.py /path/to/feeds.opml 2025-11-01`
Then the script scrapes articles published after the given date and sends the payload to OpenAI
And it prints the model's response to standard output
