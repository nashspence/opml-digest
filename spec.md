# opml-digest

## Scenario: Summarize YAML-configured feeds with OpenAI
Given a YAML config file that matches config-schema.yaml and includes prompt, openaiModel, maxOutputTokens, and feeds entries
And each feed entry includes url, priority, description, and representation fields
And the OPENAI_API_KEY environment variable is set
When I run `python opml-digest.py /path/to/config.yaml 2025-11-01`
Then the script scrapes articles published after the given date and sends the payload to OpenAI
And it prints the model's response to standard output
And when I run `python opml-digest.py --validate /path/to/config.yaml`
Then the config is validated against config-schema.yaml without contacting feeds or OpenAI
And when I run `python opml-digest.py --write-example ./my-config.yaml`
Then an example YAML config is written to that path and the script exits
