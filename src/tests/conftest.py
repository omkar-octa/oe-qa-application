import os

# Unit tests never call the real API; provide a placeholder key so importing
# models.config does not fail when no .env is present in the test cwd.
os.environ.setdefault("CLAUDE_API_KEY", "test-key")
