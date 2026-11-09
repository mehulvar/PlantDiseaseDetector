import os

# Integration tests send many requests from one client; lift the production limit.
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "100000")
