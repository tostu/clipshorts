"""Docker healthcheck: /healthz answers 200 (it is exempt from AUTH_TOKEN)."""

import sys
import urllib.request

try:
    urllib.request.urlopen("http://127.0.0.1:8000/healthz", timeout=4)
except Exception:
    sys.exit(1)
