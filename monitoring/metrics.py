from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API Requests"
)

REQUEST_LATENCY = Histogram(
    "api_latency_seconds",
    "API latency"
)