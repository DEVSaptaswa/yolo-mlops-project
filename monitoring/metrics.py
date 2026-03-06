from prometheus_client import Counter, Histogram, Gauge

# total requests
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests"
)

# latency
REQUEST_LATENCY = Histogram(
    "api_latency_seconds",
    "API request latency"
)

# CPU usage
CPU_USAGE = Gauge(
    "cpu_usage_percent",
    "CPU usage percentage"
)