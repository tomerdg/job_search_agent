import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0 Safari/537.36 job-search-agent",
    "Accept-Language": "he-IL,he;q=0.9,en;q=0.8",
}

session = requests.Session()
session.headers.update(HEADERS)
TIMEOUT = 30
