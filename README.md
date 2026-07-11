# Zatch Hybrid Recommendation Engine

The single, production Flask service for Zatch recommendations — reels and
products, in one deployable process. Merges what were previously two
standalone services (`Reel Based Recommendation Engine`, `Product
Recommendation Engine`) into one, so your app has exactly one URL to call
regardless of what kind of recommendation the user needs.

Both engines are **live, real-time, and MongoDB-backed** — there is no
offline-trained dataset anymore. A recommendation reflects a user's most
recent order, cart-add, like, save, view, or bit/live-session interaction
within seconds (bounded only by short in-process TTL caches on catalog/bit/
session scans; per-user signals are never cached).

## How intent works

The client tells the engine what it wants via `intent` — there's no
automatic inference. Your app already knows whether the user is on the reels
feed or browsing products, so it just passes the right value:

```http
GET /recommend?intent=reel&user_id=<id>&video_id=<current_reel_id>&top_n=<int>
GET /recommend?intent=product&user_id=<id>&limit=<int>&category=<name>&include_seen=<bool>
```

`user_id` is a MongoDB user `_id`, username, or email. `video_id` (reel
intent) is the bit/live-session id currently being viewed — optional.
Omitting `intent` defaults to reel behavior, for backward compatibility with
anything calling this exactly as the old standalone reel service did.

## Architecture

```
User request
     |
     v
  app.py  --intent=reel-->  reel_engine    (live MongoDB: bits + live sessions,
     |                                      signal-scored against the viewer's
     |                                      real orders/carts/likes/saves)
     |
     `----intent=product--> product_engine (live MongoDB signal-hybrid + trained-
                                             artifact scoring, always live metadata)
```

Both engines fail open: a Mongo hiccup degrades the response (503, with a
circuit breaker to avoid piling up blocked requests during an outage) rather
than crashing the process. `/health` reflects both engines' live status.

## Endpoints

```http
GET /                                              # demo UI
GET /health                                        # combined status, see above
GET /reel-health                                    # reel engine Mongo status/collection counts
GET /recommend?intent=reel|product&...              # the one endpoint most clients need
GET /trending?top_n=<int>                           # popular reels/bits (no login required)
GET /user/<user_id>                                 # reel cold-start check
GET /video/<video_id>                               # bit/live-session existence check
GET /zatch/reel-recommendations/<user_id>?current_reel_id=&limit=&include_types=
GET /zatch/health                                    # Mongo collection counts
GET /product-health                                  # product engine detail
GET /product-recommendations/<user_id>?limit=&category=&include_seen=
GET /similar-products/<product_id>?limit=
GET /product-interactions/<user_id>
POST /admin/reload-artifacts                         # requires X-Admin-Key header, fails closed if unset
```

## Authentication

Every route above except `/`, `/health`, and `POST /admin/reload-artifacts`
requires an `X-API-Key` header matching the `API_KEY` env var — **if
`API_KEY` isn't set, these routes run unauthenticated** (a boot-time log
warning makes this visible). Set `API_KEY` before exposing this service to
anything other than local development. `/admin/reload-artifacts` has its own
separate, stricter, fail-closed `ADMIN_API_KEY`/`X-Admin-Key` check.

## Local setup

```bash
pip install -r requirements-dev.txt
python scripts/run_server.py
```

Set env vars via `.env` based on `.env.example`:

```text
MONGO_URI=mongodb+srv://USER:PASSWORD@CLUSTER.mongodb.net/zatch?retryWrites=true&w=majority
MONGO_DB_NAME=zatch
MONGO_TIMEOUT_MS=5000
MONGO_MAX_POOL_SIZE=20
ADMIN_API_KEY=change-me
API_KEY=change-me
SENTRY_DSN=
PRODUCT_CATALOG_CACHE_TTL_SECONDS=45
ZATCH_CACHE_TTL_SECONDS=30
ALLOWED_ORIGINS=*
```

`SENTRY_DSN` is optional — leave it blank to run without error tracking, or
set it (a free https://sentry.io project works) to get exceptions reported.

## Production

```bash
gunicorn app:app --workers 1 --worker-class gthread --threads 8 --timeout 180 --bind 0.0.0.0:$PORT
```

Single worker, deliberately: `POST /admin/reload-artifacts` reloads an
in-memory artifact and the rate limiter/circuit-breaker/caches are all
in-process — with more than one worker process, each would keep its own
independent copy, so a reload or a tripped breaker in one wouldn't be seen by
the others. This is an I/O-bound workload (Mongo calls), so threads already
provide real concurrency without needing multiple processes. If real traffic
outgrows this, the path forward is externalizing those caches/limiter/breaker
to Redis and moving to multiple workers/instances — not needed yet.

Neither engine loads large files into memory anymore, so this fits
comfortably on Render's smallest (`starter`) plan; bump the plan if real
traffic needs more headroom.

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

Entirely `mongomock`-based — no real MongoDB or large model files needed.

## Training the product artifact

```bash
python scripts/train_model.py        # rebuilds product_engine/models/*.joblib
                                      # atomic write + rolling backups, run offline/manually
```

The trained artifact only ever supplies *scoring* (TF-IDF content similarity,
item-to-item neighbors, popularity) — every product's displayed name, price,
stock, and images are always resolved live against MongoDB at request time,
so a sold-out, re-priced, or renamed product is never served stale. Products
that are no longer active/in-stock are dropped from trained-artifact results
entirely rather than shown as if still available.

After training, reload the artifact into the running service:

```bash
curl -X POST https://<host>/admin/reload-artifacts -H "X-Admin-Key: <your key>"
```
