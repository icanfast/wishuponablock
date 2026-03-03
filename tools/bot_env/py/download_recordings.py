#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "https://dev.wishuponablock.com"
DEFAULT_OUT_DIR = "tools/bot_env/recordings"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

try:
    import certifi  # type: ignore
except Exception:
    certifi = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download recordings from admin API export manifest into local files."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--cookie", default=None, help="Raw wub_session cookie value.")
    parser.add_argument("--email", default=None, help="Email login (optional).")
    parser.add_argument("--password", default=None, help="Email password (optional).")
    parser.add_argument("--mode", default=None)
    parser.add_argument("--build", default=None)
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--arch", default=None)
    parser.add_argument("--reward-profile", default=None)
    parser.add_argument("--queue-policy", default=None)
    parser.add_argument("--pipeline-id", default=None)
    parser.add_argument("--actor-type", default=None, choices=["human", "bot"])
    parser.add_argument("--piece-source-profile", default=None)
    parser.add_argument("--started-from-ms", type=int, default=None)
    parser.add_argument("--started-to-ms", type=int, default=None)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-recordings", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing local files if present.",
    )
    parser.add_argument(
        "--sleep-ms",
        type=int,
        default=0,
        help="Optional delay between object downloads.",
    )
    parser.add_argument(
        "--save-manifest",
        action="store_true",
        help="Write final manifest summary JSON in output directory.",
    )
    parser.add_argument(
        "--ca-file",
        default=None,
        help="Path to custom CA bundle PEM file for TLS verification.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification (debug use only).",
    )
    args = parser.parse_args()
    if not args.cookie and not (args.email and args.password):
        parser.error("Provide either --cookie or both --email and --password.")
    return args


def trim_base_url(base_url: str) -> str:
    value = base_url.strip().rstrip("/")
    if not value:
        raise ValueError("base-url is empty.")
    if not value.startswith("http://") and not value.startswith("https://"):
        raise ValueError("base-url must include scheme, e.g. https://dev.wishuponablock.com")
    return value


def request_json(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    ssl_context: ssl.SSLContext | None = None,
) -> tuple[int, dict[str, Any], dict[str, str]]:
    req = urllib.request.Request(
        url=url,
        method=method,
        headers=headers or {},
        data=body,
    )
    try:
        with urllib.request.urlopen(req, context=ssl_context) as resp:
            status = int(resp.status)
            raw_headers = {k.lower(): v for k, v in resp.headers.items()}
            payload = json.loads(resp.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise RuntimeError(f"Expected JSON object payload from {url}")
            return status, payload, raw_headers
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"error": raw}
        if not isinstance(payload, dict):
            payload = {"error": str(payload)}
        headers_out = {k.lower(): v for k, v in (exc.headers.items() if exc.headers else [])}
        return int(exc.code), payload, headers_out


def request_bytes(
    url: str,
    headers: dict[str, str] | None = None,
    ssl_context: ssl.SSLContext | None = None,
) -> tuple[int, bytes, dict[str, str]]:
    req = urllib.request.Request(url=url, method="GET", headers=headers or {})
    try:
        with urllib.request.urlopen(req, context=ssl_context) as resp:
            return int(resp.status), resp.read(), {k.lower(): v for k, v in resp.headers.items()}
    except urllib.error.HTTPError as exc:
        body = exc.read()
        headers_out = {k.lower(): v for k, v in (exc.headers.items() if exc.headers else [])}
        return int(exc.code), body, headers_out


def parse_set_cookie(headers: dict[str, str]) -> str | None:
    set_cookie = headers.get("set-cookie")
    if not set_cookie:
        return None
    parts = [part.strip() for part in set_cookie.split(";")]
    for part in parts:
        if part.startswith("wub_session="):
            value = part.split("=", 1)[1].strip()
            return value or None
    return None


def login_and_get_cookie(base_url: str, email: str, password: str) -> str:
    endpoint = f"{base_url}/api/auth/email/login"
    payload = json.dumps({"email": email, "password": password}).encode("utf-8")
    status, body, headers = request_json(
        url=endpoint,
        method="POST",
        headers={
            "content-type": "application/json",
            "accept": "application/json",
            "origin": base_url,
            "referer": f"{base_url}/",
            "user-agent": HTTP_USER_AGENT,
            "accept-language": "en-US,en;q=0.9",
        },
        body=payload,
        ssl_context=TLS_CONTEXT,
    )
    if status != 200:
        raise RuntimeError(f"Login failed ({status}): {body.get('error', 'unknown error')}")
    cookie = parse_set_cookie(headers)
    if not cookie:
        raise RuntimeError("Login succeeded but wub_session cookie was not returned.")
    return cookie


def resolve_tls_context(args: argparse.Namespace) -> ssl.SSLContext:
    if bool(args.insecure):
        return ssl._create_unverified_context()
    if isinstance(args.ca_file, str) and args.ca_file.strip():
        return ssl.create_default_context(cafile=args.ca_file.strip())
    # Prefer certifi bundle when available; fallback to system defaults.
    if certifi is not None:
        try:
            return ssl.create_default_context(cafile=certifi.where())
        except Exception:
            pass
    return ssl.create_default_context()


TLS_CONTEXT = ssl.create_default_context()
HTTP_USER_AGENT = DEFAULT_USER_AGENT


def build_manifest_params(args: argparse.Namespace, cursor: str | None) -> dict[str, str]:
    out: dict[str, str] = {
        "limit": str(max(1, min(500, int(args.limit)))),
    }
    if args.mode:
        out["mode"] = str(args.mode).strip().lower()
    if args.build:
        out["build"] = str(args.build).strip()
    if args.user_id:
        out["user_id"] = str(args.user_id).strip()
    if args.arch:
        out["arch"] = str(args.arch).strip()
    if args.reward_profile:
        out["reward_profile"] = str(args.reward_profile).strip()
    if args.queue_policy:
        out["queue_policy"] = str(args.queue_policy).strip()
    if args.pipeline_id:
        out["pipeline_id"] = str(args.pipeline_id).strip()
    if args.actor_type:
        out["actor_type"] = str(args.actor_type).strip()
    if args.piece_source_profile:
        out["piece_source_profile"] = str(args.piece_source_profile).strip()
    if args.started_from_ms is not None:
        out["started_from_ms"] = str(int(args.started_from_ms))
    if args.started_to_ms is not None:
        out["started_to_ms"] = str(int(args.started_to_ms))
    if args.min_samples is not None:
        out["min_samples"] = str(max(1, int(args.min_samples)))
    if cursor:
        out["cursor"] = cursor
    return out


def safe_name(value: str) -> str:
    out = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def main() -> None:
    args = parse_args()
    global TLS_CONTEXT
    global HTTP_USER_AGENT
    TLS_CONTEXT = resolve_tls_context(args)
    HTTP_USER_AGENT = str(args.user_agent).strip() or DEFAULT_USER_AGENT
    base_url = trim_base_url(args.base_url)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cookie_value = args.cookie.strip() if isinstance(args.cookie, str) and args.cookie.strip() else None
    if not cookie_value:
        cookie_value = login_and_get_cookie(base_url, str(args.email), str(args.password))

    headers = {
        "accept": "application/json",
        "cookie": f"wub_session={cookie_value}",
        "origin": base_url,
        "referer": f"{base_url}/",
        "user-agent": HTTP_USER_AGENT,
        "accept-language": "en-US,en;q=0.9",
    }

    manifest_endpoint = f"{base_url}/api/admin/recordings/export-manifest"
    object_endpoint = f"{base_url}/api/admin/recordings/object"

    downloaded = 0
    skipped_existing = 0
    failed = 0
    page_count = 0
    cursor: str | None = None
    all_manifest_rows: list[dict[str, Any]] = []

    max_recordings = int(args.max_recordings)
    max_recordings = max_recordings if max_recordings > 0 else 0

    while True:
        params = build_manifest_params(args, cursor)
        url = f"{manifest_endpoint}?{urllib.parse.urlencode(params)}"
        status, payload, _ = request_json(url=url, headers=headers, ssl_context=TLS_CONTEXT)
        if status != 200:
            raise RuntimeError(f"Manifest request failed ({status}): {payload.get('error', 'unknown error')}")

        recordings = payload.get("recordings")
        page = payload.get("page")
        if not isinstance(recordings, list):
            raise RuntimeError("Manifest payload missing recordings array.")
        if not isinstance(page, dict):
            raise RuntimeError("Manifest payload missing page object.")

        page_count += 1
        for rec in recordings:
            if not isinstance(rec, dict):
                continue
            all_manifest_rows.append(rec)
            recording_id = rec.get("id")
            if not isinstance(recording_id, str) or not recording_id:
                continue

            local_path = out_dir / f"{safe_name(recording_id)}.json"
            if local_path.exists() and not args.overwrite:
                skipped_existing += 1
                continue

            fetch_url = f"{object_endpoint}?{urllib.parse.urlencode({'id': recording_id})}"
            status_obj, body_obj, _ = request_bytes(
                fetch_url, headers=headers, ssl_context=TLS_CONTEXT
            )
            if status_obj != 200:
                failed += 1
                continue

            local_path.write_bytes(body_obj)
            downloaded += 1
            if args.sleep_ms and args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)

            if max_recordings > 0 and downloaded >= max_recordings:
                cursor = None
                break

        if max_recordings > 0 and downloaded >= max_recordings:
            break

        next_cursor = page.get("nextCursor")
        if not isinstance(next_cursor, str) or not next_cursor:
            break
        cursor = next_cursor

    summary = {
        "baseUrl": base_url,
        "outputDir": str(out_dir),
        "downloaded": downloaded,
        "skippedExisting": skipped_existing,
        "failed": failed,
        "pages": page_count,
        "recordsSeen": len(all_manifest_rows),
    }
    print(f"[download-recordings] {json.dumps(summary, ensure_ascii=True)}")

    if args.save_manifest:
        timestamp = int(time.time())
        manifest_path = out_dir / f"manifest_{timestamp}.json"
        manifest_payload = {
            "schema": "wishuponablock.recording_manifest_dump.v1",
            "createdAtMs": int(time.time() * 1000),
            "selector": build_manifest_params(args, None),
            "summary": summary,
            "recordings": all_manifest_rows,
        }
        manifest_path.write_text(
            json.dumps(manifest_payload, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        print(f"[download-recordings] wrote manifest {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[download-recordings] interrupted", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"[download-recordings] failed: {exc}", file=sys.stderr)
        sys.exit(1)
