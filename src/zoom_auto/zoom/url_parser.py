"""Parse Zoom meeting URLs into meeting ID and password."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse


@dataclass
class ParsedMeeting:
    """Parsed meeting info from a URL or string."""

    meeting_id: str
    password: str = ""


def parse_meeting_input(input_str: str) -> ParsedMeeting:
    """Parse a meeting URL, meeting ID, or meeting ID with password.

    Supports:
    - Full URL: https://zoom.us/j/123456789?pwd=abc123
    - Short URL: https://us02web.zoom.us/j/123456789
    - Meeting ID only: 123456789
    - Meeting ID with spaces: 123 456 789
    - Meeting ID with dashes: 123-456-789

    Args:
        input_str: URL, meeting ID, or formatted meeting string.

    Returns:
        ParsedMeeting with meeting_id and optional password.

    Raises:
        ValueError: If the input cannot be parsed into a valid meeting ID.
    """
    input_str = input_str.strip()
    if not input_str:
        msg = "Empty meeting input"
        raise ValueError(msg)

    # Try parsing as a URL first
    if "zoom.us" in input_str or input_str.startswith(("http://", "https://")):
        return _parse_url(input_str)

    # Otherwise treat as a meeting ID (possibly with spaces/dashes)
    return _parse_meeting_id(input_str)


def _parse_url(url: str) -> ParsedMeeting:
    """Parse a Zoom URL into meeting info.

    Args:
        url: A Zoom meeting URL.

    Returns:
        ParsedMeeting with meeting_id and optional password.

    Raises:
        ValueError: If the URL does not contain a valid meeting ID.
    """
    # Ensure scheme is present for urlparse
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urlparse(url)
    query = parse_qs(parsed.query)

    # Extract password from query string (pwd parameter)
    password = ""
    if "pwd" in query:
        password = query["pwd"][0]

    # Extract meeting ID from path: /j/123456789 or /j/123456789/
    path = parsed.path.rstrip("/")
    match = re.search(r"/j/(\d+)", path)
    if match:
        return ParsedMeeting(meeting_id=match.group(1), password=password)

    # Personal meeting room: /my/username — no numeric ID
    if "/my/" in path:
        # Extract the personal room name
        room_match = re.search(r"/my/(.+)", path)
        if room_match:
            room_name = room_match.group(1).strip("/")
            msg = (
                f"Personal meeting room URL detected: '{room_name}'. "
                "Please use a numeric meeting ID or a /j/ URL instead."
            )
            raise ValueError(msg)

    msg = f"Could not extract meeting ID from URL: {url}"
    raise ValueError(msg)


def _parse_meeting_id(raw_id: str) -> ParsedMeeting:
    """Parse a raw meeting ID string (may contain spaces or dashes).

    Args:
        raw_id: A meeting ID, possibly formatted with spaces or dashes.

    Returns:
        ParsedMeeting with cleaned meeting_id.

    Raises:
        ValueError: If the cleaned ID is not numeric.
    """
    # Strip spaces, dashes, and other formatting
    cleaned = re.sub(r"[\s\-]", "", raw_id)

    if not cleaned:
        msg = "Empty meeting ID after cleaning"
        raise ValueError(msg)

    if not cleaned.isdigit():
        msg = f"Meeting ID must be numeric, got: '{raw_id}'"
        raise ValueError(msg)

    return ParsedMeeting(meeting_id=cleaned)
