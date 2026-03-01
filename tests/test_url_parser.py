"""Tests for Zoom meeting URL parser."""

from __future__ import annotations

import pytest

from zoom_auto.zoom.url_parser import ParsedMeeting, parse_meeting_input  # noqa: E402


class TestParseFullURL:
    """Test parsing full Zoom URLs."""

    def test_full_url_with_password(self) -> None:
        result = parse_meeting_input(
            "https://zoom.us/j/123456789?pwd=abc123"
        )
        assert result == ParsedMeeting(meeting_id="123456789", password="abc123")

    def test_full_url_without_password(self) -> None:
        result = parse_meeting_input("https://zoom.us/j/123456789")
        assert result == ParsedMeeting(meeting_id="123456789", password="")

    def test_url_with_trailing_slash(self) -> None:
        result = parse_meeting_input("https://zoom.us/j/123456789/")
        assert result == ParsedMeeting(meeting_id="123456789", password="")

    def test_url_with_multiple_query_params(self) -> None:
        result = parse_meeting_input(
            "https://zoom.us/j/999888777?pwd=secret&uname=test"
        )
        assert result.meeting_id == "999888777"
        assert result.password == "secret"


class TestParseSubdomainURL:
    """Test parsing URLs with custom subdomains."""

    def test_us02web_subdomain(self) -> None:
        result = parse_meeting_input(
            "https://us02web.zoom.us/j/987654321?pwd=xyz789"
        )
        assert result == ParsedMeeting(meeting_id="987654321", password="xyz789")

    def test_us04web_subdomain(self) -> None:
        result = parse_meeting_input(
            "https://us04web.zoom.us/j/111222333"
        )
        assert result == ParsedMeeting(meeting_id="111222333", password="")

    def test_custom_vanity_subdomain(self) -> None:
        result = parse_meeting_input(
            "https://mycompany.zoom.us/j/444555666?pwd=pass123"
        )
        assert result == ParsedMeeting(meeting_id="444555666", password="pass123")


class TestParseMeetingID:
    """Test parsing plain meeting IDs."""

    def test_plain_numeric_id(self) -> None:
        result = parse_meeting_input("123456789")
        assert result == ParsedMeeting(meeting_id="123456789", password="")

    def test_id_with_spaces(self) -> None:
        result = parse_meeting_input("123 456 789")
        assert result == ParsedMeeting(meeting_id="123456789", password="")

    def test_id_with_dashes(self) -> None:
        result = parse_meeting_input("123-456-789")
        assert result == ParsedMeeting(meeting_id="123456789", password="")

    def test_id_with_leading_trailing_whitespace(self) -> None:
        result = parse_meeting_input("  123456789  ")
        assert result == ParsedMeeting(meeting_id="123456789", password="")

    def test_long_meeting_id(self) -> None:
        result = parse_meeting_input("12345678901")
        assert result == ParsedMeeting(meeting_id="12345678901", password="")

    def test_id_with_mixed_formatting(self) -> None:
        result = parse_meeting_input("123 456-789")
        assert result == ParsedMeeting(meeting_id="123456789", password="")


class TestParseEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Empty meeting input"):
            parse_meeting_input("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="Empty meeting input"):
            parse_meeting_input("   ")

    def test_non_numeric_id_raises(self) -> None:
        with pytest.raises(ValueError, match="numeric"):
            parse_meeting_input("abc123")

    def test_letters_only_raises(self) -> None:
        with pytest.raises(ValueError, match="numeric"):
            parse_meeting_input("notanumber")

    def test_personal_room_url_raises(self) -> None:
        with pytest.raises(ValueError, match="Personal meeting room"):
            parse_meeting_input("https://zoom.us/my/johndoe")

    def test_url_without_meeting_id_raises(self) -> None:
        with pytest.raises(ValueError, match="Could not extract meeting ID"):
            parse_meeting_input("https://zoom.us/signin")

    def test_url_without_scheme(self) -> None:
        result = parse_meeting_input("zoom.us/j/123456789?pwd=test")
        assert result == ParsedMeeting(meeting_id="123456789", password="test")

    def test_http_url(self) -> None:
        result = parse_meeting_input("http://zoom.us/j/123456789")
        assert result == ParsedMeeting(meeting_id="123456789", password="")


class TestParsedMeetingDataclass:
    """Test the ParsedMeeting dataclass."""

    def test_default_password(self) -> None:
        meeting = ParsedMeeting(meeting_id="123")
        assert meeting.password == ""

    def test_equality(self) -> None:
        a = ParsedMeeting(meeting_id="123", password="abc")
        b = ParsedMeeting(meeting_id="123", password="abc")
        assert a == b

    def test_inequality(self) -> None:
        a = ParsedMeeting(meeting_id="123", password="abc")
        b = ParsedMeeting(meeting_id="456", password="abc")
        assert a != b
