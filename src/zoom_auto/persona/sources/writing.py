"""Email and document analysis for persona building.

Extracts communication patterns from written documents
(emails, reports, documentation) for persona building.
"""

from __future__ import annotations

import email
import logging
import re
from email import policy
from pathlib import Path

logger = logging.getLogger(__name__)

# Supported text-based document extensions
_TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".text"}
_EMAIL_EXTENSIONS = {".eml"}


class WritingAnalyzer:
    """Analyzes written documents for persona building.

    Extracts writing style, vocabulary, and communication patterns
    from emails, documents, and other written content.
    """

    def analyze_emails(self, email_dir: Path) -> list[str]:
        """Analyze email files for communication patterns.

        Supports .eml files (standard email format) and .txt files
        containing email body text.

        Args:
            email_dir: Directory containing email files.

        Returns:
            List of extracted email body texts.
        """
        if not email_dir.is_dir():
            logger.warning("Email directory not found: %s", email_dir)
            return []

        bodies: list[str] = []

        for path in sorted(email_dir.iterdir()):
            if not path.is_file():
                continue

            if path.suffix.lower() in _EMAIL_EXTENSIONS:
                body = self._parse_eml(path)
                if body:
                    bodies.append(body)
            elif path.suffix.lower() in _TEXT_EXTENSIONS:
                text = self._read_text(path)
                if text:
                    bodies.append(self._clean_email_body(text))

        logger.info(
            "Extracted %d email bodies from %s",
            len(bodies), email_dir,
        )
        return bodies

    def analyze_documents(self, doc_dir: Path) -> list[str]:
        """Analyze document files for writing patterns.

        Supports .txt, .md, .rst plain text files.

        Args:
            doc_dir: Directory containing documents.

        Returns:
            List of extracted document texts.
        """
        if not doc_dir.is_dir():
            logger.warning("Document directory not found: %s", doc_dir)
            return []

        texts: list[str] = []

        for path in sorted(doc_dir.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() in _TEXT_EXTENSIONS:
                text = self._read_text(path)
                if text:
                    texts.append(text)

        logger.info(
            "Extracted %d documents from %s",
            len(texts), doc_dir,
        )
        return texts

    def _parse_eml(self, path: Path) -> str:
        """Parse an .eml file and extract the body text."""
        try:
            raw = path.read_bytes()
            msg = email.message_from_bytes(
                raw, policy=policy.default,
            )

            # Try to get plain text body
            body = msg.get_body(preferencelist=("plain",))
            if body:
                content = body.get_content()
                if isinstance(content, str):
                    return self._clean_email_body(content)

            # Fall back to HTML body, strip tags
            body = msg.get_body(preferencelist=("html",))
            if body:
                content = body.get_content()
                if isinstance(content, str):
                    return self._strip_html(content)

        except Exception as exc:
            logger.warning("Failed to parse email %s: %s", path, exc)

        return ""

    def _read_text(self, path: Path) -> str:
        """Read a plain text file."""
        try:
            text = path.read_text(
                encoding="utf-8", errors="replace",
            ).strip()
            return text if text else ""
        except OSError as exc:
            logger.warning("Failed to read %s: %s", path, exc)
            return ""

    def _clean_email_body(self, text: str) -> str:
        """Clean email body: remove signatures, quoted replies."""
        lines: list[str] = []
        for line in text.split("\n"):
            # Stop at common signature/reply markers
            stripped = line.strip()
            if stripped.startswith("--"):
                break
            if stripped.startswith(">"):
                continue  # Skip quoted replies
            if re.match(
                r"^On .+ wrote:$", stripped, re.IGNORECASE,
            ):
                break
            lines.append(line)

        return "\n".join(lines).strip()

    def _strip_html(self, html: str) -> str:
        """Strip HTML tags and return plain text."""
        # Remove script/style blocks
        text = re.sub(
            r"<(script|style)[^>]*>.*?</\1>",
            "", html, flags=re.DOTALL | re.IGNORECASE,
        )
        # Remove tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text
