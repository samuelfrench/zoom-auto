"""Persona builder — aggregates data from multiple sources to build a persona.

Combines analysis from transcripts, Slack exports, emails, and test
conversations to create a comprehensive communication profile.
Profiles are stored as TOML for easy human editing.
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from zoom_auto.config import PersonaConfig
from zoom_auto.persona.style_analyzer import StyleAnalyzer, StyleMetrics

logger = logging.getLogger(__name__)


@dataclass
class PersonaProfile:
    """A complete communication persona profile.

    All float traits are scored 0.0-1.0.

    Attributes:
        name: Name of the person this persona represents.
        formality: How formal the communication style is (0=casual).
        verbosity: How wordy responses tend to be (0=terse).
        technical_depth: Density of technical jargon (0=non-tech).
        assertiveness: How direct/assertive (0=tentative).
        filler_words: Filler words with frequency per 100 words.
        common_phrases: Frequently used phrases.
        greeting_style: Typical greeting (e.g., "Hey folks").
        agreement_style: Typical agreement (e.g., "Sounds good").
        avg_response_words: Average response length in words.
        preferred_terms: Terms the person prefers to use.
        avoided_terms: Terms the person avoids.
        standup_format: How they structure standup updates.
        vocabulary_richness: Type-token ratio.
        question_frequency: How often they ask questions (0-1).
        exclamation_rate: How often sentences end with '!'.
    """

    name: str = ""
    formality: float = 0.5
    verbosity: float = 0.5
    technical_depth: float = 0.3
    assertiveness: float = 0.5
    filler_words: dict[str, float] = field(default_factory=dict)
    common_phrases: list[str] = field(default_factory=list)
    greeting_style: str = "Hi everyone"
    agreement_style: str = "Sounds good"
    avg_response_words: int = 50
    preferred_terms: list[str] = field(default_factory=list)
    avoided_terms: list[str] = field(default_factory=list)
    standup_format: str = (
        "Yesterday I worked on X. Today I'm working on Y. No blockers."
    )
    vocabulary_richness: float = 0.5
    question_frequency: float = 0.1
    exclamation_rate: float = 0.0

    def to_toml(self, path: Path) -> None:
        """Write this profile to a TOML file.

        Args:
            path: File path to write the TOML output.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = [
            "# Persona Profile — auto-generated, feel free to edit",
            "",
            f'name = "{_escape_toml_str(self.name)}"',
            "",
            "# Trait scores (0.0 = low, 1.0 = high)",
            f"formality = {self.formality}",
            f"verbosity = {self.verbosity}",
            f"technical_depth = {self.technical_depth}",
            f"assertiveness = {self.assertiveness}",
            f"vocabulary_richness = {self.vocabulary_richness}",
            f"question_frequency = {self.question_frequency}",
            f"exclamation_rate = {self.exclamation_rate}",
            "",
            "# Response style",
            f"avg_response_words = {self.avg_response_words}",
            (
                f'greeting_style = '
                f'"{_escape_toml_str(self.greeting_style)}"'
            ),
            (
                f'agreement_style = '
                f'"{_escape_toml_str(self.agreement_style)}"'
            ),
            (
                f'standup_format = '
                f'"{_escape_toml_str(self.standup_format)}"'
            ),
            "",
        ]

        # Preferred / avoided terms
        lines.append(
            f"preferred_terms = {_list_to_toml(self.preferred_terms)}"
        )
        lines.append(
            f"avoided_terms = {_list_to_toml(self.avoided_terms)}"
        )
        lines.append(
            f"common_phrases = {_list_to_toml(self.common_phrases)}"
        )
        lines.append("")

        # Filler words as sub-table
        lines.append("[filler_words]")
        for word, rate in self.filler_words.items():
            key = word.replace(" ", "_")
            lines.append(f'"{key}" = {rate}')

        lines.append("")  # trailing newline
        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Saved persona profile to %s", path)

    @classmethod
    def from_toml(cls, path: Path) -> PersonaProfile:
        """Load a persona profile from a TOML file.

        Args:
            path: Path to the TOML file.

        Returns:
            A PersonaProfile populated from the file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        with open(path, "rb") as f:
            data = tomllib.load(f)

        # Reconstruct filler_words keys (underscores back to spaces)
        raw_fillers = data.get("filler_words", {})
        fillers: dict[str, float] = {}
        for key, val in raw_fillers.items():
            fillers[key.replace("_", " ")] = float(val)

        return cls(
            name=data.get("name", ""),
            formality=float(data.get("formality", 0.5)),
            verbosity=float(data.get("verbosity", 0.5)),
            technical_depth=float(data.get("technical_depth", 0.3)),
            assertiveness=float(data.get("assertiveness", 0.5)),
            filler_words=fillers,
            common_phrases=data.get("common_phrases", []),
            greeting_style=data.get("greeting_style", "Hi everyone"),
            agreement_style=data.get("agreement_style", "Sounds good"),
            avg_response_words=int(
                data.get("avg_response_words", 50)
            ),
            preferred_terms=data.get("preferred_terms", []),
            avoided_terms=data.get("avoided_terms", []),
            standup_format=data.get(
                "standup_format",
                "Yesterday I worked on X. Today I'm working on Y. "
                "No blockers.",
            ),
            vocabulary_richness=float(
                data.get("vocabulary_richness", 0.5)
            ),
            question_frequency=float(
                data.get("question_frequency", 0.1)
            ),
            exclamation_rate=float(
                data.get("exclamation_rate", 0.0)
            ),
        )


@dataclass
class TextSample:
    """A text sample with metadata for persona building.

    Attributes:
        text: The raw text content.
        source_type: Type of source (transcript, slack, email, etc).
        weight: Importance weight for merging (higher = more influence).
    """

    text: str
    source_type: str = "general"
    weight: float = 1.0


class PersonaBuilder:
    """Builds a communication persona from multiple data sources.

    Aggregates analysis from transcripts, Slack messages, emails,
    and test conversations to create a realistic communication profile.
    """

    # Default weights by source type (transcript > slack > email)
    DEFAULT_WEIGHTS: dict[str, float] = {
        "transcript": 2.0,
        "slack": 1.5,
        "email": 1.0,
        "writing": 1.0,
        "conversation": 1.2,
        "general": 1.0,
    }

    def __init__(self, config: PersonaConfig | None = None) -> None:
        self.config = config or PersonaConfig()
        self._analyzer = StyleAnalyzer()

    def build_from_samples(
        self,
        samples: list[TextSample],
        name: str = "",
    ) -> PersonaProfile:
        """Build a persona profile from text samples.

        Each sample is analyzed independently, then metrics are merged
        using weighted averaging.

        Args:
            samples: Text samples with source types and weights.
            name: Name for the persona profile.

        Returns:
            A complete PersonaProfile.
        """
        if not samples:
            logger.warning("No samples provided, returning default profile")
            return PersonaProfile(name=name)

        # Analyze each sample
        analyzed: list[tuple[StyleMetrics, dict[str, float], float]] = []
        for sample in samples:
            if not sample.text.strip():
                continue
            weight = sample.weight * self.DEFAULT_WEIGHTS.get(
                sample.source_type, 1.0,
            )
            metrics = self._analyzer.analyze([sample.text])
            fillers = self._analyzer.detect_filler_words([sample.text])
            analyzed.append((metrics, fillers, weight))

        if not analyzed:
            return PersonaProfile(name=name)

        # Merge metrics with weighted averaging
        profile = self._merge_metrics(analyzed, name)

        # Extract vocabulary from all samples
        all_texts = [s.text for s in samples if s.text.strip()]
        vocab = self._analyzer.extract_vocabulary(all_texts)
        profile.preferred_terms = vocab.top_words[:15]
        profile.common_phrases = vocab.top_bigrams[:10]

        return profile

    def build_from_texts(
        self,
        texts: list[str],
        name: str = "",
        source_type: str = "general",
    ) -> PersonaProfile:
        """Convenience: build a profile from plain text strings.

        Args:
            texts: Raw text strings to analyze.
            name: Name for the persona.
            source_type: Source type for all texts.

        Returns:
            A PersonaProfile.
        """
        samples = [
            TextSample(text=t, source_type=source_type)
            for t in texts
        ]
        return self.build_from_samples(samples, name=name)

    def generate_system_prompt(self, profile: PersonaProfile) -> str:
        """Generate an LLM system prompt from a persona profile.

        The prompt instructs the LLM to mimic the person's
        communication style.

        Args:
            profile: The persona profile to convert.

        Returns:
            A system prompt string for the LLM.
        """
        parts: list[str] = []

        # Identity
        if profile.name:
            parts.append(
                f"You are responding as {profile.name} in a meeting."
            )
        else:
            parts.append("You are responding in a meeting.")

        # Formality
        if profile.formality < 0.3:
            parts.append(
                "Use a casual, conversational tone. "
                "Contractions and informal language are fine."
            )
        elif profile.formality > 0.7:
            parts.append(
                "Use a professional, formal tone. "
                "Avoid slang and contractions."
            )
        else:
            parts.append(
                "Use a balanced, semi-formal tone."
            )

        # Verbosity
        if profile.verbosity < 0.3:
            parts.append(
                "Keep responses brief and to the point. "
                f"Aim for around {profile.avg_response_words} words."
            )
        elif profile.verbosity > 0.7:
            parts.append(
                "Give detailed, thorough responses. "
                f"Aim for around {profile.avg_response_words} words."
            )
        else:
            parts.append(
                f"Aim for around {profile.avg_response_words} words "
                "per response."
            )

        # Technical depth
        if profile.technical_depth > 0.6:
            parts.append(
                "Use technical terminology naturally."
            )
        elif profile.technical_depth < 0.2:
            parts.append(
                "Avoid heavy jargon; use plain language."
            )

        # Assertiveness
        if profile.assertiveness > 0.7:
            parts.append(
                "Be direct and decisive in responses."
            )
        elif profile.assertiveness < 0.3:
            parts.append(
                "Use hedging language like 'I think', 'maybe', "
                "'it seems like'."
            )

        # Filler words
        if profile.filler_words:
            top_fillers = list(profile.filler_words.keys())[:3]
            filler_str = ", ".join(f'"{f}"' for f in top_fillers)
            parts.append(
                f"Occasionally use filler words like {filler_str} "
                "to sound natural."
            )

        # Greeting / agreement style
        if profile.greeting_style:
            parts.append(
                f'When greeting, use variations of: '
                f'"{profile.greeting_style}".'
            )
        if profile.agreement_style:
            parts.append(
                f'When agreeing, use variations of: '
                f'"{profile.agreement_style}".'
            )

        # Preferred terms
        if profile.preferred_terms:
            terms = ", ".join(profile.preferred_terms[:8])
            parts.append(
                f"Incorporate these terms naturally: {terms}."
            )

        # Standup format
        if profile.standup_format:
            parts.append(
                f'For standup updates, follow this format: '
                f'"{profile.standup_format}"'
            )

        return " ".join(parts)

    def load(self, path: Path) -> PersonaProfile:
        """Load a previously saved persona profile.

        Args:
            path: Path to the saved TOML profile file.

        Returns:
            The loaded PersonaProfile.
        """
        return PersonaProfile.from_toml(path)

    def save(self, profile: PersonaProfile, path: Path) -> None:
        """Save a persona profile to disk as TOML.

        Args:
            profile: The profile to save.
            path: Path to save the TOML file.
        """
        profile.to_toml(path)

    def _merge_metrics(
        self,
        analyzed: list[tuple[StyleMetrics, dict[str, float], float]],
        name: str,
    ) -> PersonaProfile:
        """Merge multiple StyleMetrics into a PersonaProfile."""
        total_weight = sum(w for _, _, w in analyzed)
        if total_weight == 0:
            return PersonaProfile(name=name)

        # Weighted averages for numeric fields
        def _wavg(getter: Any) -> float:
            return sum(
                getter(m) * w for m, _, w in analyzed
            ) / total_weight

        avg_sent_len = _wavg(lambda m: m.avg_sentence_length)
        vocab_rich = _wavg(lambda m: m.vocabulary_richness)
        formality = _wavg(lambda m: m.formality_score)
        question_freq = _wavg(lambda m: m.question_frequency)
        filler_rate = _wavg(lambda m: m.filler_word_rate)
        tech_depth = _wavg(lambda m: m.technical_depth)
        excl_rate = _wavg(lambda m: m.exclamation_rate)

        # Weighted total words (for avg response length)
        total_samples = len(analyzed)
        total_words = sum(m.total_words for m, _, _ in analyzed)
        avg_resp_words = (
            total_words // total_samples if total_samples else 50
        )

        # Verbosity: derived from sentence length (longer = more verbose)
        # Normalize: 5 words/sentence -> 0.0, 25+ -> 1.0
        verbosity = min(1.0, max(0.0, (avg_sent_len - 5) / 20))

        # Assertiveness: inverse of question frequency + low filler rate
        assertiveness = max(
            0.0,
            min(1.0, 1.0 - question_freq - filler_rate / 100),
        )

        # Merge filler word dicts (weighted)
        merged_fillers: dict[str, float] = {}
        for _, fillers, weight in analyzed:
            for word, rate in fillers.items():
                if word in merged_fillers:
                    merged_fillers[word] += rate * weight
                else:
                    merged_fillers[word] = rate * weight
        for word in merged_fillers:
            merged_fillers[word] /= total_weight
            merged_fillers[word] = round(merged_fillers[word], 2)

        # Sort by rate descending, keep top 10
        sorted_fillers = dict(
            sorted(
                merged_fillers.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]
        )

        return PersonaProfile(
            name=name,
            formality=round(formality, 4),
            verbosity=round(verbosity, 4),
            technical_depth=round(tech_depth, 4),
            assertiveness=round(assertiveness, 4),
            filler_words=sorted_fillers,
            avg_response_words=max(10, avg_resp_words),
            vocabulary_richness=round(vocab_rich, 4),
            question_frequency=round(question_freq, 4),
            exclamation_rate=round(excl_rate, 4),
        )


def _escape_toml_str(s: str) -> str:
    """Escape a string for TOML double-quoted value."""
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )


def _list_to_toml(items: list[str]) -> str:
    """Convert a list of strings to TOML array syntax."""
    if not items:
        return "[]"
    quoted = ", ".join(f'"{_escape_toml_str(i)}"' for i in items)
    return f"[{quoted}]"
