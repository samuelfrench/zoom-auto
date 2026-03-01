"""NLP analysis of communication patterns and style.

Analyzes text samples to extract vocabulary, sentence structure,
formality level, and other communication style indicators.
Uses basic Python text analysis — no external NLP libraries required.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Filler words commonly found in speech transcripts
FILLER_WORDS = {
    "um", "uh", "er", "ah", "like", "you know", "i mean",
    "sort of", "kind of", "basically", "actually", "right",
    "so", "well", "anyway", "literally", "honestly",
}

# Formal language markers
FORMAL_MARKERS = {
    "therefore", "however", "furthermore", "moreover", "consequently",
    "nevertheless", "regarding", "concerning", "accordingly", "whereas",
    "hereby", "henceforth", "pursuant", "notwithstanding", "additionally",
    "subsequently", "preceding", "respective", "aforementioned",
    "shall", "ought", "thus", "hence",
}

# Informal language markers
INFORMAL_MARKERS = {
    "gonna", "wanna", "gotta", "kinda", "sorta", "yeah", "yep",
    "nope", "cool", "awesome", "totally", "stuff", "things",
    "ok", "okay", "hey", "hi", "sup", "yo", "lol", "haha",
    "btw", "fyi", "imo", "tbh", "nah", "dunno", "yup",
}

# Common technical/jargon terms (cross-domain)
TECHNICAL_MARKERS = {
    "api", "deploy", "pipeline", "architecture", "infrastructure",
    "implementation", "algorithm", "repository", "database",
    "framework", "integration", "microservice", "scalability",
    "throughput", "latency", "bandwidth", "optimization",
    "regression", "refactor", "sprint", "backlog", "standup",
    "blocker", "dependency", "endpoint", "schema", "migration",
    "stakeholder", "deliverable", "milestone", "kpi", "roi",
    "workflow", "methodology", "paradigm", "abstraction",
    "aggregate", "baseline", "benchmark", "capacity",
}


@dataclass
class StyleMetrics:
    """Quantified communication style metrics.

    Attributes:
        avg_sentence_length: Average words per sentence.
        vocabulary_richness: Type-token ratio (unique/total words).
        formality_score: Formality level (0=casual, 1=formal).
        question_frequency: Fraction of sentences that are questions.
        filler_word_rate: Filler words per 100 words.
        passive_voice_rate: Fraction of sentences using passive voice.
        technical_depth: Density of technical/jargon terms (0-1).
        avg_word_length: Average word length in characters.
        exclamation_rate: Fraction of sentences ending with '!'.
        total_words: Total word count across all samples.
    """

    avg_sentence_length: float = 0.0
    vocabulary_richness: float = 0.0
    formality_score: float = 0.5
    question_frequency: float = 0.0
    filler_word_rate: float = 0.0
    passive_voice_rate: float = 0.0
    technical_depth: float = 0.0
    avg_word_length: float = 0.0
    exclamation_rate: float = 0.0
    total_words: int = 0


@dataclass
class VocabularyProfile:
    """Vocabulary analysis results.

    Attributes:
        top_words: Most frequently used content words.
        top_bigrams: Most common two-word phrases.
        unique_ratio: Ratio of unique to total words.
        avg_word_length: Average word length.
    """

    top_words: list[str] = field(default_factory=list)
    top_bigrams: list[str] = field(default_factory=list)
    unique_ratio: float = 0.0
    avg_word_length: float = 0.0


# Common English stop words to filter from vocabulary analysis
_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "and",
    "but", "or", "nor", "not", "no", "so", "if", "then", "than",
    "too", "very", "just", "about", "up", "out", "that", "this",
    "it", "its", "i", "me", "my", "we", "our", "you", "your",
    "he", "she", "they", "them", "his", "her", "their", "what",
    "which", "who", "when", "where", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some",
    "such", "only", "own", "same", "also", "there", "here",
}

# Passive voice auxiliary + past participle pattern
_PASSIVE_RE = re.compile(
    r"\b(is|are|was|were|be|been|being|get|gets|got|gotten)\s+"
    r"\w+ed\b",
    re.IGNORECASE,
)


def _tokenize(text: str) -> list[str]:
    """Simple word tokenizer: lowercase, split on non-alpha."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on .!? boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


class StyleAnalyzer:
    """Analyzes communication style from text samples.

    Uses basic Python text analysis to extract quantified style
    metrics from text data (transcripts, messages, emails).
    """

    def analyze(self, texts: list[str]) -> StyleMetrics:
        """Analyze communication style from a collection of texts.

        Args:
            texts: List of text samples to analyze.

        Returns:
            StyleMetrics with quantified style indicators.
        """
        if not texts:
            return StyleMetrics()

        combined = " ".join(texts)
        words = _tokenize(combined)
        sentences = _split_sentences(combined)

        if not words:
            return StyleMetrics()

        total_words = len(words)
        unique_words = len(set(words))

        # Average sentence length
        if sentences:
            sent_lengths = [len(_tokenize(s)) for s in sentences]
            avg_sent_len = (
                sum(sent_lengths) / len(sent_lengths)
                if sent_lengths else 0.0
            )
        else:
            avg_sent_len = float(total_words)

        # Vocabulary richness (type-token ratio)
        vocab_richness = unique_words / total_words if total_words else 0.0

        # Formality score
        formality = self._compute_formality(words)

        # Question frequency
        question_freq = self._compute_question_frequency(sentences)

        # Filler word rate
        filler_rate = self._compute_filler_rate(combined, total_words)

        # Passive voice rate
        passive_rate = self._compute_passive_rate(sentences)

        # Technical depth
        tech_depth = self._compute_technical_depth(words)

        # Average word length
        avg_word_len = (
            sum(len(w) for w in words) / total_words
            if total_words else 0.0
        )

        # Exclamation rate
        exclamation_count = sum(
            1 for s in sentences if s.rstrip().endswith("!")
        )
        excl_rate = (
            exclamation_count / len(sentences) if sentences else 0.0
        )

        return StyleMetrics(
            avg_sentence_length=round(avg_sent_len, 2),
            vocabulary_richness=round(vocab_richness, 4),
            formality_score=round(formality, 4),
            question_frequency=round(question_freq, 4),
            filler_word_rate=round(filler_rate, 4),
            passive_voice_rate=round(passive_rate, 4),
            technical_depth=round(tech_depth, 4),
            avg_word_length=round(avg_word_len, 2),
            exclamation_rate=round(excl_rate, 4),
            total_words=total_words,
        )

    def extract_vocabulary(self, texts: list[str]) -> VocabularyProfile:
        """Extract characteristic vocabulary from text samples.

        Args:
            texts: List of text samples.

        Returns:
            VocabularyProfile with top words and bigrams.
        """
        if not texts:
            return VocabularyProfile()

        combined = " ".join(texts)
        words = _tokenize(combined)

        if not words:
            return VocabularyProfile()

        # Filter stop words for content words
        content_words = [w for w in words if w not in _STOP_WORDS]
        word_counts = Counter(content_words)
        top_words = [w for w, _ in word_counts.most_common(30)]

        # Bigrams from content words
        bigrams: list[str] = []
        for i in range(len(words) - 1):
            if words[i] not in _STOP_WORDS or words[i + 1] not in _STOP_WORDS:
                bigrams.append(f"{words[i]} {words[i + 1]}")
        bigram_counts = Counter(bigrams)
        top_bigrams = [
            b for b, c in bigram_counts.most_common(15) if c >= 2
        ]

        unique_ratio = len(set(words)) / len(words) if words else 0.0
        avg_word_len = (
            sum(len(w) for w in words) / len(words) if words else 0.0
        )

        return VocabularyProfile(
            top_words=top_words,
            top_bigrams=top_bigrams,
            unique_ratio=round(unique_ratio, 4),
            avg_word_length=round(avg_word_len, 2),
        )

    def detect_filler_words(
        self, texts: list[str],
    ) -> dict[str, float]:
        """Detect filler words and their relative frequency.

        Args:
            texts: List of text samples.

        Returns:
            Dict mapping filler words to frequency per 100 words.
        """
        if not texts:
            return {}

        combined = " ".join(texts)
        words = _tokenize(combined)
        total_words = len(words)

        if total_words == 0:
            return {}

        filler_counts: dict[str, int] = {}

        # Check multi-word fillers first
        lower_combined = combined.lower()
        for filler in FILLER_WORDS:
            if " " in filler:
                count = lower_combined.count(filler)
                if count > 0:
                    filler_counts[filler] = count
            else:
                count = words.count(filler)
                if count > 0:
                    filler_counts[filler] = count

        # Convert to rate per 100 words
        return {
            filler: round((count / total_words) * 100, 2)
            for filler, count in sorted(
                filler_counts.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        }

    def _compute_formality(self, words: list[str]) -> float:
        """Compute formality score from word list (0=casual, 1=formal)."""
        if not words:
            return 0.5

        formal_count = sum(1 for w in words if w in FORMAL_MARKERS)
        informal_count = sum(1 for w in words if w in INFORMAL_MARKERS)
        total_markers = formal_count + informal_count

        if total_markers == 0:
            return 0.5

        return formal_count / total_markers

    def _compute_question_frequency(
        self, sentences: list[str],
    ) -> float:
        """Compute fraction of sentences that are questions."""
        if not sentences:
            return 0.0

        question_count = sum(
            1 for s in sentences if s.rstrip().endswith("?")
        )
        return question_count / len(sentences)

    def _compute_filler_rate(
        self, text: str, total_words: int,
    ) -> float:
        """Compute filler words per 100 words."""
        if total_words == 0:
            return 0.0

        lower_text = text.lower()
        words = _tokenize(text)
        filler_count = 0

        for filler in FILLER_WORDS:
            if " " in filler:
                filler_count += lower_text.count(filler)
            else:
                filler_count += words.count(filler)

        return (filler_count / total_words) * 100

    def _compute_passive_rate(self, sentences: list[str]) -> float:
        """Compute fraction of sentences using passive voice."""
        if not sentences:
            return 0.0

        passive_count = sum(
            1 for s in sentences if _PASSIVE_RE.search(s)
        )
        return passive_count / len(sentences)

    def _compute_technical_depth(self, words: list[str]) -> float:
        """Compute technical jargon density (0-1)."""
        if not words:
            return 0.0

        tech_count = sum(1 for w in words if w in TECHNICAL_MARKERS)
        # Normalize: even 5% technical words is pretty high
        raw = tech_count / len(words)
        # Scale so 5% -> ~1.0
        return min(1.0, raw * 20)
