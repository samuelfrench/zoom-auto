"""Recording prompts for voice sample collection.

These prompts cover a range of speech patterns typical of Zoom meetings:
greetings, technical statements, questions, casual chat, and emotional range.
"""

from __future__ import annotations

RECORDING_PROMPTS: list[str] = [
    "Hey everyone, good morning. How's it going?",
    "Yesterday I worked on the API integration and got the auth flow working.",
    "Today I'm planning to finish the database migration and write tests.",
    "No blockers on my end right now.",
    "I think we should prioritize the deployment pipeline this sprint.",
    "What's the current status on the frontend redesign?",
    "That's a good point. I hadn't thought about it from that angle.",
    "I'm not sure I agree with that approach. Here's my concern...",
    "The latency issue is probably caused by the N+1 query in the user service.",
    "Could you walk me through the architecture decisions there?",
    "Yeah, that makes total sense. I'm on board with that.",
    "Hmm, let me think about that for a second.",
    "I ran into a weird bug with the OAuth callback yesterday.",
    "The numbers look good this quarter. We're trending above target.",
    "I'll take that action item. Should have it done by end of week.",
    "One two three four five six seven eight nine ten.",
    "Can we circle back to the resource allocation question?",
    "I'd estimate about three to five days for that feature.",
    "Great meeting everyone. Talk to you all tomorrow.",
    "The quick brown fox jumps over the lazy dog.",
]
