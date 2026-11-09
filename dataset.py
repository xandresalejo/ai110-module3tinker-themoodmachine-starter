"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
    # Part 3 fix: slang + emoji signals the starter list missed
    "fire",
    "🔥",
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
    # Part 3 fix: slang + emoji signals the starter list missed
    "trash",
    "ugh",
    "💀",
    "😩",
]

# ---------------------------------------------------------------------
# Labeled dataset
# ---------------------------------------------------------------------

SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
]

TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
]

# --- Added posts (Part 1) ---
# Each post sits next to its label so the two lists can't drift apart.
NEW_POSTS = [
    ("Lowkey stressed but kind of proud of myself", "mixed"),
    ("I absolutely love getting stuck in traffic", "negative"),   # sarcasm
    ("that concert was fire 🔥🔥 no cap", "positive"),             # slang + emoji
    ("I'm fine 🙂", "mixed"),                                      # ambiguous
    ("not bad at all, kinda impressed", "positive"),              # negation
    ("this update is lowkey trash 💀", "negative"),                # slang
    ("ugh mondays 😩", "negative"),                                # emoji
    ("got the job!!! can't stop smiling", "positive"),
    ("going to the store later", "neutral"),
]

for post, label in NEW_POSTS:
    SAMPLE_POSTS.append(post)
    TRUE_LABELS.append(label)

assert len(SAMPLE_POSTS) == len(TRUE_LABELS), "Posts and labels are out of sync!"

