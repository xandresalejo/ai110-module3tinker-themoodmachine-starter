# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

Pipeline:
  - Preprocess the text (lowercase, normalize, tokenize, keep emojis)
  - Look for positive and negative words
  - Handle negation ("not happy" -> negative, "not bad" -> positive)
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
import string
from typing import List, Optional, Tuple

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS

# Words that flip the meaning of the next sentiment word.
NEGATION_WORDS = {
    "not", "no", "never", "nothing",
    "isn't", "wasn't", "aren't", "don't", "doesn't",
    "didn't", "can't", "won't", "ain't",
}

# How many tokens after a negation word it can still "reach".
# Example: "not very happy" -> "not" still affects "happy".
NEGATION_WINDOW = 3

# Token pattern, checked in order:
#   1. text emoticons like :) :( ;) :-( :d :p
#   2. words and numbers, keeping apostrophes ("can't")
#   3. any other single symbol (this is what catches emojis like 🔥 💀)
TOKEN_PATTERN = re.compile(r"[:;]-?[()dp]|[a-z0-9']+|[^\w\s]")


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens.

        Steps:
          - Strip whitespace and lowercase
          - Turn curly apostrophes into straight ones ("can’t" -> "can't")
          - Shorten repeated characters ("soooo" -> "soo", "!!!!" -> "!!")
          - Split into words, emoticons, and emojis
          - Drop plain punctuation (",", "!", "?") but keep emojis/emoticons
        """
        cleaned = text.strip().lower()
        cleaned = cleaned.replace("\u2019", "'")
        cleaned = re.sub(r"(.)\1{2,}", r"\1\1", cleaned)

        raw_tokens = TOKEN_PATTERN.findall(cleaned)

        tokens: List[str] = []
        for token in raw_tokens:
            if token in string.punctuation:
                continue  # skip lone punctuation marks
            token = token.strip("'") if len(token) > 1 else token
            if token:
                tokens.append(token)

        return tokens

    # ---------------------------------------------------------------------
    # Core analysis (shared by score_text and explain)
    # ---------------------------------------------------------------------

    def _analyze(self, text: str) -> Tuple[int, List[str], List[str]]:
        """
        Walk through the tokens once and return:
          (score, positive_hits, negative_hits)

        Enhancement: negation handling.
        A negation word flips the sign of the next sentiment word,
        as long as it appears within NEGATION_WINDOW tokens.
        """
        tokens = self.preprocess(text)

        score = 0
        positive_hits: List[str] = []
        negative_hits: List[str] = []
        negate_left = 0  # tokens remaining where a negation still applies

        for token in tokens:
            if token in NEGATION_WORDS:
                negate_left = NEGATION_WINDOW
                continue

            if token in self.positive_words:
                value = 1
            elif token in self.negative_words:
                value = -1
            else:
                value = 0

            if value == 0:
                # Not a sentiment word: the negation fades a bit.
                if negate_left > 0:
                    negate_left -= 1
                continue

            label = token
            if negate_left > 0:
                value = -value
                label = f"not {token}"
                negate_left = 0  # a negation only flips one word

            if value > 0:
                positive_hits.append(label)
            else:
                negative_hits.append(label)
            score += value

        return score, positive_hits, negative_hits

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

          - Start at 0
          - Positive word -> +1
          - Negative word -> -1
          - A negated word counts the opposite way
        """
        score, _, _ = self._analyze(text)
        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score into a mood label.

          - Both positive AND negative signals, and the score is close
            to zero (-1, 0, or 1) -> "mixed"
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"
        """
        score, positive_hits, negative_hits = self._analyze(text)

        if positive_hits and negative_hits and abs(score) <= 1:
            return "mixed"
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        Example:
          'Score = -1 (positive: [], negative: ['not happy'])'
        """
        score, positive_hits, negative_hits = self._analyze(text)

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )