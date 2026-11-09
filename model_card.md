# Model Card: The Mood Machine

## Intended Use

The Mood Machine classifies short, informal posts (like social media updates or text messages) as **positive**, **negative**, **neutral**, or **mixed**. It is a learning project for comparing a hand-written rule-based classifier with a small machine learning classifier. It is **not** meant for real decisions about people, such as content moderation or mental health screening.

## Data

- **Size:** 15 labeled posts in `SAMPLE_POSTS` / `TRUE_LABELS` (`dataset.py`).
- **Starter data:** 6 posts written in plain, standard English.
- **What I added:** 9 posts covering realistic language:
  - slang ("lowkey", "fire", "no cap", "trash")
  - emojis (🔥, 💀, 😩, 🙂)
  - sarcasm ("I absolutely love getting stuck in traffic")
  - negation ("not bad at all")
  - mixed feelings ("Lowkey stressed but kind of proud of myself")
  - one plain neutral statement
- **How I labeled them:** one label per post, based on what the writer most likely meant, not on the literal words. Some labels are debatable. "I'm fine 🙂" could be neutral, mixed, or passive-aggressive, and I labeled it mixed. Different people could reasonably disagree.
- Each post is stored next to its label, and an `assert` checks that the two lists stay the same length.

## How the Model Works

### Rule-based model (`mood_analyzer.py`)

1. **`preprocess`**
   - Lowercases the text and fixes curly apostrophes.
   - Shortens stretched letters ("soooo" becomes "soo").
   - Splits the text into words, emoticons (`:)`), and emojis, and drops plain punctuation.
2. **`score_text`**
   - Starts at 0, adds +1 for each word in `POSITIVE_WORDS`, and subtracts 1 for each word in `NEGATIVE_WORDS`.
3. **Enhancement: negation handling.**
   - A negation word ("not", "never", "can't", "no", and similar) flips the next sentiment word within 3 tokens.
   - So "not happy" scores −1 and "not bad" scores +1.
4. **`predict_label`**
   - If the post contains both positive and negative words and the score is between −1 and 1, it returns **mixed**.
   - Otherwise: score > 0 is positive, score < 0 is negative, and score = 0 is neutral.
5. **`explain`** shows the score and which words counted, for example `negative: ['not happy']`.

### ML model (`ml_experiments.py`)

A bag-of-words `CountVectorizer` combined with `LogisticRegression` (scikit-learn), trained on the same 15 posts. It learns which words tend to appear with which labels, instead of following rules I wrote.

## Evaluation

### Rule-based

| Version | Accuracy |
|---|---|
| Before the Part 3 fix | 0.47 (7/15) |
| After the Part 3 fix | **0.67 (10/15)** |

**Part 3 fix.** The failure pattern I chose was slang and emojis being ignored because they weren't in the word lists. I added "fire" and 🔥 to the positive list, and "trash", "ugh", 💀, and 😩 to the negative list. This fixed three posts: the concert post, the "trash 💀" post, and "ugh mondays 😩". It didn't break any other examples in the dataset.

Posts it still gets wrong:

| Post | Predicted | True | Why |
|---|---|---|---|
| I absolutely love getting stuck in traffic | positive | negative | Sarcasm: it only sees "love" |
| Feeling tired but kind of hopeful | negative | mixed | "hopeful" isn't a known word |
| Lowkey stressed but kind of proud of myself | negative | mixed | "proud" isn't a known word |
| I'm fine 🙂 | neutral | mixed | "fine" and 🙂 carry no score |
| got the job!!! can't stop smiling | neutral | positive | "smiling" is unknown |

### ML model

- **Accuracy: 1.00** on the same 15 posts. This number is misleading, because the model is tested on the exact posts it was trained on. It memorized the examples rather than learning mood.
- New sentences not in the dataset show what it actually learned:

| New sentence | ML prediction | What happened |
|---|---|---|
| this movie was fire | negative | Wrong. It weights words like "this" and "was" from other posts |
| not happy at all | positive | Wrong. It learned "not … at all" from "not bad at all" |
| my house caught fire | positive | Wrong. "fire" became a positive word |
| I love sitting in traffic | negative | Right, but only because "traffic" appeared in the sarcasm post |

## Limitations

- **Sarcasm.** "I absolutely love getting stuck in traffic" is labeled positive because the rules only see "love." A word-matching system can't tell when the context reverses the meaning.
- **Word meaning depends on context.** After the fix, "my house caught fire" is classified as *positive* by both models, because "fire" is now a positive word everywhere. Fixing slang introduced a new failure.
- **Negation clashes with slang.** In "fire 🔥🔥 no cap," the word "no" triggers the negation rule, even though "no cap" means "for real." It happened not to matter here, because no sentiment word followed it, but in a post like "no cap that was amazing," the negation would flip "amazing" to negative.
- **Emojis are ambiguous.** I listed 💀 as negative, but people often use it to mean "I'm dying laughing," which is positive.
- **Tiny vocabulary.** Common emotional words like "proud," "hopeful," and "smiling" score 0, so many real posts default to neutral.
- **The ML model ignores emojis.** `CountVectorizer` only keeps word tokens with 2 or more letters, so 🔥, 💀, 😩, and 🙂 are thrown away completely.
- **Tiny dataset.** With 15 examples, both models are fragile. A single added post can change the ML model's behavior.

## Ethical Considerations

### Overt bias

The starter word lists only contained standard English words. Before my fix, every slang or emoji-heavy post scored neutral, regardless of what it meant. The errors clustered around informal writing, not around the content of the posts.

This means the model underserves:
- younger users and people who write mostly in slang and emoji
- speakers of dialects such as African American Vernacular English, where expressions like "no cap" come from

My fix only added the few slang terms in my own dataset. Slang from other communities, regions, or languages would still be misread. I documented this rather than trying to fully fix it.

### Covert compliance

I used an AI assistant (Claude) to help write the scoring logic and choose new words for the lists.

**What it raised on its own:**
- that sarcasm would fool the model
- that "no" in "no cap" would trigger negation
- that the ML model drops emojis
- that the ML model's 1.00 accuracy came from testing on the training data

**What it did not raise:**
- It did not warn that adding "fire" as a positive word would make "my house caught fire" positive. That only showed up when I tested a new sentence.
- It did not question whose slang the word lists represent. It added the terms from my dataset without asking whether they cover other groups' language.

In hindsight, it should have flagged both, since each one is a coverage problem that a reviewer would catch.

### Misuse risk

A system this simple should never be used to judge someone's emotional state or flag their posts. It would treat informal writers unfairly and misread sarcasm.

## Rule-Based vs. ML Comparison

- **Did they behave differently?** Yes. The rule-based model fails in predictable ways that I can explain, such as unknown words or sarcasm. The ML model fails in surprising ways, because it latches onto unrelated words like "this," "was," and "at all."
- **Did ML fix or introduce failures?** On the training posts it "fixed" everything, but only by memorizing them. On new sentences it introduced new errors, like calling "not happy at all" positive, which the rule-based model gets right thanks to negation handling.
- **How sensitive is it to labels?** Very. Every word in a post inherits that post's label, so one label choice (for example, the "not bad at all" post being positive) changed how the model treats "not" in other sentences. The rule-based model only changes when I edit the word lists or logic, so it's less sensitive to the data but needs more manual work.
