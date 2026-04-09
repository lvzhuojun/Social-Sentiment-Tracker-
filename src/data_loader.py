"""
src/data_loader.py — Data loading, cleaning, and splitting utilities.

Supports the Sentiment140 dataset (twitter_training.csv) and falls back to
auto-generated mock data when the real dataset is unavailable.
"""

import re
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    MOCK_DATA_PATH,
    RANDOM_SEED,
    TEST_SIZE,
    TWEET_EVAL_PATH,
    VAL_SIZE,
    get_logger,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Sentiment140 loader
# ---------------------------------------------------------------------------


def load_sentiment140(filepath: Path | str) -> pd.DataFrame:
    """Load and normalise the Sentiment140 CSV dataset.

    The raw file has no header; columns are assigned and label 4 is mapped to 1
    so the result is a binary classification problem (0 = negative, 1 = positive).

    Args:
        filepath: Path to the CSV file (``twitter_training.csv``).

    Returns:
        DataFrame with columns ``['id', 'label', 'user', 'date', 'query', 'text']``
        and labels in {0, 1}.

    Raises:
        FileNotFoundError: If *filepath* does not exist.

    Example:
        >>> df = load_sentiment140("data/raw/twitter_training.csv")
        >>> df['label'].value_counts()
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    logger.info("Loading Sentiment140 from %s", filepath)
    df = pd.read_csv(
        filepath,
        encoding="latin-1",
        header=None,
        names=["label", "id", "date", "query", "user", "text"],
    )

    # Map 4 → 1 for binary classification
    df["label"] = df["label"].map({0: 0, 4: 1})
    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].astype(int)

    logger.info(
        "Loaded %d rows, label distribution: %s",
        len(df), df["label"].value_counts().to_dict(),
    )
    return df


# ---------------------------------------------------------------------------
# TweetEval loader
# ---------------------------------------------------------------------------


def load_tweet_eval(filepath: Path | str) -> pd.DataFrame:
    """Load the downloaded TweetEval sentiment CSV.

    Expected CSV columns: ``label, text, id``.
    Labels follow project convention: 0=negative, 1=positive, 2=neutral.

    Args:
        filepath: Path to ``tweet_eval_sentiment.csv``.

    Returns:
        DataFrame with columns ``['id', 'label', 'text']``.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"TweetEval dataset not found: {filepath}")

    logger.info("Loading TweetEval sentiment from %s", filepath)
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].astype(int)

    logger.info(
        "Loaded %d rows, label distribution: %s",
        len(df), df["label"].value_counts().to_dict(),
    )
    return df


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

# Pre-compiled patterns for performance
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")          # keep the word, drop the #
_NON_ALPHA_RE = re.compile(r"[^a-z0-9\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Clean a raw social-media string.

    Steps applied in order:
    1. Lower-case
    2. Remove URLs (http/https/www)
    3. Remove @mentions
    4. Strip ``#`` from hashtags (keep the word)
    5. Remove remaining non-alphanumeric characters
    6. Collapse multiple whitespace to a single space

    Args:
        text: Raw input string.

    Returns:
        Cleaned string (may be empty if all tokens were noise).

    Example:
        >>> clean_text("Hello @user! Check https://example.com #NLP :)")
        'hello check nlp'
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(r" \1 ", text)   # #nlp → " nlp "
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# DataFrame preprocessing
# ---------------------------------------------------------------------------

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply :func:`clean_text` to the ``text`` column and remove noise rows.

    Args:
        df: DataFrame containing at least a ``text`` column.

    Returns:
        Cleaned DataFrame with:
        * ``clean_text`` column added
        * Rows with empty clean text removed
        * Duplicate rows removed
        * Index reset

    Example:
        >>> df = pd.DataFrame({'text': ['Hello world!', '', None], 'label': [1, 0, 1]})
        >>> preprocess_dataframe(df)
    """
    logger.info("Preprocessing %d rows …", len(df))
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_text)

    before = len(df)
    df = df[df["clean_text"].str.strip().astype(bool)]   # drop empty
    df = df.drop_duplicates(subset=["clean_text"])
    df = df.reset_index(drop=True)

    logger.info("Preprocessing complete: %d → %d rows", before, len(df))
    return df


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    save_dir: Path | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split into train, validation, and test sets.

    Args:
        df: Full cleaned DataFrame with a ``label`` column.
        test_size: Fraction reserved for the test set (default 0.2).
        val_size: Fraction of the *total* data reserved for validation
                  (default 0.1, i.e. 10 % of total).
        save_dir: If provided, save ``train.csv``, ``val.csv``, ``test.csv``
                  to this directory. Defaults to ``config.PROCESSED_DATA_DIR``.

    Returns:
        Tuple of ``(train_df, val_df, test_df)``.

    Example:
        >>> train, val, test = split_data(df)
        >>> len(train) + len(val) + len(test) == len(df)
        True
    """
    from config import PROCESSED_DATA_DIR  # local import to avoid circular

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size / (1 - test_size),
        random_state=RANDOM_SEED,
        stratify=train_val_df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info(
        "Split sizes — train: %d | val: %d | test: %d",
        len(train_df), len(val_df), len(test_df),
    )

    # Persist processed splits
    out_dir = Path(save_dir) if save_dir else PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    logger.info("Processed splits saved to %s", out_dir)

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Mock data generator (development fallback)
# ---------------------------------------------------------------------------

_POSITIVE_TEMPLATES = [
    "I absolutely love this product! It works perfectly.",
    "What an amazing day! Feeling grateful and happy.",
    "Great job team, the new feature is fantastic.",
    "Just had the best coffee ever. Highly recommend!",
    "So excited about the upcoming vacation. Can't wait!",
    "The customer service was outstanding. Five stars!",
    "This movie was brilliant. Loved every minute of it.",
    "Feeling motivated and ready to take on the world today.",
    "The weather is beautiful today. Perfect for a walk.",
    "Just finished a great book. Totally recommend it!",
    "This restaurant exceeded all my expectations. Incredible food!",
    "My new laptop arrived and it's blazing fast. So happy!",
    "The concert last night was absolutely phenomenal!",
    "Promotion came through! Hardest I've worked for anything.",
    "Finally finished the project and the client loved it.",
    "Best birthday surprise ever. Feeling so loved and grateful.",
    "The training session today was incredibly insightful.",
    "My team delivered amazing results this quarter!",
    "Just received the most thoughtful gift. Made my day.",
    "Woke up feeling energised and ready to crush it today.",
    "The new update is smooth, fast, and beautifully designed.",
    "Just got a standing ovation during the presentation. Wow!",
    "Delicious homemade pasta tonight. Life is good.",
    "Marathon completed! Never felt more proud of myself.",
    "Surprise visit from an old friend. Best afternoon ever.",
    "The sunset view from the hilltop was absolutely breathtaking.",
    "Finally mastered this algorithm. Persistence pays off!",
    "My kids' laughter is the best sound in the world.",
    "Exceeded my sales target for the third month in a row!",
    "Great news from the doctor today. Healthy as ever!",
    "The team pulled together under pressure and nailed it.",
    "Waking up early has genuinely changed my life for the better.",
    "New gym personal record today. Consistency is everything.",
    "The book club discussion tonight was absolutely riveting.",
    "Volunteer work today reminded me why kindness matters so much.",
    "The new coffee shop downtown is an absolute gem.",
    "Passed my certification exam on the first attempt!",
    "Children's laughter filled the park today. Pure joy.",
    "Just adopted a puppy. Officially the best day ever.",
    "The quarterly earnings crushed expectations. Great teamwork!",
    "Fresh air, good company, perfect hiking trail. Couldn't ask for more.",
    "Mentoring session today was deeply rewarding for both of us.",
    "Just saw the most heartwarming short film. Highly recommended.",
    "The product launch went even better than planned. Thrilled!",
    "Homemade sourdough finally turned out perfect. Worth every hour.",
    "The scholarship came through! Dreams really do come true.",
    "My flight was upgraded to business class. What a treat!",
    "Finished the quarter ahead of every single target. Incredible feeling.",
    "The garden is blooming beautifully this spring.",
    "Customer feedback scores hit an all-time high this month!",
]

_NEGATIVE_TEMPLATES = [
    "Terrible experience. The product broke after one day.",
    "I'm so frustrated with this service. Total waste of money.",
    "The worst customer support I've ever encountered.",
    "Really disappointed with the quality. Not worth it.",
    "Had a horrible commute today. Everything went wrong.",
    "The app keeps crashing. Absolutely unusable right now.",
    "So tired of all this unnecessary drama and stress.",
    "This is the third time my order was delayed. Unacceptable.",
    "The food was cold and tasteless. Never coming back.",
    "Feeling really down today. Nothing seems to go right.",
    "Waited two hours and still no response from customer service.",
    "The new software update broke everything that was working fine.",
    "Flight cancelled with zero notice. Absolutely furious right now.",
    "Spent three hours on hold just to get disconnected.",
    "The contractor left a complete mess. Extremely unprofessional.",
    "My subscription was charged twice this month without explanation.",
    "Devastating news from the doctor today. Really struggling.",
    "The conference was disorganised and an utter waste of time.",
    "Laptop died right before the big deadline. Absolute nightmare.",
    "The rental car had hidden fees that doubled the original price.",
    "Horrible experience at the hotel. Dirty room, rude staff.",
    "The new manager micromanages everything. Morale is at rock bottom.",
    "Three packages lost in transit this year from the same courier.",
    "The refund was denied despite the product being defective.",
    "Terrible traffic made me two hours late to an important meeting.",
    "The restaurant overcharged and refused to correct the bill.",
    "Gym equipment broken for weeks and still no maintenance.",
    "Constant bugs in the app are making simple tasks impossible.",
    "The promised delivery date was pushed back for the fifth time.",
    "Awful noise from upstairs neighbours every single night.",
    "The project got cancelled after months of hard work. Devastating.",
    "My data was corrupted and there were no backups available.",
    "The bank's system locked me out during an urgent transaction.",
    "Poorly written instructions caused the entire setup to fail.",
    "The so-called express lane took longer than regular checkout.",
    "Rude service ruined what should have been a lovely evening out.",
    "The subscription auto-renewed without any warning or reminder.",
    "Three attempts to fix the same issue and still broken.",
    "Misleading advertisement got me to pay for a useless product.",
    "The online course was outdated and full of factual errors.",
    "Wi-Fi keeps dropping during critical video calls. Unacceptable.",
    "The medicine had serious side effects not listed on the label.",
    "Scheduled maintenance brought down the entire platform.",
    "My car repair cost three times the original estimate.",
    "The new policy makes an already difficult job even harder.",
    "Completely unhelpful chatbot wasted thirty minutes of my time.",
    "The charity event was poorly managed and raised almost nothing.",
    "Toxic work environment is draining every last bit of energy.",
    "Important email went to spam and cost us a major contract.",
    "The keynote speaker was unprepared and completely off-topic.",
]

_NEUTRAL_TEMPLATES = [
    "Just got back from the grocery store.",
    "The meeting is scheduled for 3 PM tomorrow.",
    "I updated the software to the latest version.",
    "The report has been submitted for review.",
    "Currently reading a book about machine learning.",
    "The bus arrives every 20 minutes on weekdays.",
    "Watched a documentary about climate change last night.",
    "The package was delivered this morning.",
    "Attended a webinar on data science trends.",
    "The library is open until 9 PM on Fridays.",
    "The quarterly report is due at the end of the week.",
    "Set up a new workspace in the home office today.",
    "Installed the latest security patch on the server.",
    "The annual conference is happening in October this year.",
    "Printed the slides for tomorrow's presentation.",
    "Renewed the domain registration for another two years.",
    "The new employee handbook has been distributed to all staff.",
    "Backed up all project files to the cloud storage.",
    "The canteen is closed on public holidays.",
    "Sent the invoice to the client this afternoon.",
    "The dashboard shows stable traffic over the past seven days.",
    "Completed the mandatory compliance training this morning.",
    "The engineering team is reviewing the pull request.",
    "Updated the project timeline in the shared calendar.",
    "The database migration is scheduled for Saturday night.",
    "Two new team members will join the department next Monday.",
    "The weekly standup has been moved to 9:30 AM.",
    "Ordered replacement parts for the office printer.",
    "The parking lot will be closed for maintenance on Thursday.",
    "Submitted the expense reimbursement form to finance.",
    "The road construction on Fifth Avenue is expected to continue.",
    "Confirmed the venue booking for the team offsite.",
    "The API documentation has been updated to reflect version 3.",
    "Turned off automatic replies now that the holiday is over.",
    "The water cooler will be replaced by facilities on Friday.",
    "Reviewed the vendor proposal and forwarded it to procurement.",
    "The fire drill is scheduled for next Wednesday at 11 AM.",
    "System monitoring shows normal CPU and memory usage.",
    "The new product SKU has been added to the inventory system.",
    "Formatted the hard drive and reinstalled the operating system.",
    "The onboarding session for new hires runs until noon.",
    "Mailed the contract to the legal team for final review.",
    "The climate control settings were adjusted in the server room.",
    "Checked the flight status — departure is running on time.",
    "The research paper has been submitted to the journal.",
    "Archived last year's project files to long-term storage.",
    "Office supplies have been restocked for the month.",
    "The software licence expires at the end of next quarter.",
    "Posted the agenda for Friday's board meeting.",
    "Registered for the upcoming Python conference in the city.",
]


def generate_mock_data(n: int = 500, save_path: Path | None = None) -> pd.DataFrame:
    """Generate synthetic sentiment data for development / testing.

    Produces a balanced dataset covering positive (label=1), negative (label=0),
    and neutral (label=2) classes with realistic-looking timestamps.

    Args:
        n: Total number of rows to generate (default 500).
        save_path: If provided, save the CSV to this path.

    Returns:
        DataFrame with columns ``['id', 'label', 'date', 'text']``.

    Example:
        >>> df = generate_mock_data(300)
        >>> df['label'].value_counts()
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    today = datetime.today()
    records = []
    per_class = n // 3
    remainder = n - per_class * 3

    templates = [
        (1, _POSITIVE_TEMPLATES),
        (0, _NEGATIVE_TEMPLATES),
        (2, _NEUTRAL_TEMPLATES),
    ]

    # Variation prefixes / suffixes to ensure textual diversity
    _prefixes = [
        "", "Honestly, ", "Just wanted to say: ", "Update: ", "Quick note — ",
        "Day 3: ", "Week recap: ", "Personal opinion: ", "Hot take: ", "FYI: ",
    ]
    _suffixes = [
        "", ".", "!", " 100%.", " No question.", " Seriously.",
        " Worth sharing.", " Just my two cents.", " Period.", " Truly.",
    ]

    idx = 0
    for i, (label, pool) in enumerate(templates):
        count = per_class + (1 if i < remainder else 0)
        for j in range(count):
            days_ago = random.randint(0, 90)
            date_str = (today - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            # Rotate through pool and add prefix/suffix variation for uniqueness
            base_text = pool[j % len(pool)]
            prefix = random.choice(_prefixes) if random.random() < 0.35 else ""
            suffix = random.choice(_suffixes) if random.random() < 0.25 else ""
            text = (prefix + base_text.rstrip("!.") + suffix).strip()
            records.append({"id": idx, "label": label, "date": date_str, "text": text})
            idx += 1

    df = pd.DataFrame(records).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info("Mock data saved to %s", save_path)

    logger.info(
        "Generated %d mock records (label dist: %s)",
        len(df), df["label"].value_counts().to_dict(),
    )
    return df


# ---------------------------------------------------------------------------
# Convenience loader — auto-selects real data or mock fallback
# ---------------------------------------------------------------------------

def load_data(real_path: Path | None = None) -> pd.DataFrame:
    """Load the best available dataset.

    Tries the Sentiment140 CSV first; falls back to mock data if unavailable.

    Args:
        real_path: Optional explicit path to the Sentiment140 CSV.
                   Defaults to ``config.SENTIMENT140_PATH``.

    Returns:
        Preprocessed DataFrame ready for modelling.
    """
    from config import SENTIMENT140_PATH  # local import to avoid circular

    # Priority: real_path arg → TweetEval → Sentiment140 → mock
    if real_path:
        explicit = Path(real_path)
        if explicit.exists():
            try:
                df = load_tweet_eval(explicit)
            except Exception:
                df = load_sentiment140(explicit)
                df = df[df["label"].isin([0, 1])].copy()
        else:
            raise FileNotFoundError(f"Specified path not found: {explicit}")
    elif TWEET_EVAL_PATH.exists():
        df = load_tweet_eval(TWEET_EVAL_PATH)
    else:
        try:
            df = load_sentiment140(SENTIMENT140_PATH)
            df = df[df["label"].isin([0, 1])].copy()
        except FileNotFoundError:
            logger.warning("No real dataset found — generating mock data.")
            df = generate_mock_data(save_path=MOCK_DATA_PATH)

    df = preprocess_dataframe(df)
    return df


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print("Shape:", df.shape)
    print("Labels:", df["label"].value_counts().to_dict())
