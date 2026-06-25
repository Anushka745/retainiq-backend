"""
Synthetic SaaS customer data generator.
Produces 30,000 realistic AcmeFlow customers with correlated churn signals.
"""

import random
import sqlite3
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PLAN_TYPES = ["Free", "Starter", "Pro", "Enterprise"]
PLAN_WEIGHTS = [0.30, 0.35, 0.25, 0.10]

FIRST_NAMES = [
    "James", "Maria", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
    "William", "Barbara", "David", "Elizabeth", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Lisa", "Daniel", "Nancy",
    "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
    "Steven", "Dorothy", "Paul", "Kimberly", "Andrew", "Emily", "Joshua", "Donna",
    "Kenneth", "Michelle", "Kevin", "Carol", "Brian", "Amanda", "George", "Melissa",
    "Timothy", "Deborah", "Ronald", "Stephanie", "Edward", "Rebecca", "Jason", "Sharon",
    "Jeffrey", "Laura", "Ryan", "Cynthia", "Jacob", "Kathleen", "Gary", "Amy",
    "Nicholas", "Angela", "Eric", "Shirley", "Jonathan", "Anna", "Stephen", "Brenda",
    "Larry", "Pamela", "Justin", "Emma", "Scott", "Nicole", "Brandon", "Helen",
    "Frank", "Samantha", "Raymond", "Katherine", "Gregory", "Christine", "Samuel", "Debra",
    "Benjamin", "Rachel", "Patrick", "Carolyn", "Jack", "Janet", "Dennis", "Catherine",
    "Jerry", "Maria", "Alexander", "Heather", "Tyler", "Diane", "Aaron", "Julie"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
    "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
    "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts", "Phillips", "Evans", "Turner", "Torres", "Parker", "Collins",
    "Edwards", "Stewart", "Flores", "Morris", "Nguyen", "Murphy", "Rivera", "Cook",
    "Rogers", "Morgan", "Peterson", "Cooper", "Reed", "Bailey", "Bell", "Gomez",
    "Kelly", "Howard", "Ward", "Cox", "Diaz", "Richardson", "Wood", "Watson",
    "Brooks", "Bennett", "Gray", "James", "Reyes", "Cruz", "Hughes", "Price",
    "Myers", "Long", "Foster", "Sanders", "Ross", "Morales", "Powell", "Sullivan",
    "Russell", "Ortiz", "Jenkins", "Gutierrez", "Perry", "Butler", "Barnes", "Fisher"
]

EMAIL_DOMAINS = [
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "company.io",
    "techcorp.com", "startup.co", "enterprise.net", "biz.org", "work.com"
]


def _fake_email(name: str, idx: int) -> str:
    slug = name.lower().replace(" ", ".")
    domain = EMAIL_DOMAINS[idx % len(EMAIL_DOMAINS)]
    return f"{slug}.{idx}@{domain}"


def _fake_user_id(idx: int) -> str:
    raw = f"acme-{idx:06d}"
    return "USR-" + hashlib.md5(raw.encode()).hexdigest()[:8].upper()


def _random_date(start_days_ago: int = 1460, end_days_ago: int = 7) -> str:
    delta = random.randint(end_days_ago, start_days_ago)
    d = datetime.now() - timedelta(days=delta)
    return d.strftime("%Y-%m-%d")


def generate_customers(n: int = 30_000) -> pd.DataFrame:
    """Generate n synthetic SaaS customers with correlated churn labels."""
    rng = np.random.default_rng(SEED)

    plan_type = rng.choice(PLAN_TYPES, size=n, p=PLAN_WEIGHTS)

    # Higher-tier plans → more engagement
    plan_engagement = {"Free": 0.4, "Starter": 0.6, "Pro": 0.8, "Enterprise": 0.95}
    base_engagement = np.array([plan_engagement[p] for p in plan_type])

    # Core behavioral features with realistic distributions
    subscription_age = np.clip(
        rng.integers(1, 1460, size=n).astype(float)
        * (0.5 + 0.5 * base_engagement),
        1, 1460
    ).astype(int)

    session_count = np.clip(
        rng.poisson(lam=20 * base_engagement + 2, size=n),
        0, 200
    ).astype(int)

    avg_session_duration = np.clip(
        rng.normal(loc=12 * base_engagement + 2, scale=5, size=n),
        0.5, 60
    ).round(2)

    days_since_last_login = np.clip(
        rng.exponential(scale=15 / (base_engagement + 0.1), size=n),
        0, 365
    ).astype(int)

    support_tickets = np.clip(
        rng.poisson(lam=1.5 + (1 - base_engagement) * 3, size=n),
        0, 20
    ).astype(int)

    feature_usage_score = np.clip(
        rng.beta(a=2 * base_engagement + 0.5, b=1.5, size=n) * 100,
        1, 100
    ).round(2)

    payment_failures = np.clip(
        rng.poisson(lam=(1 - base_engagement) * 2, size=n),
        0, 10
    ).astype(int)

    # Compute churn score (business logic)
    churn_score = (
        (days_since_last_login / 365) * 0.30
        + (1 - session_count / 200) * 0.20
        + (1 - feature_usage_score / 100) * 0.20
        + (payment_failures / 10) * 0.15
        + (support_tickets / 20) * 0.10
        + (1 - avg_session_duration / 60) * 0.05
    )
    churn_score = np.clip(churn_score + rng.normal(0, 0.08, n), 0, 1)
    # Target ~28% churn — use 40th percentile as dynamic threshold
    threshold = float(np.percentile(churn_score, 72))
    churn_label = (churn_score > threshold).astype(int)

    activity_score = np.clip(
        feature_usage_score * 0.4
        + (session_count / 200 * 100) * 0.3
        + (1 - days_since_last_login / 365) * 100 * 0.3,
        0, 100
    ).round(2)

    # Generate identifiers
    first = rng.choice(FIRST_NAMES, size=n)
    last = rng.choice(LAST_NAMES, size=n)
    names = [f"{f} {l}" for f, l in zip(first, last)]
    user_ids = [_fake_user_id(i) for i in range(n)]
    emails = [_fake_email(names[i], i) for i in range(n)]
    signup_dates = [_random_date() for _ in range(n)]

    df = pd.DataFrame(
        {
            "user_id": user_ids,
            "name": names,
            "email": emails,
            "signup_date": signup_dates,
            "plan_type": plan_type,
            "activity_score": activity_score,
            "churn_label": churn_label,
            "session_count": session_count,
            "avg_session_duration": avg_session_duration,
            "days_since_last_login": days_since_last_login,
            "support_tickets": support_tickets,
            "subscription_age": subscription_age,
            "feature_usage_score": feature_usage_score,
            "payment_failures": payment_failures,
        }
    )
    return df


def seed_database(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    """Insert generated customers into the users table (skip existing)."""
    cursor = conn.cursor()
    rows = df.to_dict(orient="records")

    cursor.executemany(
        """
        INSERT OR IGNORE INTO users (
            user_id, name, email, signup_date, plan_type, activity_score, churn_label,
            session_count, avg_session_duration, days_since_last_login,
            support_tickets, subscription_age, feature_usage_score, payment_failures
        ) VALUES (
            :user_id, :name, :email, :signup_date, :plan_type, :activity_score, :churn_label,
            :session_count, :avg_session_duration, :days_since_last_login,
            :support_tickets, :subscription_age, :feature_usage_score, :payment_failures
        )
        """,
        rows,
    )
    conn.commit()
    print(f"[DATA] Seeded {len(rows)} customers into users table.")


if __name__ == "__main__":
    df = generate_customers()
    csv_path = Path(__file__).parent / "customers.csv"
    df.to_csv(csv_path, index=False)
    print(f"[DATA] Saved {len(df)} rows → {csv_path}")
    print(df.head(3).to_string())
    print(f"\nChurn rate: {df['churn_label'].mean():.2%}")
