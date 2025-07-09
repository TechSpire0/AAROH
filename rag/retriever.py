import pandas as pd

# --- Load CSV with example queries and tags ---
CSV_PATH = "examples.csv"
df = pd.read_csv(CSV_PATH)
# Add a column with tag lists for easier matching
df["tags_list"] = df["tags"].apply(lambda x: [t.strip() for t in x.split(",")])

def retrieve_similar_examples(user_query, top_n=6):
    """
    Retrieve top N examples from the CSV whose tags overlap with the user query.
    Args:
        user_query: str, the user's question
        top_n: int, number of examples to return
    Returns:
        DataFrame of top matches or None if no match
    """
    query_words = set(user_query.lower().split())

    def score_row(row):
        tags = set(row["tags_list"])
        return len(tags & query_words)

    df["score"] = df.apply(score_row, axis=1)
    top_matches = df[df["score"] > 0].sort_values(by="score", ascending=False).head(top_n)

    return top_matches if not top_matches.empty else None

    def score_row(row):
        tags = set(row["tags_list"])
        return len(tags & query_words)

    df["score"] = df.apply(score_row, axis=1)
    top_matches = df[df["score"] > 0].sort_values(by="score", ascending=False).head(top_n)

    return top_matches if not top_matches.empty else None
