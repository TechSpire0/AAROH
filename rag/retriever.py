import pandas as pd

# Load CSV once
CSV_PATH = "examples.csv"  # Update this path accordingly
df = pd.read_csv(CSV_PATH)
df["tags_list"] = df["tags"].apply(lambda x: [t.strip() for t in x.split(",")])

def retrieve_similar_example(user_query, top_n=1):
    query_words = set(user_query.lower().split())

    def score_row(row):
        tags = set(row["tags_list"])
        return len(tags & query_words)

    df["score"] = df.apply(score_row, axis=1)
    top_matches = df.sort_values(by="score", ascending=False).head(top_n)

    return top_matches.iloc[0] if not top_matches.empty else None
