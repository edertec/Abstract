import pandas as pd
import spacy
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import STOPWORDS

# Load the spaCy model for English
nlp = spacy.load("en_core_web_sm")

# Step 1: Load the CSV data
df = pd.read_csv("sintese_estudos.csv")

# Define a list of additional stop words (common words we don't want in the keywords)
custom_stop_words = STOPWORDS.union({
    "the", "a", "an", "of", "and", "to", "in", "for", "on", "with", "this", "study", "paper", "research",
    "is", "are", "was", "were", "has", "have", "be", "by", "which", "that", "from", "as", "it", "also",
    "can", "may", "such", "these", "one", "two", "used", "using", "abstract", "synthesize", "problem", "context", "describe", "address"
})

# Step 2: Function to extract keywords and remove stop words
def extract_keywords(text):
    doc = nlp(text)
    keywords = [
        token.lemma_.lower()
        for token in doc
        if token.pos_ in {"NOUN", "PROPN", "VERB", "ADJ"}
        and token.lemma_ not in custom_stop_words
        and not token.is_stop
        and not token.is_punct
    ]
    return keywords

# Step 3: Apply the keyword extraction function to the Objectives and Problems columns
df["Objectives Keywords"] = df["objectives"].apply(lambda text: extract_keywords(str(text)))
df["Problems Keywords"] = df["problem"].apply(lambda text: extract_keywords(str(text)))

# Step 4: Flatten the lists and get frequency counts for top keywords in both columns
objectives_keywords = [keyword for sublist in df["Objectives Keywords"] for keyword in sublist]
problems_keywords = [keyword for sublist in df["Problems Keywords"] for keyword in sublist]

# Step 5: Get the top 10 keywords
top_objectives_keywords = Counter(objectives_keywords).most_common(10)
top_problems_keywords = Counter(problems_keywords).most_common(10)

# Step 6: Convert to DataFrame for easier plotting
top_keywords_df = pd.DataFrame({
    "Objectives Keywords": dict(top_objectives_keywords),
    "Problems Keywords": dict(top_problems_keywords)
})

# Step 7: Create the plot
top_keywords_df.plot(kind="bar", figsize=(10, 6), width=0.8)
plt.title("Top 10 Keywords in Objectives and Problems")
plt.ylabel("Frequency")
plt.xlabel("Keywords")
plt.xticks(rotation=45)
plt.legend(loc="best")
plt.tight_layout()

# Step 8: Save the plot
plt.savefig("filtered_keywords_top10.png")

# Show the plot
plt.show()