# ===== NLP TEXT RANKING =====

from transformers import pipeline

def load_model():
    print("Loading ranking model...")
    return pipeline("text-classification", model="cross-encoder/ms-marco-MiniLM-L-6-v2")

def rank_sentences(ranker, query, sentences):
    pairs = [[query, s] for s in sentences]
    results = ranker(pairs)

    ranked = []
    for sent, res in zip(sentences, results):
        ranked.append((sent, res['score']))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked

def main():
    ranker = load_model()
    print("Model loaded!\n")

    while True:
        print("\nOPTIONS:")
        print("1. Rank Sentences")
        print("2. Exit")

        choice = input("Enter choice: ")

        if choice == "1":
            query = input("\nEnter query: ").strip()

            if not query:
                print("Empty query!")
                continue

            sentences = []
            print("\nEnter sentences (type 'done' to finish):")

            while True:
                s = input(f"Sentence {len(sentences)+1}: ")
                if s.lower() == "done":
                    break
                if s.strip():
                    sentences.append(s)

            if not sentences:
                print("No sentences entered!")
                continue

            ranked = rank_sentences(ranker, query, sentences)

            print("\nRanking Results:\n")
            for i, (sent, score) in enumerate(ranked, 1):
                print(f"{i}. {sent}")
                print(f"   Score: {round(score, 4)}")

        elif choice == "2":
            print("Exiting...")
            break

        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()