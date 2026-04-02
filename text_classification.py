
from transformers import pipeline

def load_model():
    print("Loading classification model...")
    return pipeline("sentiment-analysis")

def classify_text(classifier, text):
    result = classifier(text)[0]
    return result['label'], result['score']

def main():
    classifier = load_model()
    print("Model loaded!\n")

    while True:
        print("\nOPTIONS:")
        print("1. Classify Text")
        print("2. Exit")

        choice = input("Enter choice: ")

        if choice == "1":
            text = input("\nEnter text: ").strip()

            if not text:
                print("Empty input!")
                continue

            label, score = classify_text(classifier, text)

            print("\nResult:")
            print(f"Sentiment : {label}")
            print(f"Confidence: {round(score, 4)}")

        elif choice == "2":
            print("Exiting...")
            break

        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()