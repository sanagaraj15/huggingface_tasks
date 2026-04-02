
from transformers import pipeline

def load_model():
    print("Loading text generation model...")
    return pipeline("text-generation", model="gpt2")

def generate_text(generator, prompt):
    result = generator(prompt, max_length=50, num_return_sequences=1)
    return result[0]['generated_text']

def main():
    generator = load_model()
    print("Model loaded!\n")

    while True:
        print("\nOPTIONS:")
        print("1. Generate Text")
        print("2. Exit")

        choice = input("Enter choice: ")

        if choice == "1":
            prompt = input("\nEnter starting text: ").strip()

            if not prompt:
                print("Empty input!")
                continue

            output = generate_text(generator, prompt)

            print("\nGenerated Text:")
            print(output)

        elif choice == "2":
            print("Exiting...")
            break

        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()