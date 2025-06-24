import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------- Load Model (runs on CPU) -------------------
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# -------- Trending Professions -----------------------
def get_trending_professions():
    return {
        "AI Engineer": "Designs and builds artificial intelligence and machine learning systems.",
        "Prompt Engineer": "Optimizes prompts to get the best responses from language models.",
        "Cybersecurity Analyst": "Monitors systems for cyber threats and defends against attacks.",
        "Sustainability Analyst": "Helps businesses reduce their environmental impact using data.",
        "AR/VR Developer": "Creates immersive applications using augmented or virtual reality.",
        "Bioinformatics Scientist": "Analyzes biological data to understand genetics and diseases.",
        "GenAI Product Manager": "Leads teams building products using generative AI models.",
        "Data Privacy Officer": "Ensures organizations comply with data protection regulations."
    }

# -------- Generate Answer using Local Model ----------
def generate_answer(profession, question):
    description = get_trending_professions().get(profession, "")
    prompt = f"""You are a career advisor.
Profession: {profession}
Description: {description}
Question: {question}
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        output = model.generate(
            inputs.input_ids,
            max_length=256,
            temperature=0.7,
            do_sample=True
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.split("Answer:")[-1].strip()

# -------- Command Line Interface ---------------------
def main():
    print("=" * 60)
    print("🤖 CareerBot: Explore Trending Professions (Offline Mode)")
    print("=" * 60)

    professions = get_trending_professions()
    keys = list(professions.keys())

    for idx, job in enumerate(keys, 1):
        print(f"{idx}. {job} - {professions[job]}")

    try:
        choice = int(input("\nPick a profession by number: "))
        profession = keys[choice - 1]
    except (ValueError, IndexError):
        print("❌ Invalid input. Exiting.")
        return

    print(f"\nYou picked: {profession}")
    print("Ask anything about this career (type 'exit' to quit).\n")

    while True:
        query = input("🧠 Ask: ")
        if query.lower().strip() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        answer = generate_answer(profession, query)
        print(f"\n💡 {answer}\n")

if __name__ == "__main__":
    main()
