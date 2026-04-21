
from groq import Groq


client = Groq(api_key=st.secrets["GROQ_API_KEY"])
messages = [
    {
        "role": "system",
        "content": (
            "You are Sports Analyst AI. "
            "Explain sports rules, players, and strategies simply. "
            "Keep answers short and clear."
        )
    }
]

print("Welcome to Sports Analyst AI!")
print("Ask me a sports question. Type 'quit' to stop.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        print("Chatbot: Goodbye!")
        break

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )

    reply = response.choices[0].message.content

    messages.append({"role": "assistant", "content": reply})

    print(f"Chatbot: {reply}\n")