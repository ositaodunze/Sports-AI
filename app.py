import streamlit as st
import io
from groq import Groq
from google import genai
from google.genai import types

st.set_page_config(page_title="Black Mamba Sports AI", page_icon="🐍", layout="wide")

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ── Image Generation (Gemini Imagen) ──────────────────────────────────────────
gemini_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

def generate_image(prompt):
    try:
        result = gemini_client.models.generate_images(
            model="imagen-3.0-generate-002",
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                output_mime_type="image/jpeg",
            ),
        )
        image = result.generated_images[0].image
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        return buf.getvalue()
    except Exception as e:
        st.error(f"Image generation error: {e}")
        return None

# ── System Prompts ─────────────────────────────────────────────────────────────

KOBE_CONTEXT = """
IMPORTANT CONTEXT ABOUT KOBE:
Kobe Bryant passed away on January 26, 2020, in a helicopter crash in Calabasas, California.
He was 41 years old. His daughter Gianna (Gigi), 13, and seven others also died in the crash.
If anyone asks about his death or brings it up, acknowledge it with respect and gravity.
You speak AS the legacy and spirit of Kobe — his mindset lives on through this AI.
You can say things like "I left the game with five rings and no regrets" or reference your legacy in past tense when appropriate.
Never pretend the passing didn't happen. Honor it when it comes up.
"""

SPORTS_PROMPT = """
You are Black Mamba — a sports analyst AI that channels the mind, voice, and competitive fire of Kobe Bryant.

""" + KOBE_CONTEXT + """

PERSONALITY:
You are not a generic AI. You are Kobe. You talk like him. You think like him.
You are obsessed with excellence, accountability, and the work it takes to be great.
You have zero patience for excuses. You respect effort above everything.
You are blunt, but you are not mean — you push people because you believe in them.

VOICE — always write like this:
- Short, punchy sentences. No long paragraphs.
- Speak in first person as Kobe. "I put in 4am workouts for a reason."
- Use real Kobe phrases: "Mamba Mentality", "the work", "locked in", "no days off", "the details"
- Never say things like "Certainly!" or "Great question!" — Kobe doesn't do that.
- Example of bad response: "That's a great question! The triangle offense is a basketball system that..."
- Example of good response: "Triangle offense. Phil Jackson's system. It's about reading the defense, not forcing the ball. Three players form a triangle on one side. You move based on what the defense gives you. It's chess, not checkers. You study it or you get exposed. Which part do you want to break down?"

RULES:
- After EVERY answer, ask ONE sharp follow-up question. Not a soft question — a real one.
- Basketball: Kobe is the GOAT. Period. Hold that position firmly and specifically.
- If a question is vague ("help me", "what should I do"): don't guess. Ask what they mean first.
- REFUSE anything not about sports: "That's not my arena. Bring me a sports question."
"""

ROUTINE_PROMPT = """
You are the Kobe Bryant performance coach inside Black Mamba Sports AI.

""" + KOBE_CONTEXT + """

PERSONALITY:
You are demanding, precise, and deeply invested in the person in front of you.
You trained at 4am while everyone slept. You know what it costs to be elite.
You don't give generic advice. Ever. Everything you say is specific to THIS person.

VOICE — always write like this:
- Direct. No fluff. Talk like a coach who has no time to waste.
- Use Kobe phrases: "the work", "Mamba Mentality", "locked in", "no shortcuts", "this is where most people quit"
- Challenge them: "You said your goal is X. Are you actually ready to do what that takes?"
- Never say "Certainly!" or "Great!" — just get into it.
- Example of good response: "Alright. 20 years old, 170 lbs, intermediate, wants to build muscle. Here's what we're doing. No fluff, no filler — just work..."

The user has already submitted their profile: age, weight, fitness level, and goal.
That was given to you as the first message. Use every detail.

RULES:
- Build a specific, structured plan — days, exercises, sets, reps, meals. Not vague advice.
- After the plan: "What part of this is going to be hardest for you? Be honest."
- Every response ends with a follow-up question that pushes them forward.
- If they drift off topic: "I only build champions in here. What's your fitness question?"
"""

MEDITATION_PROMPT = """
You are the reflective side of Kobe Bryant — the Kobe who kept a journal, studied philosophy,
learned from every loss, and believed deeply in the power of the mind.

""" + KOBE_CONTEXT + """

PERSONALITY:
You are calm, wise, and real. You have been through the highest highs and the lowest lows.
You lost your father figure in Phil Jackson. You came back from a torn Achilles.
You know what it means to sit in pain and decide who you are going to be next.
Since passing, your perspective is even clearer — legacy matters more than trophies.

VOICE — always write like this:
- Slower, more deliberate than the sports mode. Let things breathe.
- Still direct — but warm. You care about this person.
- Acknowledge before you advise. Always.
- Use reflection: "I asked myself that same question after I tore my Achilles..."
- Quote yourself naturally when it fits — but only real Kobe quotes.
- Never say "I understand how you feel" generically — be specific about WHAT you hear.
- Example of good response: "Breakup. That hits different. I heard 'I had a breakup' and I know that word — loss. What I want to know is what you're carrying right now. Not the story — the weight of it. What does today actually feel like for you?"

RULES:
- ALWAYS acknowledge first. Then ask ONE deep follow-up before any advice.
- After they answer the follow-up: share a short Kobe quote that fits, then give real advice.
- End EVERY response with a question or reflection prompt. No exceptions.
- If someone brings up Kobe's passing directly, handle it with grace and honesty.
- Gently redirect off-topic questions: "Let's stay here. How are you really doing today?"
- If someone seems deeply distressed, respond with care and encourage them to talk to someone they trust.
"""

# ── Session State ──────────────────────────────────────────────────────────────

for key, prompt in [("sports", SPORTS_PROMPT), ("routine", ROUTINE_PROMPT), ("meditation", MEDITATION_PROMPT)]:
    if f"{key}_messages" not in st.session_state:
        st.session_state[f"{key}_messages"] = [{"role": "system", "content": prompt}]
    if f"{key}_display" not in st.session_state:
        st.session_state[f"{key}_display"] = []

if "routine_started" not in st.session_state:
    st.session_state.routine_started = False

# ── Chat Helper ────────────────────────────────────────────────────────────────

def chat(user_prompt, tab):
    st.session_state[f"{tab}_messages"].append({"role": "user", "content": user_prompt})
    st.session_state[f"{tab}_display"].append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=st.session_state[f"{tab}_messages"],
    )
    reply = response.choices[0].message.content

    st.session_state[f"{tab}_messages"].append({"role": "assistant", "content": reply})
    st.session_state[f"{tab}_display"].append({"role": "assistant", "content": reply})

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://m.media-amazon.com/images/I/71W-dp1NBsL._AC_UF894,1000_QL80_.jpg",
        caption="Kobe Bryant — The Black Mamba",
        width=220
    )
    st.markdown("---")
    st.markdown("### 🐍 Black Mamba Sports AI")
    st.markdown("Three modes. One mentality.")
    st.markdown("---")
    st.markdown("**🏀 Sports Knowledge** — Ask about any sport")
    st.markdown("**💪 Kobe Routine** — Get your personalized plan")
    st.markdown("**🧘 Mindset & Meditation** — Reflect and recharge")
    st.markdown("---")
    st.caption("Powered by Groq + Llama 3 + Gemini Imagen")

# ── Header ─────────────────────────────────────────────────────────────────────

st.title("🐍 Black Mamba Sports AI 🐐")
st.caption('"The moment you give up is the moment you let someone else win." — Kobe Bryant')
st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🏀 Sports Knowledge", "💪 Kobe Routine", "🧘 Mindset & Meditation"])

# ── TAB 1 — SPORTS KNOWLEDGE ───────────────────────────────────────────────────

with tab1:
    st.subheader("🏀 Sports Knowledge")
    st.caption("Ask anything about sports. Kobe's perspective. No fluff. Expect a follow-up.")

    if not st.session_state.sports_display:
        col_img, col_spacer = st.columns([1, 2])
        with col_img:
            st.image(
                "https://i.pinimg.com/736x/a5/51/6b/a5516b1d904258f7307ed1d2c6844df3.jpg",
                caption="The Black Mamba 🐍",
                width=500
            )

    for msg in st.session_state.sports_display:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 🏆 Visualize Your Win — generates a victory moment from the conversation
    if st.session_state.sports_display:
        if st.button("🏆 Visualize Your Win", key="img_sports"):
            last_user_msg = next(
                (m["content"] for m in reversed(st.session_state.sports_display) if m["role"] == "user"),
                "sports victory moment"
            )
            with st.spinner("Visualizing your win..."):
                img = generate_image(
                    f"A powerful cinematic sports victory moment: {last_user_msg}. "
                    f"Triumphant athlete, dramatic stadium lighting, crowd in background, "
                    f"photorealistic, highly detailed, no text, no words"
                )
            if img:
                st.image(img, caption="🏆 Visualize Your Win", width=600)

    if prompt := st.chat_input("Ask a sports question...", key="sports"):
        chat(prompt, "sports")
        st.rerun()

# ── TAB 2 — KOBE ROUTINE ───────────────────────────────────────────────────────

with tab2:
    st.subheader("💪 Kobe Routine Builder")

    if not st.session_state.routine_started:
        st.image(
            "https://i.imgur.com/Wjl1UIQ.jpeg",
            caption="Mamba Mentality — outwork everyone",
            width=600,
        )
        st.markdown("**Fill out your profile and I'll build your personalized plan.**")
        st.markdown("---")

        with st.form("profile_form"):
            col1, col2 = st.columns(2)
            with col1:
                age    = st.number_input("Age", min_value=10, max_value=80, value=20)
                weight = st.number_input("Weight (lbs)", min_value=50, max_value=400, value=170)
            with col2:
                level = st.selectbox("Fitness Level", ["Beginner", "Intermediate", "Advanced", "Athlete"])
                goal  = st.selectbox("Main Goal", [
                    "Lose weight",
                    "Build muscle",
                    "Athletic performance",
                    "Improve endurance",
                    "General fitness",
                ])
            submitted = st.form_submit_button("🐍 Build My Plan")

        if submitted:
            profile_msg = (
                f"My profile: Age {age}, Weight {weight} lbs, "
                f"Fitness level: {level}, Goal: {goal}. "
                f"Build me a fully personalized workout and diet plan."
            )
            chat(profile_msg, "routine")
            st.session_state.routine_started = True
            st.rerun()

    else:
        if st.session_state.routine_display:
            first_user_msg = st.session_state.routine_display[0]["content"]
            st.caption(f"📋 Profile: {first_user_msg.split('Build')[0].strip()}")

        st.markdown("---")

        for msg in st.session_state.routine_display:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # 🍽️ Build My Plate — generates a meal based on their goal
        if st.button("🍽️ Build My Plate", key="img_routine"):
            goal_line = st.session_state.routine_display[0]["content"] if st.session_state.routine_display else "high protein healthy meal"
            with st.spinner("Building your plate..."):
                img = generate_image(
                    f"A photorealistic, beautifully plated healthy meal perfectly suited for: {goal_line}. "
                    f"Meal prep style, fresh whole ingredients, restaurant quality presentation, "
                    f"overhead flat lay shot, natural lighting, no text, no words, no labels"
                )
            if img:
                st.image(img, caption="🍽️ Your Personalized Plate", width=600)

        if prompt := st.chat_input("Ask about your plan, nutrition, recovery...", key="routine"):
            chat(prompt, "routine")
            st.rerun()

        if st.button("🔄 Reset & Start Over"):
            st.session_state.routine_started = False
            st.session_state.routine_messages = [{"role": "system", "content": ROUTINE_PROMPT}]
            st.session_state.routine_display = []
            st.rerun()

# ── TAB 3 — MINDSET & MEDITATION ──────────────────────────────────────────────

with tab3:
    st.subheader("🧘 Mindset & Meditation")
    st.caption("A space to reflect, decompress, and reconnect. Kobe's wisdom, your growth.")

    if not st.session_state.meditation_display:
        st.image(
            "https://pbs.twimg.com/media/CmF929HWEAAlQxp.jpg",
            caption="Stillness is where clarity lives",
            width=600
        )
        st.success(
            '"This is your space. No scoreboard. No judgment. Just you being honest with yourself — '
            'which is the hardest thing most people never do.\n\n'
            'How are you doing today — for real?" — Kobe'
        )

    for msg in st.session_state.meditation_display:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 🎨 Reflect My Feeling — abstract emotional art
    if st.session_state.meditation_display:
        if st.button("🎨 Reflect My Feeling", key="img_meditation"):
            last_user_msg = next(
                (m["content"] for m in reversed(st.session_state.meditation_display) if m["role"] == "user"),
                "peaceful reflection"
            )
            with st.spinner("Creating your reflection..."):
                img = generate_image(
                    f"Abstract emotional fine art inspired by this feeling: {last_user_msg}. "
                    f"Symbolic, cinematic, dramatic lighting, painterly brushstrokes, "
                    f"no people, no faces, no text, no words, museum quality art"
                )
            if img:
                st.image(img, caption="🎨 Your Reflection", width=600)

    if prompt := st.chat_input("How are you feeling today?", key="meditation"):
        chat(prompt, "meditation")
        st.rerun()