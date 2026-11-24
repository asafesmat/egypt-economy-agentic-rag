import streamlit as st

from agent_backend import run_multilingual_agent


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Egypt Economy Analyst",
    page_icon="🇪🇬",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# =========================
# STYLE (calm, warm, chat-first)
# =========================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Noto+Sans+Arabic:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stMarkdown, .stChatInput textarea {
        font-family: 'Inter', 'Noto Sans Arabic', sans-serif;
    }

    footer, #MainMenu, .stDeployButton {
        visibility: hidden;
        display: none;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 760px;
        padding-top: 2.5rem;
        padding-bottom: 6rem;
    }

    /* Welcome screen */
    .welcome {
        text-align: center;
        margin: 9vh 0 2.2rem 0;
    }

    .welcome-title {
        font-size: 2rem;
        font-weight: 600;
        color: #1F1E1D;
        letter-spacing: -0.02em;
    }

    .welcome-sub {
        color: #73726C;
        font-size: 1rem;
        margin-top: 0.6rem;
    }

    /* Messages: no boxes, no avatars */
    div[data-testid="stChatMessage"] {
        background: transparent;
        border: none;
        padding: 0.4rem 0;
    }

    div[data-testid="stChatMessageAvatarUser"],
    div[data-testid="stChatMessageAvatarAssistant"] {
        display: none;
    }

    /* User messages as a soft bubble */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
        background: #F0EEE6;
        border-radius: 18px;
        padding: 0.7rem 1.1rem;
        margin: 0.8rem 0;
    }

    /* Each paragraph picks its own direction (Arabic -> right-to-left) */
    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] li {
        unicode-bidi: plaintext;
        text-align: start;
        line-height: 1.7;
    }

    /* Input */
    div[data-testid="stChatInput"] {
        border-radius: 24px;
    }

    /* Suggestion buttons */
    .stButton button {
        border-radius: 14px;
        border: 1px solid #E2DFD3;
        background: #FFFFFF;
        color: #3D3D3A;
        padding: 0.7rem 1rem;
        text-align: start;
        font-weight: 400;
    }

    .stButton button:hover {
        border-color: #C96442;
        color: #C96442;
        background: #FFFFFF;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        border: 1px solid #E8E5DA;
        border-radius: 12px;
        background: transparent;
    }

    .meta {
        color: #8A8880;
        font-size: 0.78rem;
        margin-top: 0.4rem;
    }

    .disclaimer {
        text-align: center;
        color: #9A978C;
        font-size: 0.75rem;
        margin-top: 1.2rem;
    }
    
    /* ===== Navy + teal palette ===== */
section[data-testid="stSidebar"] {
    background: #0B121A;
    border-right: 1px solid #1C2733;
}

.welcome-title { color: #F1F5F9; }
.welcome-sub   { color: #9AA8B5; }

div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
    background: #172230;
    border: 1px solid #243344;
}

.stButton button {
    background: #16212C;
    border: 1px solid #243344;
    color: #E2E8F0;
    min-height: 4.2rem;          /* كل الكروت بنفس الارتفاع */
}
.stButton button p {
    white-space: normal;
    unicode-bidi: plaintext;
    text-align: start;
}
.stButton button:hover {
    background: #1B2A38;
    border-color: #3FB8AF;
    color: #7FE0D8;
}

div[data-testid="stExpander"] {
    border: 1px solid #243344;
    background: #101923;
}

div[data-testid="stChatInput"] {
    background: #16212C;
    border: 1px solid #243344;
}
div[data-testid="stChatInput"]:focus-within { border-color: #3FB8AF; }

.meta       { color: #9AA8B5; }
.disclaimer { color: #7C8A98; }

section[data-testid="stSidebar"] .stButton button {
    min-height: 3rem;
    padding: 0.4rem 1rem;
}

</style>
""", unsafe_allow_html=True)


# =========================
# HELPERS
# =========================

VERIFIED_LABELS = {
    True: "Passed",
    False: "Failed (answer was revised)",
    None: "Not verified",
}


def render_details(details):
    """Confidence line, sources, English answer and execution details."""

    conf = details.get("confidence") or {}

    st.markdown(
        f'<div class="meta">Confidence: {conf.get("label", "n/a")} '
        f'({conf.get("score")}) · {details.get("model", "")} · '
        f'{details.get("tokens")} tokens · {details.get("seconds")}s</div>',
        unsafe_allow_html=True
    )

    sources = details.get("sources") or {}

    if sources:
        with st.expander("Sources"):
            for tag, s in sources.items():
                st.write(f"**[{tag}]** {s.get('title')} — {s.get('location')}")

    if details.get("language") == "ar" and details.get("answer_en"):
        with st.expander("English answer"):
            st.markdown(details["answer_en"])

    with st.expander("How the agent got this answer"):

        if details.get("question_en") and details.get("language") == "ar":
            st.write("**Translated question**")
            st.write(details["question_en"])

        st.write("**Plan**")
        st.write(details.get("plan", "Not available"))

        st.write("**Steps**")
        steps = details.get("steps", [])

        if steps:
            for index, step in enumerate(steps, 1):
                st.write(f"Step {index}: `{step.get('tool')}`")
                st.json(step.get("args", {}), expanded=False)
        else:
            st.info("No tools were used.")

        st.write("**Verification:** " + VERIFIED_LABELS.get(details.get("verified"), "Not available"))

        if details.get("guard_problems"):
            st.write("**Guard findings**")
            st.write(details["guard_problems"])

        st.write(
            f"**Performance:** {details.get('llm_calls')} LLM calls · "
            f"{details.get('tokens')} tokens · {details.get('seconds')}s · "
            f"{details.get('guard_rounds')} guard round(s)"
        )


# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.markdown("### Egypt Economy Analyst")

    if st.button("＋ New chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    max_steps = st.slider("Agent steps", min_value=2, max_value=10, value=6)

    verify = st.toggle("Verify answers", value=True)

    st.divider()

    st.caption(
        "Sources: World Bank Egypt Economic Monitor (Dec 2021, Dec 2022), "
        "IMF Country Report 25/186 (Jul 2025), and World Bank WDI indicators."
    )

    st.caption(
        "The World Bank reports end in 2022. Arabic answers are translations "
        "of the English answer."
    )


# =========================
# CHAT
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

pending = st.session_state.pop("pending_question", None)
prompt = st.chat_input("Ask about Egypt's economy in English or Arabic...") or pending

# Welcome screen with suggestions
if not st.session_state.messages and not prompt:

    st.markdown("""
    <div class="welcome">
        <div class="welcome-title">Egypt Economy Analyst</div>
        <div class="welcome-sub">
            Ask about inflation, growth, unemployment and policy,
            grounded in World Bank and IMF data.
        </div>
    </div>
    """, unsafe_allow_html=True)

    suggestions = [
        "How did Egypt's inflation change between 2020 and 2023?",
        "Compare Egypt and Jordan unemployment between 2015 and 2022.",
        "What were the main causes of inflation in Egypt in 2022?",
        "ما هي توصيات صندوق النقد الدولي بشأن سعر الصرف في مصر؟"
    ]

    left, right = st.columns(2)

    for i, question in enumerate(suggestions):

        with (left if i % 2 == 0 else right):

            if st.button(question, key=f"sug_{i}", width="stretch"):
                st.session_state.pending_question = question
                st.rerun()

# Conversation history
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("details"):
            render_details(message["details"])

# New question
if prompt:

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:
                result = run_multilingual_agent(
                    prompt,
                    max_steps=max_steps,
                    verify=verify
                )

                answer = result.get("answer", "The agent returned no answer.")

                st.markdown(answer)
                render_details(result)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "details": result
                })

            except RuntimeError as e:

                error_message = (
                    "The Groq rate limit was reached. "
                    "Please wait a few minutes and try again."
                )

                st.warning(error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })

            except Exception as e:

                error_message = f"Something went wrong: {type(e).__name__}: {str(e)[:200]}"

                st.error(error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })

st.markdown(
    '<div class="disclaimer">AI-generated analysis. Check the cited reports for official figures. '
    'Each question uses about 5,000–15,000 tokens of the free Groq quota.</div>',
    unsafe_allow_html=True
)
