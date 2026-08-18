import os
import re
import streamlit as st
from groq import Groq
from tavily import TavilyClient

# --- Clients ---
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

MODEL = "llama-3.3-70b-versatile"
MAX_ROUNDS = 4

# --- Search ---
def search(query: str) -> str:
    results = tavily_client.search(
        query=query,
        search_depth="advanced",
        max_results=5,
        include_raw_content=False
    )
    formatted = []
    for r in results.get("results", []):
        formatted.append(f"Source: {r['url']}\nTitle: {r['title']}\nContent: {r['content']}\n")
    return "\n".join(formatted)

# --- Generator ---
def generator(question: str, feedback: str = "") -> tuple[str, str]:
    search_query = question
    if feedback:
        search_query = f"{question} {feedback}"
    
    search_results = search(search_query)
    
    feedback_section = f"\n\nPrevious evaluation feedback — address these gaps specifically:\n{feedback}" if feedback else ""
    
    prompt = f"""You are a research assistant producing answers for a professional newsletter.

Your job: answer the research question below using the search results provided.

Rules for sources:
- Only cite named publications, research institutions, company blogs, or official reports
- Never cite SEO content farms, listicles, or promotional pages
- If a source doesn't clearly identify its author or institution, don't cite it

Rules for the answer:
- Cover all major angles of the question — don't stop at the first sufficient-looking answer
- Name specific sources for every claim
- Flag explicitly if two sources contradict each other
- Write in clear, direct prose{feedback_section}

Research question: {question}

Search results:
{search_results}

Answer:"""

    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0
    )
    
    answer = response.choices[0].message.content
    return answer, search_results

# --- Evaluator ---
def evaluator(question: str, answer: str) -> tuple[str, str]:
    prompt = f"""You are a strict research editor. Your only job is to check whether this answer is complete.

Research question: {question}

Answer to evaluate:
{answer}

Check for these specific gaps:
1. Are there major angles of this question that weren't covered?
2. Are there claims made without a named source — publication, institution, company blog, or official report?
3. Is there a specific number or data point for every key claim, or are claims stated as general facts without evidence?
4. If two sources disagree, has that contradiction been named explicitly — or was it averaged into one clean sentence?
5. Is there a named company, market, or India-specific context where relevant — or is the answer too generic to be useful for an Indian business audience?
6. Does the answer explain the business or market implication of each finding — not just the fact itself?
7. Would a domain expert read this and immediately ask "but what about X?"

If the answer passes all seven checks, respond with exactly:
VERDICT: PASS

If any check fails, respond with exactly:
VERDICT: FAIL
GAPS:
- [specific gap 1]
- [specific gap 2]

Be strict. If in doubt, FAIL."""

    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0
    )
    
    evaluation = response.choices[0].message.content
    
    if "VERDICT: PASS" in evaluation:
        return "PASS", ""
    else:
        gaps_match = re.search(r"GAPS:(.*)", evaluation, re.DOTALL)
        gaps = gaps_match.group(1).strip() if gaps_match else "Gaps not specified"
        return "FAIL", gaps

# --- Loop ---
def research_loop(question: str, status_container, output_container):
    feedback = ""
    final_answer = ""
    
    for round_num in range(1, MAX_ROUNDS + 1):
        status_container.markdown(f"**Round {round_num}** — Searching...")
        
        answer, sources = generator(question, feedback)
        
        status_container.markdown(f"**Round {round_num}** — Evaluating completeness...")
        verdict, gaps = evaluator(question, answer)
        
        if verdict == "PASS":
            status_container.markdown(f"**Round {round_num}** — ✅ Complete")
            final_answer = answer
            break
        else:
            status_container.markdown(f"**Round {round_num}** — ❌ Gaps found:\n{gaps}")
            feedback = gaps
            final_answer = answer
            
            if round_num == MAX_ROUNDS:
                status_container.markdown(f"**Round {round_num}** — ⚠️ Max rounds reached. Returning best answer.")
    
    output_container.markdown("### Research Complete")
    output_container.markdown(final_answer)

# --- Streamlit UI ---
st.set_page_config(page_title="Research Loop", page_icon="🔍", layout="wide")

st.title("🔍 Research Loop")
st.markdown("Ask a broad research question. The loop keeps searching until the answer is complete.")

question = st.text_area(
    "Your research question",
    placeholder="e.g. What's actually happening in the Indian D2C funding space right now?",
    height=100
)

if st.button("Run Research Loop", type="primary"):
    if not question.strip():
        st.error("Please enter a research question.")
    else:
        st.markdown("---")
        st.markdown("**Loop Progress**")
        status_container = st.empty()
        st.markdown("---")
        output_container = st.container()
        
        with st.spinner("Running research loop..."):
            research_loop(question, status_container, output_container)
