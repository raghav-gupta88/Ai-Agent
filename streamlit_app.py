from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

# Streamlit page config
st.set_page_config(page_title="Content Researcher & Writer", layout="wide")

# Title and description
st.title("✍️ Content Researcher & Writer")
st.markdown("Generate blog posts about any topic using AI agents powered by CrewAI.")

# Sidebar
with st.sidebar:
    st.header("Content Settings")

topic = st.text_area("Enter your topic", height=100, placeholder="Enter the topic")

# LLM Settings
st.markdown("### LLM Settings")
temperature = st.slider("Temperature", 0.0, 1.0, 0.7)

# Generate button
generate_button = st.button("Generate Content", type="primary", use_container_width=True)

# How-to
with st.expander("ℹ️ How to use"):
    st.markdown("""
    1. Enter your topic  
    2. Adjust temperature  
    3. Click **Generate Content**  
    4. Download your blog post in Markdown  
    """)

# Generation function
def generate_content(topic):
    llm = LLM(
        model="openrouter/mistralai/mistral-7b-instruct",  # ✅ Use a model from OpenRouter
        api_key=os.getenv("OPENROUTER_API_KEY")            # ✅ Make sure .env is loaded
    )

    search_tool = SerperDevTool(n_results=10)

    # Research Agent
    senior_research_analyst = Agent(
        role="Senior Research Analyst",
        goal=f"Research and analyze detailed information on '{topic}' using web sources.",
        backstory=(
            "You're a skilled research analyst who finds, verifies, and summarizes "
            "information from the internet for blogs and content teams."
        ),
        allow_delegation=False,
        verbose=True,
        llm=llm
    )

    # Writer Agent
    content_writer = Agent(
        role="Content Writer",
        goal="Transform research into a well-written, engaging, and factual blog post.",
        backstory=(
            "You're a professional writer who excels at turning dense research into "
            "engaging, well-structured, and easy-to-read blog posts."
        ),
        allow_delegation=False,
        verbose=True,
        llm=llm
    )

    # Research Task
    research_task = Task(
        description=f"""
        1. Research '{topic}' thoroughly using current web sources.
        2. Identify key insights, trends, and expert views.
        3. Verify all facts, stats, and cite original sources.
        """,
        expected_output="""
        A research summary with:
        - Executive Summary
        - Key Findings
        - Relevant Stats and Trends
        - All citations and links to sources
        """,
        agent=senior_research_analyst
    )

    # Writing Task
    writing_task = Task(
        description="""
        Based on the research, create a detailed blog post with:
        1. Hooking intro
        2. Structured body with H3s
        3. Strong conclusion
        4. Markdown formatting
        5. Inline [Source: URL] citations
        """,
        expected_output="""
        A complete blog post in Markdown format with structure, citations,
        and readable flow for an online audience.
        """,
        agent=content_writer
    )

    # Crew setup
    crew = Crew(
        agents=[senior_research_analyst, content_writer],
        tasks=[research_task, writing_task],
        verbose=True
    )

    return crew.kickoff(inputs={"topic": topic})


# Main execution
if generate_button:
    if not topic:
        st.warning("Please enter a topic first.")
    else:
        with st.spinner("Generating content..."):
            try:
                result = generate_content(topic)
                st.markdown("### 📝 Generated Content")
                st.markdown(result)

                st.download_button(
                    label="📥 Download Markdown",
                    data=result.raw,
                    file_name=f"{topic.lower().replace(' ', '_')}_blog.md",
                    mime="text/markdown"
                )
            except Exception as e:
                st.error(f"An error occurred:\n\n{str(e)}")

# Footer
st.markdown("---")
st.markdown("Built using CrewAI, Streamlit, and OpenRouter")
