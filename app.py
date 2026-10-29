
import os
import json
import re

import streamlit as st
from pypdf import PdfReader
from google import genai
from google.genai import types


# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Resume ATS Checker",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_NAME = "gemini-2.5-flash"


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fc;
    }
    .hero {
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(120deg, #102a43, #2563eb);
        color: white;
        margin-bottom: 22px;
    }
    .hero h1 {
        color: white;
        margin-bottom: 8px;
    }
    .hero p {
        color: #e5efff;
        font-size: 16px;
    }
    .small-note {
        color: #64748b;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def get_api_key():
    """Read the API key from Streamlit secrets or environment."""
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""

    return key or os.getenv("GEMINI_API_KEY", "")


def extract_pdf_text(uploaded_file):
    """Extract readable text from an uploaded PDF."""
    reader = PdfReader(uploaded_file)
    pages = []

    for page in reader.pages:
        pages.append(page.extract_text() or "")

    text = "\n".join(pages).strip()

    if not text:
        raise ValueError(
            "No readable text was found. This may be a scanned PDF. "
            "Please upload a text-based PDF."
        )

    # Keep the request within reasonable size limits.
    return text[:40000]


def analyze_resume(resume_text, job_description, api_key):
    """Ask Gemini to evaluate the resume against the job description."""
    client = genai.Client(api_key=api_key)

    prompt = f"""
You are an expert resume reviewer and ATS-style job matching assistant.

Compare the candidate's resume with the supplied job description.
Use evidence from the supplied text. Never invent candidate experience,
skills, qualifications, or certifications.

Return ONLY a valid JSON object with this exact structure:

{{
  "match_score": 0,
  "summary": "A concise overall assessment",
  "matched_skills": [
    {{"skill": "Skill name", "evidence": "Evidence from the resume"}}
  ],
  "missing_skills": [
    {{"skill": "Skill name", "importance": "High, Medium, or Low"}}
  ],
  "matched_keywords": ["keyword1", "keyword2"],
  "missing_keywords": ["keyword1", "keyword2"],
  "strengths": ["strength1", "strength2"],
  "improvements": [
    {{
      "priority": "High",
      "issue": "What needs improvement",
      "suggestion": "Specific, honest action the candidate can take"
    }}
  ],
  "ats_tips": ["practical tip1", "practical tip2"]
}}

Rules:
- match_score must be an integer from 0 to 100.
- Estimate relevance across required skills, relevant experience,
  qualifications, responsibilities, and important keywords.
- Do not reward a keyword merely for appearing if context shows no
  relevant experience or understanding.
- Distinguish explicitly required skills from preferred skills.
- Do not assume a missing skill is present.
- Give actionable recommendations specific to this job description.
- Do not recommend adding skills or experience the candidate does
  not actually possess.
- If the resume is weakly aligned, give a correspondingly low score.
- Do not claim this is a real ATS vendor's score or an exact hiring
  probability.
- Keep each list concise and useful.
- Return valid JSON without Markdown fences.

JOB DESCRIPTION:
{job_description[:20000]}

RESUME TEXT:
{resume_text}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )

    if not response.text:
        raise ValueError("The AI returned an empty response. Please retry.")

    result_text = response.text.strip()

    # Defensive handling in case the model includes code fences.
    result_text = re.sub(r"^```(?:json)?\s*", "", result_text)
    result_text = re.sub(r"\s*```$", "", result_text)

    result = json.loads(result_text)

    # Validate and normalize the score before displaying it.
    score = int(result.get("match_score", 0))
    result["match_score"] = max(0, min(100, score))

    return result


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>📄 Resume ATS Checker</h1>
        <p>
            Compare your CV with a job description, identify missing
            skills, and get AI-powered suggestions to improve your resume.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("Resume ATS Checker")
    st.caption("AI-powered resume analysis")
    st.divider()

    st.markdown("### How it works")
    st.markdown(
        """
        1. Upload your resume as a PDF.
        2. Paste the target job description.
        3. Click **Analyze Resume**.
        4. Review your match score and improvements.
        """
    )

    st.divider()
    st.caption("Powered by Google Gemini and Streamlit.")
    st.caption(
        "Your resume text is sent to the configured AI provider "
        "for analysis. Avoid uploading sensitive information."
    )


# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

left, right = st.columns([1, 1.15], gap="large")

with left:
    st.subheader("1. Upload your CV")

    uploaded_file = st.file_uploader(
        "Choose your resume PDF",
        type=["pdf"],
        help="Upload a text-based PDF. Maximum size: 5 MB.",
    )

    if uploaded_file is not None:
        if uploaded_file.size > 5 * 1024 * 1024:
            st.error("Your PDF is larger than 5 MB. Please upload a smaller file.")
            uploaded_file = None
        else:
            st.success(f"Uploaded: {uploaded_file.name}")

    st.info(
        "Scanned PDFs that contain only images may not work. "
        "Use a PDF with selectable text."
    )

with right:
    st.subheader("2. Add the job description")

    job_description = st.text_area(
        "Paste the complete job description",
        height=280,
        placeholder=(
            "Paste the job responsibilities, required skills, "
            "qualifications, and experience here..."
        ),
        help="Include the required and preferred qualifications for better results.",
    )

st.divider()

# --------------------------------------------------
# ANALYSIS ACTION
# --------------------------------------------------

analyze_clicked = st.button(
    "🔍 Analyze My Resume",
    type="primary",
    use_container_width=True,
)

if analyze_clicked:
    api_key = get_api_key()

    if not api_key:
        st.error(
            "Gemini API key not found. Add GEMINI_API_KEY to "
            "your Streamlit secrets or environment variables."
        )

    elif uploaded_file is None:
        st.warning("Please upload your resume PDF first.")

    elif not job_description.strip():
        st.warning("Please paste the job description first.")

    elif len(job_description.strip()) < 50:
        st.warning(
            "The job description is very short. Paste more details "
            "for a more useful comparison."
        )

    else:
        try:
            with st.spinner(
                "Reading your resume and comparing it with the job..."
            ):
                resume_text = extract_pdf_text(uploaded_file)

                result = analyze_resume(
                    resume_text=resume_text,
                    job_description=job_description.strip(),
                    api_key=api_key,
                )

            st.session_state["ats_result"] = result
            st.success("Resume analysis completed!")

        except json.JSONDecodeError:
            st.error(
                "The AI returned an unreadable result. Please try again."
            )

        except Exception as exc:
            st.error(
                "Analysis failed. Check your API key, internet connection, "
                "Gemini model availability, and API quota."
            )
            with st.expander("Technical details"):
                st.code(str(exc))


# --------------------------------------------------
# RESULTS SECTION
# --------------------------------------------------

result = st.session_state.get("ats_result")

if result:
    st.divider()
    st.header("📊 Your ATS Match Report")

    score = result.get("match_score", 0)

    score_col, summary_col = st.columns([1, 2], gap="large")

    with score_col:
        st.metric("Estimated Job Match", f"{score}%")
        st.progress(score / 100)

        if score >= 80:
            st.success("Strong alignment")
        elif score >= 60:
            st.info("Moderate alignment")
        else:
            st.warning("More improvement recommended")

    with summary_col:
        st.subheader("Overall assessment")
        st.write(result.get("summary", "No summary available."))

        st.caption(
            "This percentage is an AI-generated estimate of resume "
            "relevance, not an official ATS score or a hiring prediction."
        )

    # Skills comparison
    st.subheader("✅ Matched skills")

    matched_skills = result.get("matched_skills", [])

    if matched_skills:
        for item in matched_skills:
            if isinstance(item, dict):
                skill = item.get("skill", "Skill")
                evidence = item.get("evidence", "")
                with st.expander(f"✓ {skill}"):
                    st.write(evidence or "No evidence details supplied.")
            else:
                st.write(f"✓ {item}")
    else:
        st.write("No clear skill matches were identified.")

    st.subheader("⚠️ Missing or unverified skills")

    missing_skills = result.get("missing_skills", [])

    if missing_skills:
        for item in missing_skills:
            if isinstance(item, dict):
                skill = item.get("skill", "Skill")
                importance = item.get("importance", "Not specified")
                st.markdown(f"- **{skill}** — Importance: {importance}")
            else:
                st.markdown(f"- {item}")
    else:
        st.success("No major missing skills were identified.")

    # Keywords
    keyword_col1, keyword_col2 = st.columns(2)

    with keyword_col1:
        st.subheader("🔑 Matched keywords")
        keywords = result.get("matched_keywords", [])
        if keywords:
            st.write(", ".join(str(k) for k in keywords))
        else:
            st.write("No matched keywords listed.")

    with keyword_col2:
        st.subheader("🔎 Missing keywords")
        keywords = result.get("missing_keywords", [])
        if keywords:
            st.write(", ".join(str(k) for k in keywords))
        else:
            st.write("No missing keywords listed.")

    # Strengths
    st.subheader("💪 Resume strengths")

    strengths = result.get("strengths", [])

    if strengths:
        for item in strengths:
            st.markdown(f"- {item}")
    else:
        st.write("No strengths were returned.")

    # Improvements
    st.subheader("🛠️ Recommended improvements")

    improvements = result.get("improvements", [])

    if improvements:
        for index, item in enumerate(improvements, start=1):
            if isinstance(item, dict):
                priority = item.get("priority", "General")
                issue = item.get("issue", "Improvement opportunity")
                suggestion = item.get("suggestion", "")

                with st.expander(
                    f"{index}. [{priority}] {issue}",
                    expanded=(index == 1),
                ):
                    st.write(suggestion)
            else:
                st.markdown(f"- {item}")
    else:
        st.write("No specific improvements were returned.")

    # ATS tips
    st.subheader("💡 ATS-friendly tips")

    tips = result.get("ats_tips", [])

    if tips:
        for tip in tips:
            st.markdown(f"- {tip}")
    else:
        st.write("No additional tips were returned.")

    # Download report
    report_json = json.dumps(result, indent=2, ensure_ascii=False)

    st.download_button(
        label="⬇️ Download Analysis Report (JSON)",
        data=report_json,
        file_name="resume_ats_report.json",
        mime="application/json",
        use_container_width=True,
    )

    if st.button("Clear analysis and start again"):
        st.session_state.pop("ats_result", None)
        st.rerun()

st.divider()

st.markdown(
    """
    <p class="small-note">
        Resume ATS Checker provides informational guidance only.
        Review every recommendation for accuracy before changing your CV.
    </p>
    """,
    unsafe_allow_html=True,
)
