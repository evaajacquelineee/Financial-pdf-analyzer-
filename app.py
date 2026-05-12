import os
import textwrap

import pdfplumber
import streamlit as st
from dotenv import load_dotenv
import anthropic

# Rough upper bound on how many characters of PDF text
# we send to Claude in a single request.
MAX_MODEL_CHARS = 250_000

# Keywords used to identify pages that contain core financial statements
# in very long reports (income statement, balance sheet, cash flows, etc.).
FINANCIAL_SECTION_KEYWORDS = [
    "consolidated statement of income",
    "consolidated statements of income",
    "statement of income",
    "income statement",
    "statement of profit",
    "profit and loss",
    "statement of operations",
    "consolidated balance sheet",
    "consolidated balance sheets",
    "balance sheet",
    "statement of financial position",
    "consolidated statement of cash flows",
    "consolidated statements of cash flows",
    "cash flow statement",
    "statement of cash flows",
]


def load_api_key() -> str:
    """
    Load the Anthropic API key from a .env file using python-dotenv.
    Expects ANTHROPIC_API_KEY in the environment after loading.
    """
    # Load environment variables from .env in the current directory (if present)
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to your .env file, e.g. ANTHROPIC_API_KEY=your_key_here"
        )
    return api_key


def extract_text_from_pdf(file, focus_financial_sections: bool = False) -> str:
    """
    Extract text from an uploaded PDF file-like object using pdfplumber.

    If focus_financial_sections is True, try to keep mostly the core financial
    statement pages (income statement, balance sheet, cash flows) and a small
    window of pages around them, which is better for very long reports.
    """
    with pdfplumber.open(file) as pdf:
        pages = pdf.pages

        # If not focusing, just take all text.
        if not focus_financial_sections:
            all_text = []
            for page in pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    all_text.append(page_text)
            return "\n\n".join(all_text).strip()

        # Focus mode: look for pages whose text contains any of the keywords.
        candidate_indices = []
        for idx, page in enumerate(pages):
            page_text = page.extract_text() or ""
            if not page_text:
                continue
            lower = page_text.lower()
            if any(keyword in lower for keyword in FINANCIAL_SECTION_KEYWORDS):
                # Include this page and a small window around it
                for offset in range(-2, 3):  # two pages before and after
                    j = idx + offset
                    if 0 <= j < len(pages):
                        candidate_indices.append(j)

        if not candidate_indices:
            # Fallback: no obvious financial pages found, return all text.
            all_text = []
            for page in pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    all_text.append(page_text)
            return "\n\n".join(all_text).strip()

        selected_indices = sorted(set(candidate_indices))
        selected_text = []
        for j in selected_indices:
            page_text = pages[j].extract_text() or ""
            if page_text.strip():
                selected_text.append(page_text)
        return "\n\n".join(selected_text).strip()


def extract_texts_from_pdfs(files, focus_financial_sections: bool = False) -> str:
    """
    Extract text from one or multiple uploaded PDF files and combine them.
    Each document is clearly separated so the model can do multi‑year trend analysis.
    """
    chunks = []
    for idx, file in enumerate(files, start=1):
        try:
            text = extract_text_from_pdf(file, focus_financial_sections=focus_financial_sections)
        except Exception as e:
            text = f"[Error reading this PDF: {e}]"
        name = getattr(file, "name", f"File {idx}")
        if text:
            chunks.append(f"=== DOCUMENT {idx}: {name} ===\n\n{text}")
    return "\n\n\n".join(chunks).strip()


def build_prompt(pdf_text: str) -> str:
    """
    Construct the prompt that will be sent to Claude.
    """
    # Allow a large chunk of text so Claude can see most reports.
    # Very large annual reports may still be truncated.
    truncated_text = pdf_text[:MAX_MODEL_CHARS]

    instructions = """
You are a financial analyst. One or more company PDF reports are provided below
(for different years or periods). Use them to perform a detailed, year‑by‑year
and overall analysis.

1. **Key financial metrics (per year, where possible)**  
   - Revenue, gross profit, operating income, net income  
   - Cash position and total debt  
   - Any guidance or outlook figures if given  

2. **Profitability ratios (per year)**  
   - Gross profit margin  
   - Operating margin  
   - Net margin (if feasible)  
   - Return on equity (ROE)  

3. **Liquidity ratios (per year)**  
   - Current ratio  
   - Quick ratio (if data is available)  

4. **Leverage & coverage ratios (per year)**  
   - Debt‑to‑equity ratio  
   - Interest coverage ratio (e.g., EBIT / interest expense)  

5. **Free cash flow analysis (per year and overall)**  
   - Compute or approximate free cash flow (FCF) if the statements allow  
   - Comment on whether FCF is consistently positive/negative and any trends  

6. **Trend analysis across years**  
   - Describe how revenues, margins, leverage, liquidity, and FCF are evolving  
   - Highlight improving vs. deteriorating areas  

7. **Red flags and risks**  
   - Deteriorating margins, revenues, or FCF  
   - Liquidity concerns or going‑concern language  
   - High leverage, covenant risks, or refinancing needs  
   - Accounting concerns, restatements, or unusual adjustments  
   - Customer or supplier concentration risks  

8. **Overall financial health summary**  
   - Brief plain‑language explanation of how healthy the company appears  
   - Near‑term risks to watch  
   - Any positive strengths worth highlighting  

Return your answer in this structure:

## Key Financial Metrics (by Year)
- ...

## Profitability Ratios (by Year)
- ...

## Liquidity & Leverage Ratios (by Year)
- ...

## Free Cash Flow Analysis
- ...

## Trend Analysis Across Years
- ...

## Red Flags & Risks
- ...

## Overall Financial Health
Paragraph summary.

If certain numbers or ratios cannot be calculated from the PDFs, say explicitly
that the data is not available instead of guessing.
"""

    return textwrap.dedent(instructions).strip() + "\n\n---\n\nPDF CONTENT STARTS BELOW:\n\n" + truncated_text


def analyze_with_claude(pdf_text: str) -> str:
    """
    Send the extracted PDF text to Claude and return its analysis.
    """
    api_key = load_api_key()
    client = anthropic.Anthropic(api_key=api_key)

    prompt = build_prompt(pdf_text)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        temperature=0.2,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    # Anthropic responses contain a list of content blocks; join text parts
    parts = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts).strip() or "No analysis text returned from Claude."


def main():
    st.set_page_config(
        page_title="Financial PDF Analyzer",
        page_icon="💼",
        layout="wide",
    )

    # Simple custom styling for a cleaner look
    st.markdown(
        """
        <style>
        .main-title {
            text-align: center;
            font-size: 2.4rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            text-align: center;
            font-size: 1rem;
            color: #6c757d;
            margin-bottom: 1.5rem;
        }
        .footer {
            font-size: 0.8rem;
            color: #999999;
            text-align: center;
            margin-top: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-title">Financial PDF Analyzer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Built by Eva Jacqueline &mdash; AI-powered multi-year financial statement analysis</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "For best results and due to model input limits, upload PDFs that contain "
        "primarily the financial statements and notes (income statement, balance "
        "sheet, cash flow statement), rather than the entire annual report if it is very long."
    )

    with st.sidebar:
        st.header("Upload PDF(s)")
        uploaded_file = st.file_uploader(
            "Choose one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help=(
                "Upload one or multiple financial PDFs (e.g., annual reports for "
                "different years) to enable ratio and trend analysis."
            ),
        )

    if not uploaded_file:
        st.info("Please upload at least one PDF file to begin.")
        return

    focus_financial_sections = st.checkbox(
        "Focus on financial statement sections (recommended for very long reports)",
        value=True,
        help=(
            "When checked, the app looks for pages containing income statements, "
            "balance sheets, and cash flow statements (plus nearby pages) instead "
            "of sending every page. This helps stay within model limits for very "
            "large PDFs."
        ),
    )

    st.subheader("1. Extracted Text Preview")
    with st.spinner("Extracting text from PDF(s)..."):
        try:
            # uploaded_file is a list when accept_multiple_files=True
            files = uploaded_file if isinstance(uploaded_file, list) else [uploaded_file]
            pdf_text = extract_texts_from_pdfs(files, focus_financial_sections=focus_financial_sections)
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return

    if not pdf_text:
        st.warning("No text could be extracted from this PDF. It may be scanned images or protected.")
        return

    total_chars = len(pdf_text)
    sent_chars = min(total_chars, MAX_MODEL_CHARS)

    preview_chars = 1500
    st.text_area(
        "PDF Text (preview)",
        value=pdf_text[:preview_chars],
        height=250,
        help="Showing the first part of the extracted text.",
    )

    if total_chars > MAX_MODEL_CHARS:
        st.caption(
            f"Note: The PDFs contain about {total_chars:,} characters of text. "
            f"Only the first {sent_chars:,} characters will be sent to Claude for analysis, "
            "so very long reports (hundreds of pages) may be partially sampled."
        )
    else:
        st.caption(
            f"Approximate text size: {total_chars:,} characters. "
            "All of this will be sent to Claude for analysis."
        )

    st.subheader("2. Analyze with Claude")
    analyze_button = st.button("Run Financial Analysis")

    if analyze_button:
        with st.spinner("Sending to Claude and analyzing..."):
            try:
                analysis = analyze_with_claude(pdf_text)
            except Exception as e:
                st.error(f"Error while calling Claude API: {e}")
                return

        st.subheader("3. Analysis Results")
        st.markdown(analysis)

    st.markdown('<div class="footer">© 2026 Eva Jacqueline</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

