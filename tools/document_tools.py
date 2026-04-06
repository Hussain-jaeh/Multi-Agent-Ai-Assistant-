"""
Document processing and search tools using ChromaDB and OpenAI embeddings.
Provides RAG (Retrieval-Augmented Generation) capabilities for course materials.
"""

import os
from pathlib import Path
from typing import Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# Directory constants
DOCUMENTS_DIR = Path("data/documents")
CHROMA_PERSIST_DIR = Path("data/chroma_db")

# ─── School Policies (hardcoded for demo) ────────────────────────────────────

SCHOOL_POLICIES = {
    "admissions": """
ADMISSION REQUIREMENTS:
• Minimum GPA: 2.5 for undergraduate, 3.0 for graduate programs
• Required documents: Transcripts, 2–3 letters of recommendation, personal statement,
  SAT/ACT scores (undergrad) or GRE/GMAT scores (grad)
• Application deadline: Rolling admissions — priority deadline January 15th
• International students: TOEFL 80+ or IELTS 6.5+
• Application fee: $75 (non-refundable)
• Interviews: Required for Business, Education, and Healthcare programs
""",
    "schedules": """
ACADEMIC CALENDAR:
• Fall Semester:   September 1  – December 20
• Spring Semester: January 15   – May 10
• Summer Session I:  June 1     – July 15
• Summer Session II: July 20    – August 31
• Registration opens: 8 weeks before semester start
• Add/Drop period: First 2 weeks of each semester
• Midterm exams: Week 8 of each semester
• Final exams: Last 2 weeks of semester
• Spring Break: 1 week in March
• Thanksgiving Break: Wednesday before through Sunday after Thanksgiving
""",
    "fees": """
TUITION AND FEES:
• Undergraduate tuition: $15,000/semester (12–18 credits, full-time)
• Graduate tuition:      $18,000/semester (9–12 credits, full-time)
• Part-time rate: $1,200/credit hour (undergrad) | $1,500/credit hour (grad)
• Student activity fee: $250/semester
• Technology fee: $150/semester
• Health insurance: $800/semester (waivable with proof of coverage)
• Payment due: First day of classes
• Payment plan: 4-installment plan available (1.5% processing fee)
• Financial aid: FAFSA required — priority deadline February 1st
• Scholarships: Merit-based (academic, athletic, arts) and need-based available
""",
    "academic_policies": """
ACADEMIC POLICIES:
• Grading: A (93–100), A- (90–92), B+ (87–89), B (83–86), B- (80–82),
           C+ (77–79), C (73–76), C- (70–72), D (60–69), F (below 60)
• Minimum passing grade: C for major courses; D for electives
• Academic probation: GPA below 2.0 for two consecutive semesters
• Attendance: Max 3 absences (4-credit course); max 2 absences (3-credit course)
• Late work: 10% deduction per day; max 50% deduction; not accepted after 1 week
• Academic integrity: Zero tolerance for plagiarism — automatic F + possible suspension
• Withdrawal deadline: Week 10 of semester (W grade, no GPA impact)
• Incomplete grade: Must complete within 6 weeks of semester end
• Dean's List: Semester GPA 3.7+ with 12+ credit hours
""",
    "campus_services": """
CAMPUS SERVICES AND RESOURCES:
• Library: 24/7 during finals; regular hours 7am–11pm (Mon–Thu), 7am–8pm (Fri–Sun)
• Tutoring Center: Free peer tutoring (walk-in & appointment), Mon–Fri 9am–7pm
• Writing Center: Essay review and writing help — by appointment
• Career Services: Resume review, job fairs (Fall & Spring), internship placements
• Counseling Center: Free mental health (5 sessions/semester); crisis hotline 24/7
• Health Center: Basic medical care, Mon–Fri 8am–5pm; emergency on-call
• Student ID: Required for library, gym, dining, printing (100 free pages/semester)
• Parking: Permit $150/semester; free evenings & weekends after 5pm
• Dining: Meal plans available; dining halls open 7am–9pm daily
• IT Help Desk: Mon–Fri 8am–8pm; Sat–Sun 10am–4pm; remote support available
""",
}

# ─── Document Search Tool ─────────────────────────────────────────────────────


class DocumentSearchInput(BaseModel):
    """Input schema for document search."""

    query: str = Field(description="Search query to find relevant information in uploaded course materials")


class DocumentSearchTool(BaseTool):
    """
    RAG-powered search tool for uploaded course documents.

    Uses ChromaDB as the vector store and OpenAI embeddings to perform
    semantic similarity search over indexed PDF course materials.
    """

    name: str = "document_search"
    description: str = (
        "Search through uploaded course materials and PDFs for relevant information. "
        "Use this when you need to find specific content from course documents, "
        "lecture notes, or textbooks. Input should be a descriptive search query."
    )
    args_schema: Type[BaseModel] = DocumentSearchInput

    def _run(self, query: str) -> str:
        """Perform semantic search across indexed course materials."""
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            from langchain_community.vectorstores import Chroma

            CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

            # Check if any PDFs have been uploaded
            if not DOCUMENTS_DIR.exists() or not any(DOCUMENTS_DIR.glob("*.pdf")):
                return (
                    "No course materials are uploaded yet. "
                    "I'll answer from general knowledge.\n\n"
                    "Tip: Upload PDF files via the sidebar to enable document search."
                )

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectorstore = Chroma(
                persist_directory=str(CHROMA_PERSIST_DIR),
                embedding_function=embeddings,
                collection_name="course_materials",
            )

            results = vectorstore.similarity_search_with_relevance_scores(query, k=3)

            if not results:
                return "No relevant content found in uploaded materials. Answering from general knowledge."

            formatted = []
            for i, (doc, score) in enumerate(results, 1):
                source = Path(doc.metadata.get("source", "Unknown")).name
                page = doc.metadata.get("page", "N/A")
                relevance_pct = int(score * 100)
                snippet = doc.page_content.strip()[:500]
                formatted.append(
                    f"**[Source {i}]** {source} — Page {page} (Relevance: {relevance_pct}%)\n"
                    f"{snippet}..."
                )

            return "\n\n---\n\n".join(formatted)

        except Exception as e:
            return (
                f"Document search unavailable ({str(e)}). "
                "Answering from general knowledge instead."
            )


# ─── Policy Search Tool ───────────────────────────────────────────────────────


class PolicySearchInput(BaseModel):
    """Input schema for policy search."""

    query: str = Field(
        description=(
            "Question about school policies, admissions, tuition, schedules, "
            "academic rules, or campus services"
        )
    )


class PolicySearchTool(BaseTool):
    """
    Search tool for school policies and administrative information.

    Covers admissions requirements, academic calendar, tuition/fees,
    academic policies, and campus services.
    """

    name: str = "policy_search"
    description: str = (
        "Search school policies, procedures, and administrative information. "
        "Use this to answer questions about admissions, fees, schedules, "
        "academic rules, or any campus service. Input a descriptive question."
    )
    args_schema: Type[BaseModel] = PolicySearchInput

    # Keyword → policy-key mapping for routing
    _KEYWORD_MAP: dict = {
        "admissions": [
            "admission", "apply", "application", "requirements", "gpa",
            "toefl", "gre", "sat", "enroll", "accept", "entrance",
        ],
        "schedules": [
            "schedule", "semester", "registration", "calendar", "dates",
            "exam", "break", "midterm", "final", "add", "drop",
        ],
        "fees": [
            "fee", "tuition", "cost", "payment", "financial aid",
            "scholarship", "fafsa", "price", "money", "pay",
        ],
        "academic_policies": [
            "grade", "gpa", "attendance", "policy", "plagiarism",
            "withdraw", "incomplete", "probation", "academic", "integrity",
        ],
        "campus_services": [
            "library", "tutoring", "career", "counseling", "health",
            "parking", "dining", "service", "help", "support", "it",
        ],
    }

    def _run(self, query: str) -> str:
        """Find and return relevant policy information."""
        q = query.lower()
        matched_keys = [
            key
            for key, keywords in self._KEYWORD_MAP.items()
            if any(kw in q for kw in keywords)
        ]

        # Fall back to all policies if nothing matched
        if not matched_keys:
            matched_keys = list(SCHOOL_POLICIES.keys())

        sections = [SCHOOL_POLICIES[k] for k in matched_keys[:2]]
        return "Based on our school policies:\n\n" + "\n\n".join(sections)


# ─── PDF Indexing Helper ──────────────────────────────────────────────────────


def index_documents(pdf_path: str) -> bool:
    """
    Index a PDF file into the ChromaDB vector store.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        True on success, False on failure.
    """
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_community.vectorstores import Chroma
        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

        # Load and split the PDF
        loader = PyPDFLoader(pdf_path)
        raw_docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = splitter.split_documents(raw_docs)

        # Upsert into ChromaDB
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(CHROMA_PERSIST_DIR),
            collection_name="course_materials",
        )
        vectorstore.persist()
        return True

    except Exception as exc:
        print(f"[index_documents] Failed to index {pdf_path}: {exc}")
        return False
