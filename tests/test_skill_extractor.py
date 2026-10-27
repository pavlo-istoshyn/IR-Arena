"""Unit tests for SkillExtractor, JobRoleFilter, and JobNormalizer."""

from src.services.ingestion import JobNormalizer, JobRoleFilter, SkillExtractor

extractor = SkillExtractor()


def test_extract_skills_empty_or_none():
    assert extractor.extract_skills("") == []
    assert extractor.extract_skills(None) == []
    assert extractor.extract_skills("   ") == []


def test_extract_skills_python_backend_stack():
    text = """
    Wir suchen einen Python Entwickler mit Kenntnissen in FastAPI, AsyncIO und Pydantic.
    Erfahrung mit Django oder Flask sowie aiogram für Bot-Entwicklung ist von Vorteil.
    """
    skills = extractor.extract_skills(text)
    expected = ["AsyncIO", "Django", "FastAPI", "Flask", "Pydantic", "Python", "aiogram"]
    for s in expected:
        assert s in skills


def test_extract_skills_databases_and_orm():
    text = """
    Technologien: PostgreSQL (asyncpg), SQLAlchemy 2.0, Alembic für DB-Migrationen und Redis für Caching.
    Gute Kenntnisse in reinem SQL und Elasticsearch runden dein Profil ab.
    """
    skills = extractor.extract_skills(text)
    expected = [
        "Alembic",
        "Elasticsearch",
        "PostgreSQL",
        "Redis",
        "SQL",
        "SQLAlchemy",
        "asyncpg",
    ]
    for s in expected:
        assert s in skills


def test_extract_skills_ai_llm_agents():
    text = """
    Arbeiten mit LLM Modellen via OpenAI SDK und Anthropic Claude API.
    Kenntnisse in Tool Calling / Function Calling, Single-Stream SSE Architekturen,
    RAG Pipelines mit FAISS, sentence-transformers, Embeddings und Model Context Protocol (MCP).
    Erfahrung mit LangChain oder LlamaIndex erwünscht.
    """
    skills = extractor.extract_skills(text)
    expected = [
        "Claude",
        "Embeddings",
        "FAISS",
        "LLM",
        "LangChain",
        "LlamaIndex",
        "MCP",
        "OpenAI",
        "RAG",
        "SSE",
        "Tool Calling",
        "sentence-transformers",
    ]
    for s in expected:
        assert s in skills


def test_extract_skills_testing_and_qa():
    text = "Automatisierte Tests mit pytest, Testcontainers für DB-Tests, mypy für Typisierung und ruff."
    skills = extractor.extract_skills(text)
    assert skills == ["Testcontainers", "mypy", "pytest", "ruff"]


def test_extract_skills_devops_infra_cloud():
    text = """
    Stack: Docker, Docker Compose, Kubernetes (K8s), Linux (bash, systemd), Git,
    GitHub Actions (CI/CD Pipelines) und Cloud-Erfahrung mit AWS (EC2, S3), Azure oder GCP.
    """
    skills = extractor.extract_skills(text)
    expected = [
        "AWS",
        "Azure",
        "Bash",
        "CI/CD",
        "Docker",
        "Docker Compose",
        "GCP",
        "Git",
        "GitHub Actions",
        "Kubernetes",
        "Linux",
        "systemd",
    ]
    for s in expected:
        assert s in skills


def test_extract_skills_secondary_and_architecture():
    text = "Grundkenntnisse in C# und .NET Core, Entwurf von REST APIs, Microservices, Kafka und GraphQL."
    skills = extractor.extract_skills(text)
    expected = [".NET", "C#", "GraphQL", "Kafka", "Microservices", "REST API"]
    for s in expected:
        assert s in skills


def test_extract_skills_avoid_false_positives():
    text = """
    Take a resting place in the garden.
    He used drag and drop to move the file.
    Er spielt gerne Gitarre und wirft Bälle in den basket.
    Er mag das alpine Scratching nicht.
    """
    skills = extractor.extract_skills(text)
    assert "REST API" not in skills
    assert "RAG" not in skills
    assert "Git" not in skills
    assert "Bash" not in skills
    assert "C#" not in skills


def test_extract_remote_via_raw_job_flags():
    assert extractor.extract_remote(raw_job={"homeofficemoeglich": True}) is True
    assert extractor.extract_remote(raw_job={"homeofficetyp": "NACH_VEREINBARUNG"}) is True
    assert extractor.extract_remote(raw_job={"homeofficetyp": "TEILWEISE"}) is True
    assert extractor.extract_remote(raw_job={"homeofficetyp": "VOLL"}) is True
    assert (
        extractor.extract_remote(raw_job={"arbeitszeit": "Vollzeit, Homeoffice Option vorhanden"})
        is True
    )
    assert (
        extractor.extract_remote(raw_job={"homeofficemoeglich": False, "homeofficetyp": "NEIN"})
        is False
    )


def test_extract_remote_via_text():
    assert extractor.extract_remote("Mobiles Arbeiten und bis zu 80% Homeoffice möglich.") is True
    assert extractor.extract_remote("Diese Stelle ist 100% remote.") is True
    assert extractor.extract_remote("Möglichkeit zur Telearbeit nach Einarbeitung.") is True
    assert extractor.extract_remote("Hybrides Arbeiten (2 Tage Office, 3 Tage Home Office).") is True
    assert extractor.extract_remote("Wir bieten ein attraktives Hybrides Modell.") is True
    assert extractor.extract_remote("Präsenzarbeit im Büro vor Ort in Berlin.") is False


def test_extract_remote_empty_or_none():
    assert extractor.extract_remote() is False
    assert extractor.extract_remote("") is False
    assert extractor.extract_remote(None, None) is False


def test_clean_html_text():
    raw_html = (
        "<h3>Ihre Aufgaben:</h3>\n"
        "<p>Entwicklung mit <b>Python</b> &amp; FastAPI.</p><br/>"
        "<ul><li>Gute Bezahlung &euro;</li><li>Sch&ouml;nes B&uuml;ro</li></ul>"
    )
    cleaned = extractor.clean_html_text(raw_html)
    assert "<" not in cleaned
    assert ">" not in cleaned
    assert "Python" in cleaned
    assert "FastAPI" in cleaned
    assert "&" in cleaned
    assert "€" in cleaned
    assert "Schönes" in cleaned
    assert "Büro" in cleaned


def test_clean_html_text_none_empty():
    assert extractor.clean_html_text(None) == ""
    assert extractor.clean_html_text("") == ""
    assert extractor.clean_html_text("   ") == ""


def test_extract_city():
    sample_job = {
        "stellenlokationen": [
            {
                "adresse": {
                    "strasse": "Hauptstr.",
                    "plz": "10115",
                    "ort": "Berlin",
                    "region": "BERLIN",
                }
            }
        ]
    }
    assert extractor.extract_city(sample_job) == "Berlin"

    sample_fallback = {
        "stellenlokationen": [
            {
                "adresse": {
                    "region": "BAYERN",
                }
            }
        ]
    }
    assert extractor.extract_city(sample_fallback) == "BAYERN"

    sample_direct = {"stellenlokationen": [{"ort": "München"}]}
    assert extractor.extract_city(sample_direct) == "München"

    assert extractor.extract_city({}) is None
    assert extractor.extract_city({"stellenlokationen": []}) is None
    assert extractor.extract_city({"stellenlokationen": [None]}) is None


def test_extract_skills_rest_api_variations():
    assert "REST API" in extractor.extract_skills("Erfahrung mit RESTful APIs und Microservices.")
    assert "REST API" in extractor.extract_skills("Erstellung von Rest-APIs mit FastAPI.")
    assert "REST API" in extractor.extract_skills("Konzeption moderner REST-Services.")
    assert "REST API" in extractor.extract_skills("Schnittstellen via rest api anbinden.")
    assert "REST API" in extractor.extract_skills("Backend mit REST und GraphQL Schnittstellen.")

    assert "REST API" not in extractor.extract_skills("Für den Rest des Jahres ist Urlaub geplant.")
    assert "REST API" not in extractor.extract_skills("Den Rest des Teams unterstützen.")
    assert "REST API" not in extractor.extract_skills("A relaxing resting zone.")


def test_extract_skills_aws_variations():
    assert "AWS" in extractor.extract_skills("Deployment auf AWS Cloud.")
    assert "AWS" in extractor.extract_skills("Erfahrung mit aws und Docker.")
    assert "AWS" in extractor.extract_skills("Kenntnisse in Amazon Web Services (EC2, S3).")
    assert "AWS" in extractor.extract_skills("Container Registry via ECR.")


def test_extract_skills_bash_and_shell():
    assert "Bash" in extractor.extract_skills("Automatisierung mit Shell Scripting.")
    assert "Bash" in extractor.extract_skills("Erfahrung mit Shell-Skripten und Linux.")
    assert "Bash" in extractor.extract_skills("Skripte in Bash und Python.")


def test_extract_skills_ai_tools():
    skills = extractor.extract_skills("Verwendung von Claude Code und Antigravity als AI-Agenten.")
    assert "Claude Code" in skills
    assert "Claude" in skills
    assert "Antigravity" in skills


def test_extract_remote_extended_german_phrases():
    assert extractor.extract_remote("Wir bieten die Möglichkeit zum mobilen Arbeiten.") is True
    assert extractor.extract_remote("100% ortsunabhängiges Arbeiten in ganz Deutschland.") is True
    assert extractor.extract_remote("Diese Stelle ist ortsunabhängig zu besetzen.") is True
    assert extractor.extract_remote("Mobiles Arbeiten im Team.") is True


def test_clean_html_text_nbsp_and_headings():
    raw = "<h2>Backend Anforderungen</h2><p>Python&nbsp;Entwickler gesucht.</p>"
    cleaned = extractor.clean_html_text(raw)
    assert "\u00a0" not in cleaned
    assert "Python Entwickler gesucht." in cleaned
    assert "Backend Anforderungen\n\nPython Entwickler gesucht." in cleaned


def test_extract_city_arbeitsort_fallback():
    job_dict = {"arbeitsort": {"ort": "Köln", "region": "Nordrhein-Westfalen"}}
    assert extractor.extract_city(job_dict) == "Köln"

    job_str = {"arbeitsort": "Hamburg"}
    assert extractor.extract_city(job_str) == "Hamburg"


def test_job_role_filter_and_normalizer():
    role_filter = JobRoleFilter()
    assert role_filter.is_non_target_role("Linux Systemadministrator (m/w/d)") is True
    assert role_filter.is_non_target_role("DevOps & Python Backend Engineer") is False

    normalizer = JobNormalizer()
    job = normalizer.normalize(
        refnr="100-1",
        search_item={"stellenangebotsTitel": "Python Backend Developer", "firma": "ACME GmbH"},
        details={"stellenangebotsBeschreibung": "<p>FastAPI und Docker im Homeoffice</p>"},
    )
    assert job.id == "100-1"
    assert job.title == "Python Backend Developer"
    assert job.company == "ACME GmbH"
    assert job.is_remote is True
    assert "FastAPI" in job.skills
    assert "Docker" in job.skills
