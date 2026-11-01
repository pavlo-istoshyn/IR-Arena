"""Skill and metadata extraction class tuned for Python backend, AI, and DevOps profiles."""

from __future__ import annotations

import html
import re
from typing import Any

SKILL_PATTERNS: dict[str, re.Pattern[str]] = {
    # Core Languages & Frameworks
    "Python": re.compile(r"\bpython(?:3)?\b", re.IGNORECASE),
    "FastAPI": re.compile(r"\bfast[- ]?api\b", re.IGNORECASE),
    "Django": re.compile(r"\bdjango\b", re.IGNORECASE),
    "Flask": re.compile(r"\bflask\b", re.IGNORECASE),
    "AsyncIO": re.compile(r"\basync[-. ]?io\b", re.IGNORECASE),
    "Pydantic": re.compile(r"\bpydantic(?:\s*v[12])?\b", re.IGNORECASE),
    "aiogram": re.compile(r"\baiogram(?:\s*3)?\b", re.IGNORECASE),
    "Celery": re.compile(r"\bcelery\b", re.IGNORECASE),
    "RabbitMQ": re.compile(r"\brabbit[- ]?mq\b", re.IGNORECASE),
    "gRPC": re.compile(r"\bgrpc\b", re.IGNORECASE),
    "WebSocket": re.compile(r"\bweb[- ]?sockets?\b", re.IGNORECASE),
    "SQLModel": re.compile(r"\bsql[- ]?model\b", re.IGNORECASE),
    "Tornado": re.compile(r"\btornado\b", re.IGNORECASE),
    "Uvicorn": re.compile(r"\buvicorn\b", re.IGNORECASE),
    "Poetry": re.compile(r"\bpoetry\b", re.IGNORECASE),
    # Databases & ORM
    "PostgreSQL": re.compile(r"\bpostgres(?:ql)?\b", re.IGNORECASE),
    "SQLAlchemy": re.compile(r"\bsql[- ]?alchemy\b", re.IGNORECASE),
    "Alembic": re.compile(r"\balembic\b", re.IGNORECASE),
    "asyncpg": re.compile(r"\basyncpg\b", re.IGNORECASE),
    "Redis": re.compile(r"\bredis\b", re.IGNORECASE),
    "SQL": re.compile(r"\bsql\b", re.IGNORECASE),
    "Elasticsearch": re.compile(r"\belastic[- ]?search\b", re.IGNORECASE),
    "MongoDB": re.compile(r"\bmongo(?:db)?\b", re.IGNORECASE),
    "ClickHouse": re.compile(r"\bclickhouse\b", re.IGNORECASE),
    "Cassandra": re.compile(r"\bcassandra\b", re.IGNORECASE),
    "Snowflake": re.compile(r"\bsnowflake\b", re.IGNORECASE),
    "BigQuery": re.compile(r"\bbigquery\b", re.IGNORECASE),
    "SQLite": re.compile(r"\bsqlite\b", re.IGNORECASE),
    "MySQL": re.compile(r"\bmysql\b", re.IGNORECASE),
    "MariaDB": re.compile(r"\bmariadb\b", re.IGNORECASE),
    "DuckDB": re.compile(r"\bduckdb\b", re.IGNORECASE),
    "DynamoDB": re.compile(r"\bdynamodb\b", re.IGNORECASE),
    # Data & Big Data
    "Pandas": re.compile(r"\bpandas\b", re.IGNORECASE),
    "NumPy": re.compile(r"\bnumpy\b", re.IGNORECASE),
    "Polars": re.compile(r"\bpolars\b", re.IGNORECASE),
    "PySpark": re.compile(r"\bpyspark\b", re.IGNORECASE),
    "Apache Spark": re.compile(r"\b(?:apache[- ]?)?spark\b", re.IGNORECASE),
    "Apache Airflow": re.compile(r"\b(?:apache[- ]?)?airflow\b", re.IGNORECASE),
    "dbt": re.compile(r"\bdbt\b", re.IGNORECASE),
    "Kafka": re.compile(r"\bkafka\b", re.IGNORECASE),
    "Dask": re.compile(r"\bdask\b", re.IGNORECASE),
    # AI, LLM, Agents & Tools
    "PyTorch": re.compile(r"\bpytorch\b", re.IGNORECASE),
    "TensorFlow": re.compile(r"\btensorflow\b", re.IGNORECASE),
    "Keras": re.compile(r"\bkeras\b", re.IGNORECASE),
    "Scikit-Learn": re.compile(r"\bscikit[- ]learn\b|\bsklearn\b", re.IGNORECASE),
    "OpenAI": re.compile(r"\bopen[- ]?ai\b", re.IGNORECASE),
    "Claude": re.compile(r"\bclaude\b|\banthropic\b", re.IGNORECASE),
    "Claude Code": re.compile(r"\bclaude[- ]code\b", re.IGNORECASE),
    "Antigravity": re.compile(r"\bantigravity\b", re.IGNORECASE),
    "LLM": re.compile(
        r"(?:\bLLMs?\b|\blarge language models?\b|\bsprachmodell\w*\b|\bgen(?:erative)?[- ]?(?:ai|ki)\b)",
        re.IGNORECASE,
    ),
    "RAG": re.compile(r"(?:\bRAG\b|(?i:\bretrieval[- ]augmented(?:\s+generation)?\b))"),
    "Tool Calling": re.compile(r"\b(?:tool|function)[- ]call(?:ing|s)?\b", re.IGNORECASE),
    "SSE": re.compile(r"(?:\bSSE\b|(?i:\bserver[- ]sent[- ]events?\b))"),
    "FAISS": re.compile(r"\bfaiss\b", re.IGNORECASE),
    "sentence-transformers": re.compile(r"\bsentence[- ]transformers?\b", re.IGNORECASE),
    "Embeddings": re.compile(
        r"\bembeddings?\b|\bvektor[- ]?(?:db|datenbank\w*|store\w*)\b|\bvector[- ](?:db|database|store\w*)\b",
        re.IGNORECASE,
    ),
    "MCP": re.compile(r"(?:\bMCP\b|(?i:\bmodel[- ]context[- ]protocol\b))"),
    "LangChain": re.compile(r"\blangchain\b", re.IGNORECASE),
    "LlamaIndex": re.compile(r"\bllamaindex\b", re.IGNORECASE),
    "Hugging Face": re.compile(r"\bhugging[- ]?face\b", re.IGNORECASE),
    "Transformers": re.compile(r"\btransformers?\b", re.IGNORECASE),
    "Ollama": re.compile(r"\bollama\b", re.IGNORECASE),
    "vLLM": re.compile(r"\bvllm\b", re.IGNORECASE),
    "MLflow": re.compile(r"\bmlflow\b", re.IGNORECASE),
    "Weights & Biases": re.compile(r"\bwandb\b|\bweights\s*&\s*biases\b", re.IGNORECASE),
    "Computer Vision": re.compile(r"\bcomputer[- ]vision\b|\bopencv\b", re.IGNORECASE),
    "NLP": re.compile(r"\bnlp\b|\bspacy\b|\bnatural[- ]language[- ]processing\b", re.IGNORECASE),
    "ChromaDB": re.compile(r"\bchroma(?:db)?\b", re.IGNORECASE),
    "Qdrant": re.compile(r"\bqdrant\b", re.IGNORECASE),
    "Weaviate": re.compile(r"\bweaviate\b", re.IGNORECASE),
    "Milvus": re.compile(r"\bmilvus\b", re.IGNORECASE),
    "Pinecone": re.compile(r"\bpinecone\b", re.IGNORECASE),
    # Testing & Reliability
    "pytest": re.compile(r"\bpytest\b", re.IGNORECASE),
    "Testcontainers": re.compile(r"\btestcontainers?\b", re.IGNORECASE),
    "mypy": re.compile(r"\bmypy\b", re.IGNORECASE),
    "ruff": re.compile(r"\bruff\b", re.IGNORECASE),
    "TDD": re.compile(r"\btdd\b|\btest[- ]driven[- ]development\b", re.IGNORECASE),
    "Playwright": re.compile(r"\bplaywright\b", re.IGNORECASE),
    "Selenium": re.compile(r"\bselenium\b", re.IGNORECASE),
    # Infrastructure, DevOps & OS
    "Docker": re.compile(r"\bdocker\b", re.IGNORECASE),
    "Docker Compose": re.compile(r"\bdocker[- ]compose\b", re.IGNORECASE),
    "Kubernetes": re.compile(r"\bkubernetes\b|\bk8s\b", re.IGNORECASE),
    "Helm": re.compile(r"\bhelm\b", re.IGNORECASE),
    "Linux": re.compile(r"\blinux\b", re.IGNORECASE),
    "Bash": re.compile(r"\bbash\b|\bshell[- ](?:skript|script)\w*\b", re.IGNORECASE),
    "systemd": re.compile(r"\bsystemd\b", re.IGNORECASE),
    "Git": re.compile(r"\bgit\b|\bgithub\b|\bgitlab\b", re.IGNORECASE),
    "GitHub Actions": re.compile(r"\bgithub[- ]actions\b", re.IGNORECASE),
    "GitLab CI": re.compile(r"\bgitlab[- ]ci\b", re.IGNORECASE),
    "CI/CD": re.compile(
        r"\bci\s*/\s*cd\b|\bcicd\b|\bcontinuous[- ](?:integration|delivery|deployment)\b",
        re.IGNORECASE,
    ),
    "Terraform": re.compile(r"\bterraform\b", re.IGNORECASE),
    "Ansible": re.compile(r"\bansible\b", re.IGNORECASE),
    "Nginx": re.compile(r"\bnginx\b", re.IGNORECASE),
    "Prometheus": re.compile(r"\bprometheus\b", re.IGNORECASE),
    "Grafana": re.compile(r"\bgrafana\b", re.IGNORECASE),
    "OpenTelemetry": re.compile(r"\bopentelemetry\b|\botel\b", re.IGNORECASE),
    "Datadog": re.compile(r"\bdatadog\b", re.IGNORECASE),
    # Architecture, Cloud & Interop
    "AWS": re.compile(r"(?i:\bAWS\b|\bAmazon\s+Web\s+Services\b|\bEC2\b|\bECR\b)|\bS3\b"),
    "Azure": re.compile(r"\bazure\b", re.IGNORECASE),
    "GCP": re.compile(r"(?:\bGCP\b|\bGoogle Cloud\b)", re.IGNORECASE),
    "REST API": re.compile(
        r"(?i:\brest[- ]?(?:api\w*|service\w*|endpoint\w*)\b|\brestful\b)|\bREST\b"
    ),
    "GraphQL": re.compile(r"\bgraphql\b", re.IGNORECASE),
    "Microservices": re.compile(r"\bmicroservices?\b|\bmikroservices?\b", re.IGNORECASE),
    "Event-Driven": re.compile(r"\bevent[- ]driven\b|\bedas?\b", re.IGNORECASE),
    "Clean Architecture": re.compile(r"\bclean[- ]architecture\b|\bhexagonal\b", re.IGNORECASE),
    "DDD": re.compile(r"\bddd\b|\bdomain[- ]driven[- ]design\b", re.IGNORECASE),
    "OAuth2": re.compile(r"\boauth(?:2(?:\.0)?)?\b", re.IGNORECASE),
    "JWT": re.compile(r"\bjwt\b|\bjson[- ]web[- ]tokens?\b", re.IGNORECASE),
    "Keycloak": re.compile(r"\bkeycloak\b", re.IGNORECASE),
    # Other Languages & Frontend
    "TypeScript": re.compile(r"\btypescript\b|\bts\b", re.IGNORECASE),
    "JavaScript": re.compile(r"\bjavascript\b|\bjs\b", re.IGNORECASE),
    "React": re.compile(r"\breact(?:js)?\b", re.IGNORECASE),
    "Vue.js": re.compile(r"\bvue(?:js)?\b", re.IGNORECASE),
    "Angular": re.compile(r"\bangular\b", re.IGNORECASE),
    "Node.js": re.compile(r"\bnode(?:\.js)?\b", re.IGNORECASE),
    "Golang": re.compile(r"\bgolang\b|\bgo[- ]dev\b", re.IGNORECASE),
    "Rust": re.compile(r"\brust\b", re.IGNORECASE),
    "C++": re.compile(r"\bc\+\+\b|\bcpp\b", re.IGNORECASE),
    "Java": re.compile(r"\bjava\b", re.IGNORECASE),
    "C#": re.compile(r"(?:(?<!\w)C#(?!\w)|\bc[- ]?sharp\b)", re.IGNORECASE),
    ".NET": re.compile(r"(?:(?<!\w)\.NET\b|\bdotnet\b)", re.IGNORECASE),
}

REMOTE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bhome[- ]?office\b", re.IGNORECASE),
    re.compile(r"\bremote\b", re.IGNORECASE),
    re.compile(r"\bmobil(?:e[sn]?)?\s+arbeiten\b", re.IGNORECASE),
    re.compile(r"\bmobile[- ]office\b", re.IGNORECASE),
    re.compile(r"\btelearbeit\b", re.IGNORECASE),
    re.compile(r"\borts?ungebunden\b", re.IGNORECASE),
    re.compile(r"\bortsunabh\w+", re.IGNORECASE),
    re.compile(r"\bwork[- ]from[- ]home\b|\bwfh\b", re.IGNORECASE),
    re.compile(r"\bhybrid(?:es)?\s+(?:arbeiten|modell)\b", re.IGNORECASE),
    re.compile(r"\b100%\s+remote\b", re.IGNORECASE),
]


class SkillExtractor:
    """Service responsible for cleaning HTML and extracting skills, city, and remote status."""

    def __init__(
        self,
        skill_patterns: dict[str, re.Pattern[str]] | None = None,
        remote_patterns: list[re.Pattern[str]] | None = None,
    ) -> None:
        self.skill_patterns = skill_patterns if skill_patterns is not None else SKILL_PATTERNS
        self.remote_patterns = remote_patterns if remote_patterns is not None else REMOTE_PATTERNS

    @staticmethod
    def clean_html_text(text: str | None) -> str:
        """Strip HTML markup, replace linebreaks, unescape entities, and normalize whitespace."""
        if not text:
            return ""
        cleaned = re.sub(
            r"</?(?:br|p|div|li|tr|h[1-6]|ul|ol)[^>]*>", "\n", text, flags=re.IGNORECASE
        )
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        cleaned = html.unescape(cleaned)
        cleaned = cleaned.replace("\u00a0", " ")
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
        return cleaned.strip()

    def extract_skills(self, text: str | None) -> list[str]:
        """Extract recognized technical skills as a sorted list of canonical skill names."""
        if not text:
            return []

        matched: set[str] = set()
        for skill_name, pattern in self.skill_patterns.items():
            if pattern.search(text):
                matched.add(skill_name)

        return sorted(matched)

    def extract_remote(
        self,
        text: str | None = None,
        raw_job: dict[str, Any] | None = None,
    ) -> bool:
        """Determine whether the position offers remote, hybrid, or homeoffice flexibility."""
        if raw_job and isinstance(raw_job, dict):
            if raw_job.get("homeofficemoeglich") is True:
                return True

            home_type = str(raw_job.get("homeofficetyp") or "").upper().strip()
            if home_type in {"JA", "VOLL", "TEILWEISE", "NACH_VEREINBARUNG"}:
                return True

            for field in ("arbeitszeit", "arbeitszeitmodell", "arbeitszeitmodellBezeichnung"):
                val = raw_job.get(field)
                if isinstance(val, str) and val:
                    for pat in self.remote_patterns:
                        if pat.search(val):
                            return True

        if text:
            for pat in self.remote_patterns:
                if pat.search(text):
                    return True

        return False

    @staticmethod
    def extract_city(raw_job: dict[str, Any]) -> str | None:
        """Extract primary city or region name from job location structure."""
        if not raw_job or not isinstance(raw_job, dict):
            return None

        lokationen = raw_job.get("stellenlokationen") or []
        if isinstance(lokationen, list) and lokationen:
            first_loc = lokationen[0]
            if isinstance(first_loc, dict):
                adresse = first_loc.get("adresse") or {}
                if isinstance(adresse, dict):
                    ort = adresse.get("ort") or adresse.get("region")
                    if ort:
                        return str(ort).strip()
                ort = first_loc.get("ort") or first_loc.get("region")
                if ort:
                    return str(ort).strip()

        arbeitsort = raw_job.get("arbeitsort")
        if isinstance(arbeitsort, dict):
            ort = arbeitsort.get("ort") or arbeitsort.get("region")
            if ort:
                return str(ort).strip()
        elif isinstance(arbeitsort, str) and arbeitsort.strip():
            return arbeitsort.strip()

        return None
