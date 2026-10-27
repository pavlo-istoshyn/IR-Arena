"""Job title and role filtering class for IT vacancies."""

import re

EXCLUDE_TITLE_RE = re.compile(
    r"(?:\bdevops\b|\bsystem(?:s)?[- ]?admin\w*\b|\bcloud[- ]?(?:infra|platform|operations|architect|administrator)\b|"
    r"\bsite[- ]reliability\b|\bsre\b|\bit[- ]?administrator\w*\b|\bnetzwerk\w*\b|\bhelpdesk\b|"
    r"\bsupport[- ]?(?:engineer|specialist|mitarbeiter)\b|\bit[- ]support\b)",
    re.IGNORECASE,
)

PYTHON_BACKEND_AI_RE = re.compile(
    r"(?:\bbackend\b|\bfaiss\b|\brag\b|\bai\b|\bki\b|\bllm\b|\bmachine learning\b)",
    re.IGNORECASE,
)


class JobRoleFilter:
    """Filter determining whether a job title belongs to non-target operations/support roles."""

    def __init__(
        self,
        exclude_pattern: re.Pattern[str] = EXCLUDE_TITLE_RE,
        include_override_pattern: re.Pattern[str] = PYTHON_BACKEND_AI_RE,
    ) -> None:
        self.exclude_pattern = exclude_pattern
        self.include_override_pattern = include_override_pattern

    def is_non_target_role(self, title: str) -> bool:
        """Check if title represents a pure DevOps / System Administrator / Support role."""
        if not title:
            return False
        if self.exclude_pattern.search(title):
            if not self.include_override_pattern.search(title):
                return True
        return False
