"""Transformation and normalization class converting raw job dicts into JobListing."""

from typing import Any

from src.schemas.job import JobListing
from src.services.ingestion.skill_extractor import SkillExtractor


class JobNormalizer:
    """Transforms raw Arbeitsagentur API search/detail payloads into domain JobListing objects."""

    def __init__(self, extractor: SkillExtractor | None = None) -> None:
        self.extractor = extractor or SkillExtractor()

    def normalize(
        self,
        refnr: str,
        search_item: dict[str, Any],
        details: dict[str, Any] | None,
    ) -> JobListing:
        """Transform search item and details dictionary into validated JobListing."""
        data = details if details else search_item

        title = (
            data.get("stellenangebotsTitel")
            or search_item.get("stellenangebotsTitel")
            or "Softwareentwickler"
        )
        company = (
            data.get("firma")
            or search_item.get("firma")
            or data.get("arbeitgeber")
            or search_item.get("arbeitgeber")
            or ""
        )
        city = self.extractor.extract_city(data) or self.extractor.extract_city(search_item)
        raw_desc = (
            data.get("stellenangebotsBeschreibung")
            or search_item.get("stellenangebotsBeschreibung")
            or ""
        )
        description = self.extractor.clean_html_text(raw_desc)
        pub_date = data.get("datumErsteVeroeffentlichung") or search_item.get(
            "datumErsteVeroeffentlichung"
        )

        combined_text = f"{title}\n{description}"
        skills = self.extractor.extract_skills(combined_text)
        is_remote = self.extractor.extract_remote(text=combined_text, raw_job=data or search_item)

        source_url = f"https://www.arbeitsagentur.de/jobsuche/jobdetail/{refnr}"
        return JobListing(
            id=refnr,
            title=title,
            company=company,
            city=city,
            description=description,
            skills=skills,
            is_remote=is_remote,
            source_url=source_url,
            published_date=pub_date,
        )
