"""Unit tests for seniority detection and German recruitment terminology."""

from src.services.ingestion.seniority import detect_seniority, get_seniority_badge


def test_detect_junior_from_title():
    assert detect_seniority("Junior Python Developer") == "junior"
    assert detect_seniority("Trainee Softwareentwicklung (m/w/d)") == "junior"
    assert detect_seniority("Praktikant Data Science") == "junior"
    assert detect_seniority("Werkstudent Python / Cloud") == "junior"
    assert detect_seniority("Absolvent Informatik als Entwickler") == "junior"


def test_detect_junior_from_german_description_without_junior_in_title():
    """
    Crucial requirement: vacancies with generic titles (e.g. 'Softwareentwickler')
    must be classified as junior if the description explicitly mentions German
    entry-level terms like 'Berufseinsteiger', 'Berufsanfänger', 'Absolvent', etc.
    """
    title = "Softwareentwickler (m/w/d)"
    desc = "Wir freuen uns über Bewerbungen von engagierten Berufseinsteigern mit Kenntnissen in Python."
    assert detect_seniority(title, desc) == "junior"

    desc2 = "Diese Stelle richtet sich an Berufsanfänger sowie Quereinsteiger."
    assert detect_seniority("Python Backend Entwickler", desc2) == "junior"

    desc3 = "Keine Berufserfahrung zwingend erforderlich, ideal für Hochschulabsolventen."
    assert detect_seniority("Cloud Engineer", desc3) == "junior"


def test_senior_and_lead_mentoring_junior_does_not_falsely_classify_as_junior():
    """
    Ensure senior or lead positions mentioning 'mentoring junior colleagues'
    are classified as senior/lead, avoiding false-positive junior assignment.
    """
    senior_title = "Senior Software Engineer"
    desc = "You will mentor and guide junior developers and conduct code reviews."
    assert detect_seniority(senior_title, desc) == "senior"

    lead_title = "Lead Architect Cloud Platform"
    desc_lead = "Responsible for architecture decisions and onboarding junior engineers."
    assert detect_seniority(lead_title, desc_lead) == "lead"


def test_lead_priority_over_senior():
    assert detect_seniority("Lead Developer Python") == "lead"
    assert detect_seniority("Principal Software Engineer") == "lead"
    assert detect_seniority("Software Architect / Architekt (m/w/d)") == "lead"
    assert detect_seniority("Head of Software Engineering") == "lead"
    assert detect_seniority("Engineering Manager") == "lead"


def test_default_to_mid_level():
    assert detect_seniority("Python Backend Developer (m/w/d)") == "mid"
    assert detect_seniority("DevOps Specialist", "Maintenance of Kubernetes clusters") == "senior"
    assert detect_seniority("Fullstack Engineer", "Developing modern SPAs with React and FastAPI") == "mid"


def test_seniority_badge_labels():
    assert "Junior" in get_seniority_badge("junior")
    assert "Berufseinsteiger" in get_seniority_badge("junior")
    assert "Mid-Level" in get_seniority_badge("mid")
    assert "Senior" in get_seniority_badge("senior")
    assert "Lead" in get_seniority_badge("lead")
    assert "Mid-Level" in get_seniority_badge("unknown")
