from src.core import (
    AppConfig,
    ElasticsearchConfig,
    MLConfig,
    Settings,
    get_settings,
)


def test_default_settings():
    get_settings.cache_clear()
    settings = get_settings()

    assert isinstance(settings.elasticsearch, ElasticsearchConfig)
    assert settings.elasticsearch.url == "http://localhost:9200"
    assert settings.elasticsearch.timeout_seconds == 30.0
    assert settings.elasticsearch.max_retries == 3

    assert isinstance(settings.app, AppConfig)
    assert settings.app.app_name == "IR-Arena"
    assert settings.app.app_env == "development"
    assert settings.app.debug is True
    assert settings.app.api_host == "0.0.0.0"
    assert settings.app.api_port == 8000

    assert isinstance(settings.ml, MLConfig)
    assert settings.ml.default_embedding_model == "paraphrase-multilingual-MiniLM-L12-v2"
    assert settings.ml.embedding_dimension == 384


def test_modular_settings_composition():
    es_cfg = ElasticsearchConfig(
        url="http://modular-es:9200",
        timeout_seconds=45.0,
        max_retries=2,
    )
    app_cfg = AppConfig(
        app_name="Modular-Arena",
        app_env="staging",
        debug=False,
        api_host="127.0.0.1",
        api_port=9000,
    )
    ml_cfg = MLConfig(default_embedding_model="custom-bert")

    settings = Settings(elasticsearch=es_cfg, app=app_cfg, ml=ml_cfg)

    assert settings.elasticsearch.url == "http://modular-es:9200"
    assert settings.elasticsearch.timeout_seconds == 45.0
    assert settings.elasticsearch.max_retries == 2
    assert settings.app.app_name == "Modular-Arena"
    assert settings.app.debug is False
    assert settings.ml.default_embedding_model == "custom-bert"


def test_env_var_override(monkeypatch):
    monkeypatch.setenv("ELASTICSEARCH_URL", "http://env-override:9200")
    monkeypatch.setenv("APP_NAME", "Env-Overridden-Arena")

    settings = Settings()
    assert settings.elasticsearch.url == "http://env-override:9200"
    assert settings.app.app_name == "Env-Overridden-Arena"


def test_get_settings_cached():
    get_settings.cache_clear()
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
