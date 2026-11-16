from clipshorts.config import Config


def test_defaults():
    c = Config(_env_file=None)
    assert c.language_list == ["de", "en"]
    assert c.whisper_model == "large-v3-turbo"
    assert c.fps_cap == 30
    assert c.auth_token == ""


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LANGUAGES", " DE , fr ")
    monkeypatch.setenv("AUTH_TOKEN", "s3cret")
    c = Config(_env_file=None)
    assert c.data_dir == tmp_path
    assert c.language_list == ["de", "fr"]
    assert c.auth_token == "s3cret"
    assert c.db_path == tmp_path / "clipshorts.db"
    assert c.clip_dir(7) == tmp_path / "clips" / "7"


def test_env_file(tmp_path):
    f = tmp_path / ".env"
    f.write_text("WHISPER_MODEL=base\nFPS_CAP=24\n")
    c = Config(_env_file=str(f))
    assert c.whisper_model == "base"
    assert c.fps_cap == 24
