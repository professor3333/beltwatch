from pathlib import Path

from fastapi.testclient import TestClient

from beltwatch.api.app import create_app
from beltwatch.api.settings import Settings


def test_review_ui_is_served(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert "Review unwanted materials in a paper-recycling stream." in page.text
    assert "not contamination by weight" in page.text
    for asset, kind in (("/app.js", "javascript"), ("/styles.css", "css")):
        response = client.get(asset)
        assert response.status_code == 200 and kind in response.headers["content-type"]


def test_ui_never_uses_inner_html_for_server_data() -> None:
    script = (Path(__file__).resolve().parents[2] / "frontend" / "app.js").read_text()
    assert ".innerHTML" not in script and "insertAdjacentHTML" not in script


def test_api_routes_take_precedence_over_static_files(client: TestClient) -> None:
    assert client.get("/health/live").json()["status"] == "alive"


def test_frontend_dir_is_configurable(settings: Settings, tmp_path: Path) -> None:
    custom = tmp_path / "ui"
    custom.mkdir()
    (custom / "index.html").write_text("<h1>custom</h1>")
    with TestClient(
        create_app(settings.model_copy(update={"frontend_dir": custom, "require_worker": False}))
    ) as c:
        assert c.get("/").text == "<h1>custom</h1>"
