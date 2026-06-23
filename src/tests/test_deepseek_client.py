from src.llm.deepseek_client import (
    DeepSeekConfig,
    DeepSeekResponseError,
    build_deepseek_client,
    parse_json_response,
)


def test_parse_json_response_accepts_fenced_json() -> None:
    assert parse_json_response('```json\n{"ok": true}\n```') == {"ok": True}


def test_parse_json_response_accepts_embedded_json_object() -> None:
    assert parse_json_response('结果如下 {"winner":"A"} thanks') == {"winner": "A"}


def test_build_deepseek_client_reads_env_api_key(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = build_deepseek_client()
    assert client.config.api_key == "test-key"
    assert client.config.base_url == "https://api.deepseek.com"


def test_map_json_batches_results(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = build_deepseek_client()

    def fake_complete_json(**kwargs):
        return {"ok": kwargs["user_prompt"]}

    monkeypatch.setattr(client, "complete_json", fake_complete_json)
    outputs = client.map_json(
        [
            {"system_prompt": "s", "user_prompt": "u1"},
            {"system_prompt": "s", "user_prompt": "u2"},
        ],
        workers=2,
    )
    assert len(outputs) == 2
    assert outputs[0]["ok"] in {"u1", "u2"}


def test_complete_json_raises_on_invalid_json(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = build_deepseek_client()

    class FakeChoice:
        def __init__(self):
            self.message = type("Message", (), {"content": "not-json"})()

    class FakeResponse:
        def __init__(self):
            self.choices = [FakeChoice()]

    def fake_create(**kwargs):
        return FakeResponse()

    fake_openai = type(
        "FakeOpenAI",
        (),
        {"chat": type("FakeChat", (), {"completions": type("FakeCompletions", (), {"create": staticmethod(fake_create)})()})()},
    )
    monkeypatch.setattr(client, "_build_openai_client", lambda: fake_openai)

    try:
        client.complete_json(system_prompt="s", user_prompt="u")
        assert False, "expected DeepSeekResponseError"
    except DeepSeekResponseError:
        assert True
