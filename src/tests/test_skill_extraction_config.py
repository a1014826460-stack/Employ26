from src.skill_extraction.config import load_skill_extraction_config


def test_skill_config_exposes_local_llm_paths_and_models():
    config = load_skill_extraction_config()
    assert config.llm_model_path
    assert config.llm_cheap_model
    assert config.llm_strong_model


def test_skill_config_prefers_local_model_candidates():
    config = load_skill_extraction_config()
    assert "models" in str(config.llm_model_path) or "Qwen" in str(config.llm_model_path)


def test_skill_extraction_config_exposes_occupation_detail_match_defaults():
    config = load_skill_extraction_config()

    assert config.occupation_detail_match_table == "public.occupation_detail_matches"
    assert str(config.occupation_detail_model_path).replace("\\", "/").endswith(
        "output/penghui/rag_round2_training/bge-large-round2-finetuned"
    )
    assert config.occupation_detail_top_k == 10
