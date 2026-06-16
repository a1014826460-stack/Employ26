from src.skill_extraction.hard.pg_matcher import should_trigger_strong_revalidation


def test_should_trigger_strong_revalidation_when_precision_too_high():
    summary = {
        "estimated_precision": 0.98,
        "parse_success_rate": 0.96,
        "total_samples": 50,
    }
    assert should_trigger_strong_revalidation(summary)


def test_should_not_trigger_strong_revalidation_when_precision_moderate():
    summary = {
        "estimated_precision": 0.86,
        "parse_success_rate": 0.9,
        "total_samples": 50,
    }
    assert not should_trigger_strong_revalidation(summary)
