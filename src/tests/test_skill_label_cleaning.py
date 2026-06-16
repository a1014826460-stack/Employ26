from src.skill_extraction.labeling.regression_dataset import _clean_skill_item


def test_clean_skill_item_drops_generic_container_skill():
    item = {"name": "财务软件", "evidence": "财务软件", "skill_type": "工具软件"}
    assert _clean_skill_item(item, "熟悉财务软件") is None


def test_clean_skill_item_drops_english_grade_certificate():
    item = {"name": "大学英语四级", "evidence": "英语4级以上", "skill_type": "证书/资质"}
    assert _clean_skill_item(item, "英语4级以上，能够交流") is None


def test_clean_skill_item_keeps_grounded_specific_tool():
    item = {"name": "Excel", "evidence": "EXCEL函数", "skill_type": "办公软件"}
    assert _clean_skill_item(item, "熟悉EXCEL函数") == {"name": "Excel", "evidence": "EXCEL函数", "skill_type": "办公软件"}
