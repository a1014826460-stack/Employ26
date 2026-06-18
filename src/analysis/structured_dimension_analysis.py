"""经验、公司规模等结构化维度补充统计。"""

from __future__ import annotations

import logging
import re

import pandas as pd

from src.analysis.analysis_common import normalize_city
from src.analysis.structured_common import resolve_structured_paths, write_csv_with_legacy_copy
from src.analysis.structured_pg_source import load_structured_analysis_dataframe


logger = logging.getLogger(__name__)


REQUIRED_COLUMNS = {
    "occupation_core",
    "occupation_category",
    "publish_month",
    "工作城市",
    "经验要求",
    "公司规模",
}


class StructuredDimensionAnalyzer:
    """结构化维度补充分析器。"""

    def __init__(self, base_dir=None, output_dir=None, min_group_size: int = 10):
        """初始化分析器。

        Args:
            base_dir: 可选项目根目录。
            output_dir: 可选输出目录。
            min_group_size: 进入输出表的最小岗位数。
        """
        paths = resolve_structured_paths(base_dir=base_dir, output_dir=output_dir)
        self.base_dir = paths.project_root
        self.output_dir = paths.output_dir
        self.min_group_size = min_group_size

    def load_data(self) -> tuple[pd.DataFrame, list[str]]:
        """加载并补充结构化维度字段。"""
        df = load_structured_analysis_dataframe()
        df = df.copy()
        df["experience_level"] = df["经验要求"].apply(normalize_experience)
        df["company_size_level"] = df["公司规模"].apply(normalize_company_size)
        df["city_normalized"] = df["工作城市"].apply(normalize_city)
        return df, ["postgres:public.recruitment_jobs_normalized", "postgres:public.skill_extraction_requirement_matches"]

    def analyze_experience_by_occupation(self, df: pd.DataFrame) -> pd.DataFrame:
        """统计职业维度经验要求分布。"""
        work_df = df[
            df["occupation_core"].notna()
            & df["occupation_category"].notna()
            & df["experience_level"].notna()
        ].copy()
        result = (
            work_df.groupby(["occupation_core", "occupation_category", "experience_level"])
            .size()
            .reset_index(name="job_count")
        )
        result["share"] = result.groupby(["occupation_core"])["job_count"].transform(lambda x: x / x.sum())
        return result[result["job_count"] >= self.min_group_size].sort_values(
            ["occupation_category", "occupation_core", "job_count"],
            ascending=[True, True, False],
        )

    def analyze_company_size_by_industry_city(self, df: pd.DataFrame) -> pd.DataFrame:
        """统计城市与行业下的公司规模分布。"""
        industry_column = "industry_clean" if "industry_clean" in df.columns else "公司行业"
        work_df = df[
            df["city_normalized"].notna()
            & df[industry_column].notna()
            & df["company_size_level"].notna()
        ].copy()
        result = (
            work_df.groupby(["city_normalized", industry_column, "company_size_level"])
            .size()
            .reset_index(name="job_count")
            .rename(columns={industry_column: "industry_normalized"})
        )
        result["share"] = result.groupby(["city_normalized", "industry_normalized"])["job_count"].transform(
            lambda x: x / x.sum()
        )
        return result[result["job_count"] >= self.min_group_size].sort_values(
            ["city_normalized", "industry_normalized", "job_count"],
            ascending=[True, True, False],
        )

    def analyze_city_occupation_demand(self, df: pd.DataFrame) -> pd.DataFrame:
        """统计城市与职业需求分布。"""
        work_df = df[
            df["city_normalized"].notna()
            & df["occupation_core"].notna()
            & df["occupation_category"].notna()
        ].copy()
        result = (
            work_df.groupby(["city_normalized", "occupation_core", "occupation_category"])
            .size()
            .reset_index(name="job_count")
        )
        result["share"] = result.groupby(["city_normalized"])["job_count"].transform(lambda x: x / x.sum())
        return result[result["job_count"] >= self.min_group_size].sort_values(
            ["city_normalized", "job_count"],
            ascending=[True, False],
        )

    def save_outputs(
        self,
        *,
        experience_df: pd.DataFrame,
        company_size_df: pd.DataFrame,
        city_occupation_df: pd.DataFrame,
    ) -> list[str]:
        """保存结构化维度补充分析产物。"""
        output_files: list[str] = []
        output_files.extend(
            write_csv_with_legacy_copy(
                experience_df,
                self.output_dir,
                canonical_filename="experience_by_occupation.csv",
                legacy_filename="职业经验要求分布.csv",
            )
        )
        output_files.extend(
            write_csv_with_legacy_copy(
                company_size_df,
                self.output_dir,
                canonical_filename="company_size_by_city_industry.csv",
                legacy_filename="城市行业公司规模分布.csv",
            )
        )
        output_files.extend(
            write_csv_with_legacy_copy(
                city_occupation_df,
                self.output_dir,
                canonical_filename="city_occupation_demand.csv",
                legacy_filename="城市职业需求分布.csv",
            )
        )
        report_path = self.output_dir / "结构化维度补充分析报告.md"
        report_path.write_text(
            "\n".join(
                [
                    "# Structured Dimension Supplementary Analysis Report",
                    "",
                    "## 1. Experience Requirements x Occupation",
                    experience_df.head(30).to_string(index=False) if not experience_df.empty else "无结果",
                    "",
                    "## 2. Company Size x City x Industry",
                    company_size_df.head(30).to_string(index=False) if not company_size_df.empty else "无结果",
                    "",
                    "## 3. City x Occupation Demand",
                    city_occupation_df.head(30).to_string(index=False) if not city_occupation_df.empty else "无结果",
                ]
            ),
            encoding="utf-8",
        )
        output_files.append(report_path.name)
        return output_files

    def run(self) -> dict[str, int]:
        """运行结构化维度补充分析。"""
        logger.info("运行结构化维度补充分析")
        df, _ = self.load_data()
        experience_df = self.analyze_experience_by_occupation(df)
        company_size_df = self.analyze_company_size_by_industry_city(df)
        city_occupation_df = self.analyze_city_occupation_demand(df)
        self.save_outputs(
            experience_df=experience_df,
            company_size_df=company_size_df,
            city_occupation_df=city_occupation_df,
        )
        return {
            "experience_rows": len(experience_df),
            "company_size_rows": len(company_size_df),
            "city_occupation_rows": len(city_occupation_df),
        }


def normalize_experience(value: object) -> str:
    """标准化经验要求。"""
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return "unknown"
    if "不限" in text or "无经验" in text:
        return "no_experience_required"
    numbers = [int(match) for match in re.findall(r"\d+", text)]
    if not numbers:
        return "unknown"
    years = min(numbers)
    if years <= 1:
        return "0_1_year"
    if years <= 3:
        return "1_3_years"
    if years <= 5:
        return "3_5_years"
    if years <= 10:
        return "5_10_years"
    return "10_plus_years"


def normalize_company_size(value: object) -> str:
    """标准化公司规模。"""
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return "unknown"
    if "少于" in text or "0-20" in text or "20人以下" in text:
        return "lt_20"
    numbers = [int(match) for match in re.findall(r"\d+", text)]
    if not numbers:
        return "unknown"
    upper = max(numbers)
    if upper < 100:
        return "20_99"
    if upper < 500:
        return "100_499"
    if upper < 1000:
        return "500_999"
    if upper < 10000:
        return "1000_9999"
    return "10000_plus"


def main() -> None:
    """CLI 入口。"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    StructuredDimensionAnalyzer().run()


if __name__ == "__main__":
    main()
