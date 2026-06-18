#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
学历需求分布分析模块。

用途:
- 基于 PostgreSQL `public.recruitment_jobs_normalized` + 职业匹配结果层统计学历分布。
- 适合和 `occupation_salary_analysis.py`、`industry_trend_analysis.py` 搭配，作为当前目录中的主分析链路之一。

前置依赖:
- 先完成统一招聘规范层回填，并生成 `public.skill_extraction_requirement_matches` 职业匹配结果。

关键输入字段:
- `学历要求`
- `publish_month`
- `occupation_core`
- `occupation_category`

输出文件:
- `output/reports/structured_analysis_{mm-dd}/education_by_occupation_category_year.csv`
- `output/reports/structured_analysis_{mm-dd}/education_by_occupation_year.csv`
- `output/reports/structured_analysis_{mm-dd}/education_by_occupation_category_month.csv`
- `output/reports/structured_analysis_{mm-dd}/education_by_occupation_month.csv`
- `output/reports/学历需求分布分析报告.md`

运行方式:
- `python -m src.analysis.education_distribution_analysis`

维护说明:
- 当前脚本使用 PostgreSQL 结构化统计主输入，不属于旧版关键词分析脚本。
"""

import logging
from pathlib import Path

import pandas as pd

from src.analysis.structured_common import build_structured_output_dir, write_csv_with_legacy_copy
from src.analysis.structured_pg_source import load_structured_analysis_dataframe

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EducationDistributionAnalyzer:
    """学历需求分布分析器"""
    
    def __init__(self, base_dir=None, min_jobs_monthly=5, output_dir=None):
        """初始化
        
        Args:
            base_dir: 项目根目录
            min_jobs_monthly: 月度职业层面最小岗位数阈值（默认5）
            output_dir: 可选输出目录
        """
        if base_dir is None:
            base_dir = Path(__file__).parent.parent.parent
        else:
            base_dir = Path(base_dir)
        
        self.base_dir = base_dir
        self.output_dir = Path(output_dir) if output_dir is not None else build_structured_output_dir(
            base_output_dir=base_dir / 'output' / 'reports'
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.min_jobs_monthly = min_jobs_monthly
        
        logger.info("学历需求分布分析器初始化完成")
        logger.info(f"  月度职业层面最小岗位数阈值: {min_jobs_monthly}")
    
    def standardize_education(self, edu_str):
        """标准化学历字段
        
        Args:
            edu_str: 学历要求字符串
            
        Returns:
            str: 标准化后的学历（博士/硕士/本科/大专/高中/中专/学历不限/未明确）
        """
        if pd.isna(edu_str) or edu_str == '':
            return '未明确'
        
        edu_str = str(edu_str)
        
        # 学历不限
        if '不限' in edu_str or '无要求' in edu_str:
            return '学历不限'
        
        # 标准学历（按优先级匹配）
        education_levels = ['博士', '硕士', '本科', '大专', '高中', '中专']
        for edu in education_levels:
            if edu in edu_str:
                return edu
        
        return '未明确'
    
    def extract_year(self, month_str):
        """从月度字段提取年度
        
        Args:
            month_str: YYYY-MM格式的月度字符串
            
        Returns:
            str: YYYY格式的年度
        """
        if pd.isna(month_str):
            return None
        return str(month_str)[:4]
    
    def load_data(self):
        """加载 PostgreSQL 结构化统计主输入。"""
        logger.info("从 PostgreSQL 加载结构化统计主输入...")

        df = load_structured_analysis_dataframe()
        logger.info(f"总数据: {len(df):,} 行")
        
        # 标准化学历
        logger.info("标准化学历...")
        df['学历'] = df['学历要求'].apply(self.standardize_education)
        
        # 提取年度
        logger.info("提取年度...")
        df['年度'] = df['publish_month'].apply(self.extract_year)
        
        # 过滤有效数据
        df_valid = df[
            (df['occupation_core'].notna()) &
            (df['occupation_category'].notna()) &
            (df['学历'].notna())
        ].copy()
        
        logger.info(f"有效数据: {len(df_valid):,} 行 ({len(df_valid)/len(df)*100:.1f}%)")
        
        # 统计学历分布
        logger.info("\n学历分布统计:")
        edu_counts = df_valid['学历'].value_counts()
        for edu, count in edu_counts.items():
            logger.info(f"  {edu:10s}: {count:8,} ({count/len(df_valid)*100:5.1f}%)")
        
        return df_valid
    
    def analyze_category_yearly_education(self, df):
        """职业类别年度学历分布
        
        Returns:
            DataFrame: 年度、职业类别、学历、岗位数量、占比
        """
        logger.info("\n分析职业类别年度学历分布...")
        
        # 过滤有年度数据的记录
        df_year = df[df['年度'].notna()].copy()
        logger.info(f"有年度数据: {len(df_year):,} 行")
        
        # 按年度、职业类别、学历分组
        grouped = df_year.groupby(['年度', 'occupation_category', '学历']).size().reset_index(name='岗位数量')
        
        # 计算占比（同一年度+职业类别内）
        grouped['占比'] = grouped.groupby(['年度', 'occupation_category'])['岗位数量'].transform(
            lambda x: x / x.sum()
        )
        
        # 重命名列
        grouped.rename(columns={'occupation_category': '职业类别'}, inplace=True)
        
        # 排序
        grouped = grouped.sort_values(['年度', '职业类别', '岗位数量'], ascending=[True, True, False])
        
        logger.info(f"生成数据点: {len(grouped):,} 个")
        logger.info(f"  年度数量: {grouped['年度'].nunique()}")
        logger.info(f"  职业类别数量: {grouped['职业类别'].nunique()}")
        
        return grouped[['年度', '职业类别', '学历', '岗位数量', '占比']]
    
    def analyze_occupation_yearly_education(self, df):
        """职业年度学历分布
        
        Returns:
            DataFrame: 年度、职业、职业类别、学历、岗位数量、占比
        """
        logger.info("\n分析职业年度学历分布...")
        
        # 过滤有年度数据的记录
        df_year = df[df['年度'].notna()].copy()
        
        # 按年度、职业、职业类别、学历分组
        grouped = df_year.groupby(['年度', 'occupation_core', 'occupation_category', '学历']).size().reset_index(name='岗位数量')
        
        # 计算占比（同一年度+职业内）
        grouped['占比'] = grouped.groupby(['年度', 'occupation_core'])['岗位数量'].transform(
            lambda x: x / x.sum()
        )
        
        # 重命名列
        grouped.rename(columns={
            'occupation_core': '职业',
            'occupation_category': '职业类别'
        }, inplace=True)
        
        # 排序
        grouped = grouped.sort_values(['年度', '职业类别', '职业', '岗位数量'], ascending=[True, True, True, False])
        
        logger.info(f"生成数据点: {len(grouped):,} 个")
        logger.info(f"  年度数量: {grouped['年度'].nunique()}")
        logger.info(f"  职业数量: {grouped['职业'].nunique()}")
        
        return grouped[['年度', '职业', '职业类别', '学历', '岗位数量', '占比']]
    
    def analyze_category_monthly_education(self, df):
        """职业类别月度学历分布
        
        Returns:
            DataFrame: 月度、职业类别、学历、岗位数量、占比
        """
        logger.info("\n分析职业类别月度学历分布...")
        
        # 过滤有月度数据的记录
        df_month = df[df['publish_month'].notna()].copy()
        logger.info(f"有月度数据: {len(df_month):,} 行")
        
        # 按月度、职业类别、学历分组
        grouped = df_month.groupby(['publish_month', 'occupation_category', '学历']).size().reset_index(name='岗位数量')
        
        # 计算占比（同一月度+职业类别内）
        grouped['占比'] = grouped.groupby(['publish_month', 'occupation_category'])['岗位数量'].transform(
            lambda x: x / x.sum()
        )
        
        # 重命名列
        grouped.rename(columns={
            'publish_month': '月度',
            'occupation_category': '职业类别'
        }, inplace=True)
        
        # 排序
        grouped = grouped.sort_values(['月度', '职业类别', '岗位数量'], ascending=[True, True, False])
        
        logger.info(f"生成数据点: {len(grouped):,} 个")
        logger.info(f"  月度数量: {grouped['月度'].nunique()}")
        logger.info(f"  职业类别数量: {grouped['职业类别'].nunique()}")
        
        return grouped[['月度', '职业类别', '学历', '岗位数量', '占比']]
    
    def analyze_occupation_monthly_education(self, df):
        """职业月度学历分布（设置最小岗位数过滤）
        
        Returns:
            DataFrame: 月度、职业、职业类别、学历、岗位数量、占比
        """
        logger.info("\n分析职业月度学历分布...")
        logger.info(f"  最小岗位数阈值: {self.min_jobs_monthly}（避免稀疏数据噪音）")
        
        # 过滤有月度数据的记录
        df_month = df[df['publish_month'].notna()].copy()
        
        # 按月度、职业、职业类别、学历分组
        grouped = df_month.groupby(['publish_month', 'occupation_core', 'occupation_category', '学历']).size().reset_index(name='岗位数量')
        
        # 过滤：只保留岗位数量>=阈值的数据
        original_count = len(grouped)
        grouped = grouped[grouped['岗位数量'] >= self.min_jobs_monthly].copy()
        filtered_count = original_count - len(grouped)
        
        if filtered_count > 0:
            logger.info(f"  过滤稀疏数据: {filtered_count} 个数据点（岗位数<{self.min_jobs_monthly}）")
        
        # 计算占比（同一月度+职业内）
        grouped['占比'] = grouped.groupby(['publish_month', 'occupation_core'])['岗位数量'].transform(
            lambda x: x / x.sum()
        )
        
        # 重命名列
        grouped.rename(columns={
            'publish_month': '月度',
            'occupation_core': '职业',
            'occupation_category': '职业类别'
        }, inplace=True)
        
        # 排序
        grouped = grouped.sort_values(['月度', '职业类别', '职业', '岗位数量'], ascending=[True, True, True, False])
        
        logger.info(f"生成数据点: {len(grouped):,} 个")
        logger.info(f"  月度数量: {grouped['月度'].nunique()}")
        logger.info(f"  职业数量: {grouped['职业'].nunique()}")
        
        return grouped[['月度', '职业', '职业类别', '学历', '岗位数量', '占比']]
    
    def save_reports(self, df_cat_year, df_occ_year, df_cat_month, df_occ_month):
        """保存分析报告"""
        logger.info("\n保存分析报告...")
        
        # 保存 CSV 文件：英文规范列名为主，中文历史文件名兼容旧汇总脚本。
        category_year_export = df_cat_year.rename(
            columns={'年度': 'year', '职业类别': 'occupation_category', '学历': 'education_level', '岗位数量': 'job_count', '占比': 'share'}
        )
        write_csv_with_legacy_copy(
            category_year_export,
            self.output_dir,
            canonical_filename='education_by_occupation_category_year.csv',
            legacy_filename='职业类别年度学历分布.csv',
        )
        occupation_year_export = df_occ_year.rename(
            columns={'年度': 'year', '职业': 'occupation_core', '职业类别': 'occupation_category', '学历': 'education_level', '岗位数量': 'job_count', '占比': 'share'}
        )
        write_csv_with_legacy_copy(
            occupation_year_export,
            self.output_dir,
            canonical_filename='education_by_occupation_year.csv',
            legacy_filename='职业年度学历分布.csv',
        )
        category_month_export = df_cat_month.rename(
            columns={'月度': 'publish_month', '职业类别': 'occupation_category', '学历': 'education_level', '岗位数量': 'job_count', '占比': 'share'}
        )
        write_csv_with_legacy_copy(
            category_month_export,
            self.output_dir,
            canonical_filename='education_by_occupation_category_month.csv',
            legacy_filename='职业类别月度学历分布.csv',
        )
        occupation_month_export = df_occ_month.rename(
            columns={'月度': 'publish_month', '职业': 'occupation_core', '职业类别': 'occupation_category', '学历': 'education_level', '岗位数量': 'job_count', '占比': 'share'}
        )
        write_csv_with_legacy_copy(
            occupation_month_export,
            self.output_dir,
            canonical_filename='education_by_occupation_month.csv',
            legacy_filename='职业月度学历分布.csv',
        )
        
        logger.info("  ✅ CSV文件已保存")
        
        # 生成 Markdown 报告
        report_file = self.output_dir / '学历需求分布分析报告.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Guangdong Recruitment Data - Education Requirement Distribution Report\n\n")
            
            # 1. Annual education distribution by occupation category
            f.write("## 1. Annual Education Distribution by Occupation Category\n\n")
            
            for year in sorted(df_cat_year['年度'].unique()):
                f.write(f"【{year}年】\n")
                year_data = df_cat_year[df_cat_year['年度'] == year]
                
                for category in year_data['职业类别'].unique():
                    cat_data = year_data[year_data['职业类别'] == category]
                    f.write(f"\n  {category}:\n")
                    
                    for _, row in cat_data.iterrows():
                        f.write(f"    {row['学历']:10s}: {int(row['岗位数量']):6,} 个岗位 ({row['占比']*100:5.1f}%)\n")
                
                f.write("\n")
            
            # 2. Annual education distribution by occupation
            f.write("\n## 2. Annual Education Distribution by Occupation (Top 20 Examples)\n\n")
            
            # 统计各职业总岗位数
            top_occupations = df_occ_year.groupby('职业')['岗位数量'].sum().nlargest(20).index
            
            for year in sorted(df_occ_year['年度'].unique()):
                f.write(f"【{year}年】\n")
                year_data = df_occ_year[
                    (df_occ_year['年度'] == year) & 
                    (df_occ_year['职业'].isin(top_occupations))
                ]
                
                for occupation in top_occupations:
                    occ_data = year_data[year_data['职业'] == occupation]
                    if len(occ_data) > 0:
                        category = occ_data.iloc[0]['职业类别']
                        f.write(f"\n  {occupation} ({category}):\n")
                        
                        for _, row in occ_data.iterrows():
                            f.write(f"    {row['学历']:10s}: {int(row['岗位数量']):6,} 个岗位 ({row['占比']*100:5.1f}%)\n")
                
                f.write("\n")
            
            # 3. Education requirement trends by occupation category
            f.write("\n## 3. Education Requirement Trends by Occupation Category\n\n")
            
            for category in df_cat_year['职业类别'].unique():
                f.write(f"【{category}】\n")
                cat_data = df_cat_year[df_cat_year['职业类别'] == category]
                
                # 按年度展示
                for year in sorted(cat_data['年度'].unique()):
                    year_data = cat_data[cat_data['年度'] == year]
                    total_jobs = year_data['岗位数量'].sum()
                    f.write(f"  {year}年: 总岗位 {int(total_jobs):,} 个\n")
                    
                    for _, row in year_data.iterrows():
                        f.write(f"    {row['学历']:10s}: {row['占比']*100:5.1f}%\n")
                
                f.write("\n")
                
            # 4. Notes
            f.write("\n## 4. Notes\n\n")
            f.write(f"1. 职业类别年度学历分布: {len(df_cat_year):,} 个数据点\n")
            f.write(f"2. 职业年度学历分布: {len(df_occ_year):,} 个数据点\n")
            f.write(f"3. 职业类别月度学历分布: {len(df_cat_month):,} 个数据点\n")
            f.write(f"4. 职业月度学历分布: {len(df_occ_month):,} 个数据点\n")
            f.write(f"\n5. 月度职业层面最小岗位数阈值: {self.min_jobs_monthly}\n")
            f.write("   （避免稀疏数据导致的统计噪音）\n")
            f.write("\n6. 占比计算口径:\n")
            f.write("   - 职业类别年度: 同一年度+职业类别内各学历占比\n")
            f.write("   - 职业年度: 同一年度+职业内各学历占比\n")
            f.write("   - 月度同理\n")
            f.write("\n7. 学历分类:\n")
            f.write("   - 明确学历: 博士、硕士、本科、大专、高中、中专\n")
            f.write("   - 其他: 学历不限、未明确\n")
        
        logger.info(f"  ✅ Markdown报告已保存: {report_file}")
    
    def run(self):
        """运行完整分析"""
        logger.info("=" * 80)
        logger.info("学历需求分布分析")
        logger.info("=" * 80)
        
        # 加载数据
        df = self.load_data()
        
        # 分析
        df_cat_year = self.analyze_category_yearly_education(df)
        df_occ_year = self.analyze_occupation_yearly_education(df)
        df_cat_month = self.analyze_category_monthly_education(df)
        df_occ_month = self.analyze_occupation_monthly_education(df)
        
        # 保存报告
        self.save_reports(df_cat_year, df_occ_year, df_cat_month, df_occ_month)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ 学历需求分布分析完成!")
        logger.info("=" * 80)
        logger.info("\n生成的文件:")
        logger.info("  - education_by_occupation_category_year.csv")
        logger.info("  - education_by_occupation_year.csv")
        logger.info("  - education_by_occupation_category_month.csv")
        logger.info("  - education_by_occupation_month.csv")
        logger.info("  - output/reports/学历需求分布分析报告.md")


def main():
    """主函数"""
    analyzer = EducationDistributionAnalyzer()
    analyzer.run()


if __name__ == '__main__':
    main()
