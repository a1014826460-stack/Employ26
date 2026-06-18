"""
职业/职业类别薪资分析模块。

用途:
- 基于 PostgreSQL `public.recruitment_jobs_normalized` + 职业匹配结果层分析职业薪资分布。
- 同时生成月度趋势、学历交叉统计、Markdown 报告、CSV 明细和 HTML 图表。
- 这是当前目录里最核心的薪资分析脚本，优先级高于旧版 `salary_analysis.py`。

前置依赖:
- 先完成统一招聘规范层回填，并生成 `public.skill_extraction_requirement_matches` 职业匹配结果。

关键输入字段:
- `薪资水平`
- `occupation_core`
- `occupation_category`
- `publish_month`
- `学历要求`

输出文件:
- `output/reports/职业类别薪资分析报告.md`
- `output/reports/structured_analysis_{mm-dd}/salary_by_occupation_category_month.csv`
- `output/reports/structured_analysis_{mm-dd}/salary_by_occupation_month.csv`
- `output/reports/structured_analysis_{mm-dd}/salary_by_education_occupation_category.csv`
- `output/reports/structured_analysis_{mm-dd}/salary_by_education_occupation.csv`
- `output/reports/职业类别薪资分析图.html`

运行方式:
- `python -m src.analysis.occupation_salary_analysis`

维护说明:
- 当前脚本使用 PostgreSQL 结构化统计主输入，和目录中的 `generate_standardized_tables.py`
  `generate_excel_summary.py` 存在上下游关系，不属于重复实现。
"""

import pandas as pd
import numpy as np
import logging
import re

from pathlib import Path

from src.analysis.structured_common import build_structured_output_dir, write_csv_with_legacy_copy
from src.analysis.structured_pg_source import load_structured_analysis_dataframe

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_salary(salary_str):
    """解析薪资字符串，返回最小值和最大值（月薪，单位：元）"""
    if pd.isna(salary_str) or salary_str == '':
        return None, None
    
    salary_str = str(salary_str).strip()
    
    # 匹配各种格式
    patterns = [
        (r'(\d+\.?\d*)-(\d+\.?\d*)万', 10000),
        (r'(\d+)-(\d+)万', 10000),
        (r'(\d+\.?\d*)万-(\d+\.?\d*)万', 10000),
        (r'(\d+)-(\d+)千', 1000),
        (r'(\d+)千-(\d+)千', 1000),
        (r'(\d+)-(\d+)k', 1000),
        (r'(\d+)k-(\d+)k', 1000),
        (r'(\d+)-(\d+)元', 1),
        (r'(\d+)元-(\d+)元', 1),
    ]
    
    for pattern, multiplier in patterns:
        match = re.search(pattern, salary_str, re.IGNORECASE)
        if match:
            min_sal = float(match.group(1)) * multiplier
            max_sal = float(match.group(2)) * multiplier
            
            if '年' in salary_str:
                min_sal /= 12
                max_sal /= 12
            
            return min_sal, max_sal
    
    return None, None


class OccupationSalaryAnalyzer:
    """职业类别薪资分析器"""
    
    def __init__(self, base_dir=None, output_dir=None):
        """初始化"""
        if base_dir is None:
            base_dir = Path(__file__).parent.parent.parent
        else:
            base_dir = Path(base_dir)
        
        self.base_dir = base_dir
        self.output_dir = Path(output_dir) if output_dir is not None else build_structured_output_dir(
            base_output_dir=base_dir / 'output' / 'reports'
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("职业类别薪资分析器初始化完成")
    
    def load_data(self):
        """加载 PostgreSQL 结构化统计主输入。"""
        logger.info("从 PostgreSQL 加载结构化统计主输入...")

        df = load_structured_analysis_dataframe()
        logger.info(f"总数据: {len(df):,} 行")
        
        # 解析薪资
        logger.info("解析薪资...")
        df[['最低薪资', '最高薪资']] = df['薪资水平'].apply(
            lambda x: pd.Series(parse_salary(x))
        )
        df['平均薪资'] = (df['最低薪资'] + df['最高薪资']) / 2
        
        # 过滤有效数据
        df_valid = df[
            (df['平均薪资'].notna()) & 
            (df['occupation_core'].notna()) &
            (df['occupation_category'].notna())
        ].copy()
        
        logger.info(f"有效数据: {len(df_valid):,} 行 ({len(df_valid)/len(df)*100:.1f}%)")
        
        return df_valid
    
    def analyze_by_occupation(self, df):
        """按职业类别分析薪资"""
        logger.info("\n分析职业类别薪资...")
        
        # 按职业类别统计
        category_stats = df.groupby('occupation_category')['平均薪资'].agg([
            ('平均薪资', 'mean'),
            ('中位数薪资', 'median'),
            ('最低薪资', 'min'),
            ('最高薪资', 'max'),
            ('岗位数量', 'count')
        ]).sort_values('平均薪资', ascending=False)
        
        logger.info("\n职业类别薪资排行:")
        for category, row in category_stats.iterrows():
            logger.info(f"  {category:15s} - 平均: {row['平均薪资']:8,.0f}元 | "
                       f"中位数: {row['中位数薪资']:8,.0f}元 | 岗位数: {int(row['岗位数量']):6,}")
        
        # 按具体职业核心词统计
        core_stats = df.groupby('occupation_core')['平均薪资'].agg([
            ('平均薪资', 'mean'),
            ('中位数薪资', 'median'),
            ('岗位数量', 'count')
        ]).sort_values('平均薪资', ascending=False)
        
        logger.info("\nTop 30 职业核心词薪资排行:")
        for i, (core, row) in enumerate(core_stats.head(30).iterrows(), 1):
            logger.info(f"  {i:2d}. {core:25s} - 平均: {row['平均薪资']:8,.0f}元 | "
                       f"中位数: {row['中位数薪资']:8,.0f}元 | 岗位数: {int(row['岗位数量']):5,}")
        
        return category_stats, core_stats
    
    def analyze_by_month(self, df):
        """按月度分析职业类别薪资趋势"""
        logger.info("\n分析职业类别月度薪资趋势...")
        
        # 过滤有月份数据的记录
        df_month = df[df['publish_month'].notna()].copy()
        logger.info(f"有月份数据: {len(df_month):,} 行")
        
        # 按职业类别和月份统计
        monthly_stats = df_month.groupby(['occupation_category', 'publish_month'])['平均薪资'].agg([
            ('平均薪资', 'mean'),
            ('岗位数量', 'count')
        ]).reset_index()
        
        # 只保留岗位数量>=10的数据点（避免噪音）
        monthly_stats = monthly_stats[monthly_stats['岗位数量'] >= 10]
        
        logger.info(f"有效数据点: {len(monthly_stats):,} 个")
        
        return monthly_stats
    
    def analyze_by_education(self, df):
        """按学历和职业类别交叉分析"""
        logger.info("\n分析学历×职业类别薪资...")
        
        # 提取学历
        education_levels = ['博士', '硕士', '本科', '大专', '高中', '中专']
        
        results = []
        for edu in education_levels:
            df_edu = df[df['学历要求'].str.contains(edu, na=False)]
            if len(df_edu) > 0:
                edu_stats = df_edu.groupby('occupation_category')['平均薪资'].agg([
                    ('平均薪资', 'mean'),
                    ('岗位数量', 'count')
                ])
                edu_stats['学历'] = edu
                results.append(edu_stats.reset_index())
        
        if results:
            edu_occupation_stats = pd.concat(results, ignore_index=True)
            
            # 只保留岗位数量>=5的数据
            edu_occupation_stats = edu_occupation_stats[edu_occupation_stats['岗位数量'] >= 5]
            
            logger.info(f"学历×职业类别数据点: {len(edu_occupation_stats):,} 个")
            
            # 显示部分结果
            logger.info("\n学历×职业类别薪资示例 (Top 20):")
            top_data = edu_occupation_stats.nlargest(20, '平均薪资')
            for _, row in top_data.iterrows():
                logger.info(f"  {row['学历']:6s} × {row['occupation_category']:15s} - "
                           f"平均: {row['平均薪资']:8,.0f}元 | 岗位数: {int(row['岗位数量']):5,}")
            
            return edu_occupation_stats
        
        return None
    
    def analyze_occupation_by_month(self, df):
        """按月度分析职业薪资趋势（主口径）"""
        logger.info("\n分析职业月度薪资趋势...")
        
        # 过滤有月份数据的记录
        df_month = df[df['publish_month'].notna()].copy()
        
        # 只分析Top 20职业
        top_occupations = df['occupation_core'].value_counts().head(20).index
        df_top = df_month[df_month['occupation_core'].isin(top_occupations)]
        
        logger.info(f"分析Top 20职业，有效数据: {len(df_top):,} 行")
        
        # 按职业和月份统计
        monthly_stats = df_top.groupby(['occupation_core', 'publish_month'])['平均薪资'].agg([
            ('平均薪资', 'mean'),
            ('岗位数量', 'count')
        ]).reset_index()
        
        # 只保留岗位数量>=5的数据点
        monthly_stats = monthly_stats[monthly_stats['岗位数量'] >= 5]
        
        logger.info(f"有效数据点: {len(monthly_stats):,} 个")
        
        return monthly_stats
    
    def analyze_education_by_occupation(self, df):
        """按学历和职业交叉分析（主口径）"""
        logger.info("\n分析学历×职业薪资...")
        
        # 只分析Top 15职业
        top_occupations = df['occupation_core'].value_counts().head(15).index
        df_top = df[df['occupation_core'].isin(top_occupations)]
        
        logger.info(f"分析Top 15职业，有效数据: {len(df_top):,} 行")
        
        # 提取学历
        education_levels = ['博士', '硕士', '本科', '大专']
        
        results = []
        for edu in education_levels:
            df_edu = df_top[df_top['学历要求'].str.contains(edu, na=False)]
            if len(df_edu) > 0:
                edu_stats = df_edu.groupby('occupation_core')['平均薪资'].agg([
                    ('平均薪资', 'mean'),
                    ('岗位数量', 'count')
                ])
                edu_stats['学历'] = edu
                results.append(edu_stats.reset_index())
        
        if results:
            edu_occupation_stats = pd.concat(results, ignore_index=True)
            
            # 只保留岗位数量>=3的数据
            edu_occupation_stats = edu_occupation_stats[edu_occupation_stats['岗位数量'] >= 3]
            
            logger.info(f"学历×职业数据点: {len(edu_occupation_stats):,} 个")
            
            # 显示部分结果
            logger.info("\n学历×职业薪资示例 (Top 20):")
            top_data = edu_occupation_stats.nlargest(20, '平均薪资')
            for _, row in top_data.iterrows():
                logger.info(f"  {row['学历']:6s} × {row['occupation_core']:20s} - "
                           f"平均: {row['平均薪资']:8,.0f}元 | 岗位数: {int(row['岗位数量']):5,}")
            
            return edu_occupation_stats
        
        return None
    
    def save_reports(self, category_stats, core_stats, monthly_category_stats, monthly_occupation_stats, edu_category_stats, edu_occupation_stats):
        """保存分析报告"""
        logger.info("\n保存分析报告...")
        
        report_file = self.output_dir / '职业类别薪资分析报告.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Guangdong Recruitment Data - Occupation Salary Analysis Report\n\n")
            
            f.write("## 1. Occupation Category Salary Statistics\n\n")
            for category, row in category_stats.iterrows():
                f.write(f"{category:20s} - 平均: {row['平均薪资']:10,.0f}元 | "
                       f"中位数: {row['中位数薪资']:10,.0f}元 | 岗位数: {int(row['岗位数量']):8,}\n")
            
            f.write("\n\n## 2. Occupation Core Salary Ranking (Top 50)\n\n")
            for i, (core, row) in enumerate(core_stats.head(50).iterrows(), 1):
                f.write(f"{i:3d}. {core:30s} - 平均: {row['平均薪资']:10,.0f}元 | "
                       f"中位数: {row['中位数薪资']:10,.0f}元 | 岗位数: {int(row['岗位数量']):8,}\n")
            
            if edu_category_stats is not None:
                f.write("\n\n## 3. Education x Occupation Category Salary Analysis\n\n")
                for edu in ['博士', '硕士', '本科', '大专']:
                    edu_data = edu_category_stats[edu_category_stats['学历'] == edu]
                    if len(edu_data) > 0:
                        f.write(f"\n### {edu}\n")
                        edu_data_sorted = edu_data.sort_values('平均薪资', ascending=False)
                        for _, row in edu_data_sorted.head(15).iterrows():
                            f.write(f"  {row['occupation_category']:20s} - 平均: {row['平均薪资']:10,.0f}元 | "
                                   f"岗位数: {int(row['岗位数量']):6,}\n")
            
            if edu_occupation_stats is not None:
                f.write("\n\n## 4. Education x Occupation Salary Analysis (Primary View)\n\n")
                for edu in ['博士', '硕士', '本科', '大专']:
                    edu_data = edu_occupation_stats[edu_occupation_stats['学历'] == edu]
                    if len(edu_data) > 0:
                        f.write(f"\n### {edu}\n")
                        edu_data_sorted = edu_data.sort_values('平均薪资', ascending=False)
                        for _, row in edu_data_sorted.head(15).iterrows():
                            f.write(f"  {row['occupation_core']:30s} - 平均: {row['平均薪资']:10,.0f}元 | "
                                   f"岗位数: {int(row['岗位数量']):6,}\n")
        
        logger.info(f"报告已保存: {report_file}")
        
        # 保存 CSV 数据：英文规范列名为主，中文历史文件名兼容旧汇总脚本。
        monthly_category_export = monthly_category_stats.rename(
            columns={'平均薪资': 'avg_salary', '岗位数量': 'job_count'}
        )
        write_csv_with_legacy_copy(
            monthly_category_export,
            self.output_dir,
            canonical_filename='salary_by_occupation_category_month.csv',
            legacy_filename='职业类别月度薪资数据.csv',
        )
        monthly_occupation_export = monthly_occupation_stats.rename(
            columns={'平均薪资': 'avg_salary', '岗位数量': 'job_count'}
        )
        write_csv_with_legacy_copy(
            monthly_occupation_export,
            self.output_dir,
            canonical_filename='salary_by_occupation_month.csv',
            legacy_filename='职业月度薪资数据.csv',
        )
        if edu_category_stats is not None:
            edu_category_export = edu_category_stats.rename(
                columns={'学历': 'education_level', '平均薪资': 'avg_salary', '岗位数量': 'job_count'}
            )
            write_csv_with_legacy_copy(
                edu_category_export,
                self.output_dir,
                canonical_filename='salary_by_education_occupation_category.csv',
                legacy_filename='学历职业类别薪资数据.csv',
            )
        if edu_occupation_stats is not None:
            edu_occupation_export = edu_occupation_stats.rename(
                columns={'学历': 'education_level', '平均薪资': 'avg_salary', '岗位数量': 'job_count'}
            )
            write_csv_with_legacy_copy(
                edu_occupation_export,
                self.output_dir,
                canonical_filename='salary_by_education_occupation.csv',
                legacy_filename='学历职业薪资数据.csv',
            )
        
        logger.info("数据文件已保存")
    
    def generate_visualizations(self, category_stats, monthly_stats):
        """生成可视化图表"""
        logger.info("\n生成可视化图表...")
        
        # 准备职业类别薪资数据
        category_data = []
        for category, row in category_stats.iterrows():
            category_data.append({
                'name': category,
                'value': round(row['平均薪资'], 0),
                'count': int(row['岗位数量'])
            })
        
        # 准备月度趋势数据
        monthly_data = {}
        for category in monthly_stats['occupation_category'].unique():
            cat_data = monthly_stats[monthly_stats['occupation_category'] == category]
            cat_data = cat_data.sort_values('publish_month')
            monthly_data[category] = [
                [row['publish_month'], round(row['平均薪资'], 0)]
                for _, row in cat_data.iterrows()
            ]
        
        import json
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Occupation Salary Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            background: #f5f5f5;
        }}
        h1 {{
            text-align: center;
            color: #333;
        }}
        .chart-container {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .chart {{
            width: 100%;
            height: 500px;
        }}
    </style>
</head>
<body>
    <h1>Guangdong Recruitment Data - Occupation Salary Analysis</h1>
    
    <div class="chart-container">
        <h2>Occupation Category Average Salary</h2>
        <div id="category-chart" class="chart"></div>
    </div>
    
    <div class="chart-container">
        <h2>Occupation Category Monthly Salary Trend</h2>
        <div id="trend-chart" class="chart"></div>
    </div>
    
    <script>
        // 职业类别薪资柱状图
        var categoryChart = echarts.init(document.getElementById('category-chart'));
        var categoryData = {json.dumps(category_data, ensure_ascii=False)};
        
        categoryChart.setOption({{
            title: {{
                text: '各职业类别平均薪资对比',
                left: 'center'
            }},
            tooltip: {{
                trigger: 'axis',
                axisPointer: {{
                    type: 'shadow'
                }},
                formatter: function(params) {{
                    var item = params[0];
                    return item.name + '<br/>平均薪资: ' + item.value.toLocaleString() + ' 元/月';
                }}
            }},
            xAxis: {{
                type: 'category',
                data: categoryData.map(item => item.name),
                axisLabel: {{
                    rotate: 45,
                    interval: 0
                }}
            }},
            yAxis: {{
                type: 'value',
                name: '平均薪资(元/月)',
                axisLabel: {{
                    formatter: function(value) {{
                        return (value / 1000).toFixed(0) + 'k';
                    }}
                }}
            }},
            series: [{{
                data: categoryData.map(item => item.value),
                type: 'bar',
                itemStyle: {{
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        {{ offset: 0, color: '#83bff6' }},
                        {{ offset: 1, color: '#188df0' }}
                    ])
                }},
                label: {{
                    show: true,
                    position: 'top',
                    formatter: function(params) {{
                        return (params.value / 1000).toFixed(1) + 'k';
                    }}
                }}
            }}]
        }});
        
        // 月度趋势图
        var trendChart = echarts.init(document.getElementById('trend-chart'));
        var monthlyData = {json.dumps(monthly_data, ensure_ascii=False)};
        
        var series = [];
        var colors = ['#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de', '#3ba272', '#fc8452', '#9a60b4'];
        var i = 0;
        
        for (var category in monthlyData) {{
            series.push({{
                name: category,
                type: 'line',
                data: monthlyData[category].map(item => item[1]),
                smooth: true,
                lineStyle: {{
                    width: 2
                }},
                itemStyle: {{
                    color: colors[i % colors.length]
                }}
            }});
            i++;
        }}
        
        var months = [];
        for (var category in monthlyData) {{
            if (monthlyData[category].length > months.length) {{
                months = monthlyData[category].map(item => item[0]);
            }}
        }}
        
        trendChart.setOption({{
            title: {{
                text: '职业类别月度薪资变化趋势',
                left: 'center'
            }},
            tooltip: {{
                trigger: 'axis',
                formatter: function(params) {{
                    var result = params[0].axisValue + '<br/>';
                    params.forEach(function(item) {{
                        result += item.marker + item.seriesName + ': ' + 
                                 item.value.toLocaleString() + ' 元<br/>';
                    }});
                    return result;
                }}
            }},
            legend: {{
                data: Object.keys(monthlyData),
                top: 30,
                type: 'scroll'
            }},
            grid: {{
                top: 80,
                bottom: 60
            }},
            xAxis: {{
                type: 'category',
                data: months,
                axisLabel: {{
                    rotate: 45
                }}
            }},
            yAxis: {{
                type: 'value',
                name: '平均薪资(元/月)',
                axisLabel: {{
                    formatter: function(value) {{
                        return (value / 1000).toFixed(0) + 'k';
                    }}
                }}
            }},
            series: series
        }});
        
        window.addEventListener('resize', function() {{
            categoryChart.resize();
            trendChart.resize();
        }});
    </script>
</body>
</html>
"""
        
        html_file = self.output_dir / '职业类别薪资分析图.html'
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"可视化图表已保存: {html_file}")
    
    def run(self):
        """运行完整分析"""
        logger.info("=" * 80)
        logger.info("职业与职业类别薪资分析")
        logger.info("=" * 80)
        
        # 加载数据
        df = self.load_data()
        
        # 分析
        category_stats, core_stats = self.analyze_by_occupation(df)
        monthly_category_stats = self.analyze_by_month(df)  # 职业类别×月度
        monthly_occupation_stats = self.analyze_occupation_by_month(df)  # 职业×月度（新增）
        edu_category_stats = self.analyze_by_education(df)  # 学历×职业类别
        edu_occupation_stats = self.analyze_education_by_occupation(df)  # 学历×职业（新增）
        
        # 保存报告
        self.save_reports(category_stats, core_stats, monthly_category_stats, 
                         monthly_occupation_stats, edu_category_stats, edu_occupation_stats)
        
        # 生成可视化
        self.generate_visualizations(category_stats, monthly_category_stats)
        
        logger.info("\n" + "=" * 80)
        logger.info("成功：职业与职业类别薪资分析完成!")
        logger.info("=" * 80)


def main():
    """主函数"""
    analyzer = OccupationSalaryAnalyzer()
    analyzer.run()


if __name__ == '__main__':
    main()
