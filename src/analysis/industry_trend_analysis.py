"""
行业景气度分析模块。

用途:
- 基于 PostgreSQL `public.recruitment_jobs_normalized` 统计城市 × 行业 × 月度招聘量变化。
- 输出 Markdown 报告、CSV 明细和 HTML 图表，用于观察行业热度与城市行业结构。

前置依赖:
- 先完成统一招聘规范层回填；城市、行业和月份字段由结构化统计数据源统一标准化。

关键输入字段:
- `publish_month`
- `city_clean`
- `industry_clean`

输出文件:
- `output/reports/structured_analysis_{mm-dd}/city_industry_monthly_jobs.csv`
- `output/reports/structured_analysis_{mm-dd}/industry_monthly_jobs.csv`
- `output/reports/行业景气度分析报告.md`
- `output/reports/行业景气度分析图.html`

运行方式:
- `python -m src.analysis.industry_trend_analysis`

维护说明:
- 当前脚本属于新版分析链路，城市和行业标准化字段由 PostgreSQL 结构化统计数据源统一补齐。
"""

import logging
from pathlib import Path

import pandas as pd

from src.analysis.structured_common import build_structured_output_dir, write_csv_with_legacy_copy
from src.analysis.structured_pg_source import load_structured_analysis_dataframe

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IndustryTrendAnalyzer:
    """行业景气度分析器"""
    
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
        
        logger.info("行业景气度分析器初始化完成")
    
    def load_data(self):
        """加载 PostgreSQL 结构化统计主输入。"""
        logger.info("从 PostgreSQL 加载结构化统计主输入...")

        df = load_structured_analysis_dataframe()
        logger.info(f"总数据: {len(df):,} 行")
        
        # 过滤有效数据
        df_valid = df[
            (df['publish_month'].notna()) &
            (df['city_clean'].notna()) &
            (df['industry_clean'].notna())
        ].copy()
        
        logger.info(f"有效数据: {len(df_valid):,} 行 ({len(df_valid)/len(df)*100:.1f}%)")
        
        return df_valid
    
    def analyze_city_industry_trend(self, df):
        """分析城市×行业月度趋势"""
        logger.info("\n分析城市×行业月度招聘量...")
        
        # 按城市、行业、月份统计
        trend_stats = df.groupby(['city_clean', 'industry_clean', 'publish_month']).size().reset_index(name='job_count')
        
        # 只保留招聘量>=5的数据点
        trend_stats = trend_stats[trend_stats['job_count'] >= 5]
        
        logger.info(f"有效数据点: {len(trend_stats):,} 个")
        
        # 统计各城市的主要行业
        city_industry_total = df.groupby(['city_clean', 'industry_clean']).size().reset_index(name='total_jobs')
        city_industry_total = city_industry_total.sort_values(['city_clean', 'total_jobs'], ascending=[True, False])
        
        logger.info("\n各城市Top 5行业:")
        for city in city_industry_total['city_clean'].unique()[:10]:
            city_data = city_industry_total[city_industry_total['city_clean'] == city].head(5)
            logger.info(f"\n{city}:")
            for _, row in city_data.iterrows():
                logger.info(f"  {row['industry_clean']:30s}: {row['total_jobs']:6,} 个岗位")
        
        return trend_stats, city_industry_total
    
    def analyze_industry_monthly(self, df):
        """分析行业整体月度趋势"""
        logger.info("\n分析行业整体月度趋势...")
        
        # 按行业和月份统计
        industry_monthly = df.groupby(['industry_clean', 'publish_month']).size().reset_index(name='job_count')
        
        # 只保留招聘量>=10的数据点
        industry_monthly = industry_monthly[industry_monthly['job_count'] >= 10]
        industry_monthly = industry_monthly.sort_values(['industry_clean', 'publish_month']).copy()
        industry_monthly['previous_month_job_count'] = (
            industry_monthly.groupby('industry_clean')['job_count'].shift(1)
        )
        industry_monthly['month_over_month_change'] = (
            industry_monthly['job_count'] - industry_monthly['previous_month_job_count']
        )
        industry_monthly['month_over_month_growth_rate'] = (
            industry_monthly['month_over_month_change'] / industry_monthly['previous_month_job_count']
        )
        
        # 找出Top行业
        industry_total = df.groupby('industry_clean').size().sort_values(ascending=False)
        
        logger.info("\nTop 20 行业总招聘量:")
        for i, (industry, count) in enumerate(industry_total.head(20).items(), 1):
            logger.info(f"  {i:2d}. {industry:35s}: {count:8,} 个岗位")
        
        return industry_monthly, industry_total
    
    def save_reports(self, trend_stats, city_industry_total, industry_monthly, industry_total):
        """保存分析报告"""
        logger.info("\n保存分析报告...")
        
        report_file = self.output_dir / '行业景气度分析报告.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Guangdong Recruitment Data - Industry Trend Analysis Report\n\n")
            
            f.write("## 1. Overall Industry Hiring Volume Ranking (Top 30)\n\n")
            for i, (industry, count) in enumerate(industry_total.head(30).items(), 1):
                f.write(f"{i:3d}. {industry:40s}: {count:10,} 个岗位\n")
            
            f.write("\n\n## 2. Major Industry Distribution by City\n\n")
            for city in sorted(city_industry_total['city_clean'].unique()):
                city_data = city_industry_total[city_industry_total['city_clean'] == city].head(10)
                f.write(f"\n{city}:\n")
                for i, (_, row) in enumerate(city_data.iterrows(), 1):
                    f.write(f"  {i:2d}. {row['industry_clean']:35s}: {row['total_jobs']:8,} 个岗位\n")
        
        logger.info(f"报告已保存: {report_file}")
        
        # 保存 CSV 数据：英文规范文件名为主，中文历史文件名兼容旧汇总脚本。
        write_csv_with_legacy_copy(
            trend_stats,
            self.output_dir,
            canonical_filename='city_industry_monthly_jobs.csv',
            legacy_filename='城市行业月度数据.csv',
        )
        write_csv_with_legacy_copy(
            industry_monthly,
            self.output_dir,
            canonical_filename='industry_monthly_jobs.csv',
            legacy_filename='行业月度数据.csv',
        )
        
        logger.info("数据文件已保存")
    
    def generate_visualizations(self, industry_monthly, industry_total):
        """生成可视化图表"""
        logger.info("\n生成可视化图表...")
        
        # 选择Top 15行业
        top_industries = industry_total.head(15).index.tolist()
        
        # 准备月度趋势数据
        monthly_data = {}
        for industry in top_industries:
            ind_data = industry_monthly[industry_monthly['industry_clean'] == industry]
            ind_data = ind_data.sort_values('publish_month')
            monthly_data[industry] = [
                [row['publish_month'], int(row['job_count'])]
                for _, row in ind_data.iterrows()
            ]
        
        # 准备行业总量数据
        industry_bar_data = [
            {'name': industry, 'value': int(count)}
            for industry, count in industry_total.head(20).items()
        ]
        
        import json
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Industry Trend Analysis</title>
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
            height: 600px;
        }}
    </style>
</head>
<body>
    <h1>Guangdong Recruitment Data - Industry Trend Analysis</h1>
    
    <div class="chart-container">
        <h2>Industry Hiring Volume Ranking (Top 20)</h2>
        <div id="industry-bar" class="chart"></div>
    </div>
    
    <div class="chart-container">
        <h2>Main Industry Monthly Hiring Trend (Top 15)</h2>
        <div id="industry-trend" class="chart"></div>
    </div>
    
    <script>
        // 行业招聘量柱状图
        var barChart = echarts.init(document.getElementById('industry-bar'));
        var industryData = {json.dumps(industry_bar_data, ensure_ascii=False)};
        
        barChart.setOption({{
            title: {{
                text: '各行业总招聘量对比',
                left: 'center'
            }},
            tooltip: {{
                trigger: 'axis',
                axisPointer: {{
                    type: 'shadow'
                }},
                formatter: function(params) {{
                    var item = params[0];
                    return item.name + '<br/>招聘量: ' + item.value.toLocaleString() + ' 个岗位';
                }}
            }},
            grid: {{
                left: '3%',
                right: '4%',
                bottom: '15%',
                containLabel: true
            }},
            xAxis: {{
                type: 'category',
                data: industryData.map(item => item.name),
                axisLabel: {{
                    rotate: 45,
                    interval: 0,
                    fontSize: 11
                }}
            }},
            yAxis: {{
                type: 'value',
                name: '招聘量(个)'
            }},
            series: [{{
                data: industryData.map(item => item.value),
                type: 'bar',
                itemStyle: {{
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        {{ offset: 0, color: '#91cc75' }},
                        {{ offset: 1, color: '#5470c6' }}
                    ])
                }},
                label: {{
                    show: true,
                    position: 'top',
                    fontSize: 10,
                    formatter: function(params) {{
                        return params.value.toLocaleString();
                    }}
                }}
            }}]
        }});
        
        // 行业月度趋势图
        var trendChart = echarts.init(document.getElementById('industry-trend'));
        var monthlyData = {json.dumps(monthly_data, ensure_ascii=False)};
        
        var series = [];
        var colors = ['#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de', '#3ba272', 
                     '#fc8452', '#9a60b4', '#ea7ccc', '#5470c6', '#91cc75', '#fac858',
                     '#ee6666', '#73c0de', '#3ba272'];
        var i = 0;
        
        for (var industry in monthlyData) {{
            series.push({{
                name: industry,
                type: 'line',
                data: monthlyData[industry].map(item => item[1]),
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
        for (var industry in monthlyData) {{
            if (monthlyData[industry].length > months.length) {{
                months = monthlyData[industry].map(item => item[0]);
            }}
        }}
        
        trendChart.setOption({{
            title: {{
                text: '主要行业月度招聘量变化趋势',
                left: 'center'
            }},
            tooltip: {{
                trigger: 'axis',
                formatter: function(params) {{
                    var result = params[0].axisValue + '<br/>';
                    params.forEach(function(item) {{
                        result += item.marker + item.seriesName + ': ' + 
                                 item.value.toLocaleString() + ' 个岗位<br/>';
                    }});
                    return result;
                }}
            }},
            legend: {{
                data: Object.keys(monthlyData),
                top: 30,
                type: 'scroll',
                pageButtonPosition: 'end'
            }},
            grid: {{
                top: 80,
                bottom: 80,
                left: '3%',
                right: '4%',
                containLabel: true
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
                name: '招聘量(个)'
            }},
            series: series
        }});
        
        window.addEventListener('resize', function() {{
            barChart.resize();
            trendChart.resize();
        }});
    </script>
</body>
</html>
"""
        
        html_file = self.output_dir / '行业景气度分析图.html'
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"可视化图表已保存: {html_file}")
    
    def run(self):
        """运行完整分析"""
        logger.info("=" * 80)
        logger.info("行业景气度分析")
        logger.info("=" * 80)
        
        # 加载数据
        df = self.load_data()
        
        # 分析
        trend_stats, city_industry_total = self.analyze_city_industry_trend(df)
        industry_monthly, industry_total = self.analyze_industry_monthly(df)
        
        # 保存报告
        self.save_reports(trend_stats, city_industry_total, industry_monthly, industry_total)
        
        # 生成可视化
        self.generate_visualizations(industry_monthly, industry_total)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ 行业景气度分析完成!")
        logger.info("=" * 80)


def main():
    """主函数"""
    analyzer = IndustryTrendAnalyzer()
    analyzer.run()


if __name__ == '__main__':
    main()
