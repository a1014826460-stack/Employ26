# 职业细类匹配 考核

## 考核要求

## 文件说明

| 文件 | 用途 |
|------|------|
| `data.json` | 测试数据，需预测的 495 个岗位 |
| `train_data.json` | 训练数据，6,098 条含标签，供训练模型 |
| `occupation_dictionary.json` | 中国职业大典--职业候选 |

## 数据格式

`data.json` 每条记录包含岗位名称、岗位要求、5 个候选职业（代码+名称+定义）：

```
{
  "task_id": "46437",
  "job_title": "医学信息专员",
  "job_requirements": "1、医学、药学相关专业...",
  "candidates": [
    {"letter": "A", "code": "2-05-01-18", "title": "药学技术人员", "desc": "从事药物..."},
    ...
  ]
}
```

`train_data.json` 格式相同，额外包含 `"label": "B"` 字段。

## 提交格式

对 `data.json` 中每条数据预测一个候选，输出 JSON 数组：

```json
[
  {"task_id": "46437", "prediction": "B"},
  ...
]
```

`prediction` 可以是 "A"、"B"、"C"、"D"、"E"、"NONE" 之一。
如果为"NONE"，请给出你的候选职业，该职业必须来源于"occupation_dictionary.json"

### 截止时间
2026/5/31 24:00:00

### 提交方式
请将你的结果，上传到邮箱：1014826460@qq.com

TITLE：你的姓名 + 职业细类匹配测试集
