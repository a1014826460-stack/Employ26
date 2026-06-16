# Freeze occupation_retrieval retrieval baseline on the v1 task-table recipe

occupation_retrieval 检索实验将当前 `v1` 方案定义为一套可复现基线配方，而不是单个已训练模型目录。该基线冻结在 `annotations.label_studio_tasks_v2` 任务主表口径上，继续使用 `annotations_completed` 与 `data_raw` 的 Python 解析结果构造训练样本，并以 `src.occupation_retrieval.eval_models_multimetric` 作为正式统一评估标准；`jsonb` 字段和 `annotations.v_label_studio_task_annotations_v2` 视图保留为后续挑战方案，而不在冻结基线时同步切换，以避免在“定义基线”的同时改动数据契约。


