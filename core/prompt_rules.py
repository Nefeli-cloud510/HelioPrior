"""Shared constraints injected into scientific LLM role prompts."""

from core.method_registry import METHOD_REGISTRY


def _build_model_parameter_guide() -> str:
    lines: list[str] = ["当前可用方法："]
    for item in METHOD_REGISTRY.catalog():
        parameter_names = "、".join(spec["name"] for spec in item["parameterSchema"])
        lines.append(
            f"- {item['displayName']}（{item['methodId']}）：可调 {parameter_names}"
        )
    return "\n".join(lines)


def _build_execution_rule() -> str:
    method_names = "、".join(METHOD_REGISTRY.display_strings())
    return (
        "本项目采用可插拔方法注册中心开展实验，当前已注册方法："
        f"{method_names}；默认主线来自注册中心，其余方法作为对照。"
        "可调项包括方法族、正则强度、特征组合、历史滞后窗口、预测超前窗口、"
        "测试集比例与随机种子等，具体以方法注册中心下发的参数表为准。"
        "你生成的所有假设、质询、不确定性、候选实验与结果解释，"
        "必须只基于数据变量库中的真实物理量，"
        "并能被当前注册方法流程实际执行、量化和比较（Pearson_r / RMSE）；"
        "必须符合物理规律，禁止发明变量、虚构数据来源或设计当前流程无法验证的机制。"
    )


MODEL_PARAMETER_GUIDE = _build_model_parameter_guide()
LINEAR_MODEL_EXECUTION_RULE = _build_execution_rule()

# 旧角色提示词仍引用原名，保留兼容别名。
ELASTIC_NET_EXECUTION_RULE = LINEAR_MODEL_EXECUTION_RULE
