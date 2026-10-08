from pydantic import BaseModel, Field
from typing import Literal, Optional


class ExperimentConfig(BaseModel):
    """
    Qwen可以调整的实验参数。
    只包含原JSON中Qwen能动的字段。
    """
    window_size: int = Field(
        default=3,
        ge=1, le=196,
        description="滑动窗口天数"
    )
    use_lag_feature: bool = Field(
        default=False,
        description="是否使用时滞特征"
    )
    max_lag_day: int = Field(
        default=3,
        ge=0, le=10,
        description="最大时滞天数"
    )
    alpha: float = Field(
        default=1.0,
        ge=0.0, le=2000.0,
        description="线性模型正则化强度（Ridge/ElasticNet）"
    )
    model_family: Literal["ridge", "elasticnet"] = Field(
        default="ridge",
        description="默认使用 Ridge，ElasticNet 作为对照"
    )
    ridge_solver: str = Field(
        default="svd",
        description="Ridge 求解器（默认 svd，与原实验脚本一致）"
    )
    fit_intercept: bool = Field(
        default=True,
        description="线性模型是否拟合截距"
    )
    alpha_grid: list[float] = Field(
        default_factory=lambda: [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        description="Ridge 参考方法的 alpha 搜索网格"
    )
    l1_ratio: float = Field(
        default=0.5,
        ge=0.0, le=1.0,
        description="ElasticNet 对照模型的 L1/L2 混合比例"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Qwen的调整理由（仅供记录，不参与执行）"
    )
