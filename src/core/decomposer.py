"""
QueryDecomposer - 查詢分解器
將複雜的自然語言查詢分解為多個子查詢
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import json


class DecompositionRule(Enum):
    """分解規則類型"""
    SPLIT_BY_JOIN = "split_by_join"           # 按 JOIN 拆分
    SPLIT_BY_AGGREGATION = "split_by_agg"     # 按聚合拆分
    EXTRACT_SUBQUERY = "extract_subquery"      # 提取子查詢
    FILTER_FIRST = "filter_first"              # 先篩選
    COMPUTE_THEN_JOIN = "compute_then_join"    # 先計算再 JOIN


class CompositionOrder(Enum):
    """組合順序"""
    FILTER_JOIN_AGG_SELECT = "filter_join_agg_select"
    JOIN_FILTER_AGG_SELECT = "join_filter_agg_select"
    AGG_FIRST = "agg_first"
    SUBQUERY_FIRST = "subquery_first"


@dataclass
class DecompositionStrategy:
    """
    查詢分解策略 - 可進化的基因表示

    這個類別定義了「如何分解複雜查詢」的策略，
    進化演算法會優化這些參數來找到最佳策略。
    """

    # 分解規則及其權重（可進化）
    rules: Dict[str, float] = field(default_factory=lambda: {
        "split_by_join": 0.8,
        "split_by_agg": 0.6,
        "extract_subquery": 0.7,
        "filter_first": 0.9,
        "compute_then_join": 0.5,
    })

    # 組合順序（可進化）
    composition_order: str = "filter_join_agg_select"

    # 參數（可進化）
    max_decomposition_depth: int = 3      # 最大分解深度
    join_threshold: int = 2               # 幾個 JOIN 以上才分解
    agg_threshold: int = 1                # 幾個聚合以上才分解
    subquery_extraction: bool = True      # 是否提取子查詢

    # 進化元數據
    fitness_score: float = 0.0
    generation: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """序列化為字典"""
        return {
            "rules": self.rules,
            "composition_order": self.composition_order,
            "max_decomposition_depth": self.max_decomposition_depth,
            "join_threshold": self.join_threshold,
            "agg_threshold": self.agg_threshold,
            "subquery_extraction": self.subquery_extraction,
            "fitness_score": self.fitness_score,
            "generation": self.generation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecompositionStrategy":
        """從字典反序列化"""
        return cls(**data)

    def to_json(self) -> str:
        """序列化為 JSON"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "DecompositionStrategy":
        """從 JSON 反序列化"""
        return cls.from_dict(json.loads(json_str))

    def clone(self) -> "DecompositionStrategy":
        """複製策略"""
        return DecompositionStrategy.from_dict(self.to_dict())


@dataclass
class SubQuery:
    """子查詢結構"""
    id: str                          # 子查詢 ID
    question: str                    # 自然語言子問題
    depends_on: List[str] = field(default_factory=list)  # 依賴的其他子查詢
    sql: Optional[str] = None        # 生成的 SQL（初始為空）
    result: Optional[Any] = None     # 執行結果


class QueryDecomposer:
    """
    查詢分解器

    根據分解策略，將複雜的自然語言查詢分解為多個子查詢。
    """

    def __init__(self, strategy: DecompositionStrategy, schema: Dict[str, Any]):
        """
        初始化分解器

        Args:
            strategy: 分解策略
            schema: 資料庫 schema 資訊
        """
        self.strategy = strategy
        self.schema = schema

    def analyze_complexity(self, question: str) -> Dict[str, Any]:
        """
        分析查詢複雜度

        Args:
            question: 自然語言問題

        Returns:
            複雜度分析結果
        """
        # TODO: 實作複雜度分析
        # - 估計需要的 JOIN 數量
        # - 估計聚合函數數量
        # - 檢測子查詢需求
        # - 分析篩選條件

        complexity = {
            "estimated_joins": 0,
            "estimated_aggregations": 0,
            "has_subquery": False,
            "has_time_filter": False,
            "has_comparison": False,
            "complexity_score": 0.0,
        }

        # 簡單的關鍵字檢測（實際應用需要更複雜的 NLP）
        question_lower = question.lower()

        # 檢測 JOIN 相關詞彙
        join_keywords = ["和", "與", "以及", "關聯", "連接", "包含", "屬於"]
        for kw in join_keywords:
            if kw in question_lower:
                complexity["estimated_joins"] += 1

        # 檢測聚合相關詞彙
        agg_keywords = ["總共", "平均", "最多", "最少", "數量", "總和", "統計"]
        for kw in agg_keywords:
            if kw in question_lower:
                complexity["estimated_aggregations"] += 1

        # 檢測子查詢相關詞彙
        subquery_keywords = ["其中", "當中", "符合條件的"]
        for kw in subquery_keywords:
            if kw in question_lower:
                complexity["has_subquery"] = True

        # 計算複雜度分數
        complexity["complexity_score"] = (
            complexity["estimated_joins"] * 0.4 +
            complexity["estimated_aggregations"] * 0.3 +
            (1 if complexity["has_subquery"] else 0) * 0.3
        )

        return complexity

    def decompose(self, question: str) -> List[SubQuery]:
        """
        分解查詢

        Args:
            question: 自然語言問題

        Returns:
            子查詢列表
        """
        # 分析複雜度
        complexity = self.analyze_complexity(question)

        # 如果複雜度低於閾值，不分解
        if (complexity["estimated_joins"] < self.strategy.join_threshold and
            complexity["estimated_aggregations"] < self.strategy.agg_threshold):
            return [SubQuery(id="main", question=question)]

        # 根據策略分解
        sub_queries = []

        # 應用分解規則
        if (self.strategy.rules.get("filter_first", 0) > 0.5 and
            complexity.get("has_time_filter") or "篩選" in question):
            # 先提取篩選條件
            sub_queries.append(SubQuery(
                id="filter",
                question=self._extract_filter_question(question),
            ))

        if (self.strategy.rules.get("split_by_join", 0) > 0.5 and
            complexity["estimated_joins"] >= self.strategy.join_threshold):
            # 按 JOIN 拆分
            join_questions = self._split_by_join(question)
            for i, jq in enumerate(join_questions):
                sub_queries.append(SubQuery(
                    id=f"join_{i}",
                    question=jq,
                    depends_on=[sq.id for sq in sub_queries],
                ))

        if (self.strategy.rules.get("split_by_agg", 0) > 0.5 and
            complexity["estimated_aggregations"] >= self.strategy.agg_threshold):
            # 按聚合拆分
            sub_queries.append(SubQuery(
                id="aggregate",
                question=self._extract_aggregation_question(question),
                depends_on=[sq.id for sq in sub_queries],
            ))

        # 如果沒有分解出子查詢，返回原始問題
        if not sub_queries:
            return [SubQuery(id="main", question=question)]

        return sub_queries

    def _extract_filter_question(self, question: str) -> str:
        """提取篩選相關的子問題"""
        # TODO: 實作更智能的提取邏輯
        return f"篩選條件：{question}"

    def _split_by_join(self, question: str) -> List[str]:
        """按 JOIN 拆分問題"""
        # TODO: 實作更智能的拆分邏輯
        return [question]

    def _extract_aggregation_question(self, question: str) -> str:
        """提取聚合相關的子問題"""
        # TODO: 實作更智能的提取邏輯
        return f"計算：{question}"


# 測試代碼
if __name__ == "__main__":
    # 建立預設策略
    strategy = DecompositionStrategy()
    print("預設策略：")
    print(strategy.to_json())

    # 模擬 schema
    schema = {
        "tables": ["employees", "departments", "salaries"],
        "columns": {
            "employees": ["id", "name", "dept_id"],
            "departments": ["id", "name"],
            "salaries": ["employee_id", "amount", "date"],
        }
    }

    # 建立分解器
    decomposer = QueryDecomposer(strategy, schema)

    # 測試複雜度分析
    question = "哪個部門的員工平均薪資最高？"
    complexity = decomposer.analyze_complexity(question)
    print(f"\n問題：{question}")
    print(f"複雜度分析：{complexity}")

    # 測試分解
    sub_queries = decomposer.decompose(question)
    print(f"\n分解結果：")
    for sq in sub_queries:
        print(f"  - {sq.id}: {sq.question}")
