"""
SQLComposer - SQL 組合器
將多個子查詢的 SQL 組合成最終的完整 SQL
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SQLFragment:
    """SQL 片段"""
    id: str
    sql: str
    tables_used: List[str]
    columns_used: List[str]
    is_subquery: bool = False


class SQLComposer:
    """
    SQL 組合器

    將分解後生成的子 SQL 組合成完整的 SQL 查詢。
    """

    def __init__(self, schema: Dict[str, Any], composition_order: str = "filter_join_agg_select"):
        """
        初始化組合器

        Args:
            schema: 資料庫 schema 資訊
            composition_order: 組合順序策略
        """
        self.schema = schema
        self.composition_order = composition_order

    def compose(self, fragments: List[SQLFragment]) -> str:
        """
        組合 SQL 片段

        Args:
            fragments: SQL 片段列表

        Returns:
            完整的 SQL 查詢
        """
        if not fragments:
            raise ValueError("No SQL fragments to compose")

        if len(fragments) == 1:
            return fragments[0].sql

        # 根據組合順序策略組合
        if self.composition_order == "filter_join_agg_select":
            return self._compose_filter_join_agg(fragments)
        elif self.composition_order == "subquery_first":
            return self._compose_subquery_first(fragments)
        else:
            return self._compose_default(fragments)

    def _compose_filter_join_agg(self, fragments: List[SQLFragment]) -> str:
        """按 篩選 → JOIN → 聚合 → SELECT 順序組合"""
        # 分類片段
        filter_fragments = [f for f in fragments if "WHERE" in f.sql.upper()]
        join_fragments = [f for f in fragments if "JOIN" in f.sql.upper()]
        agg_fragments = [f for f in fragments if any(
            agg in f.sql.upper() for agg in ["SUM", "AVG", "COUNT", "MAX", "MIN", "GROUP BY"]
        )]
        other_fragments = [f for f in fragments if f not in filter_fragments + join_fragments + agg_fragments]

        # 組合
        # TODO: 實作更智能的組合邏輯
        all_sql = [f.sql for f in fragments]
        return self._merge_sql_statements(all_sql)

    def _compose_subquery_first(self, fragments: List[SQLFragment]) -> str:
        """子查詢優先組合"""
        subquery_fragments = [f for f in fragments if f.is_subquery]
        main_fragments = [f for f in fragments if not f.is_subquery]

        # 將子查詢嵌入主查詢
        # TODO: 實作子查詢嵌入邏輯
        all_sql = [f.sql for f in subquery_fragments + main_fragments]
        return self._merge_sql_statements(all_sql)

    def _compose_default(self, fragments: List[SQLFragment]) -> str:
        """預設組合方式"""
        all_sql = [f.sql for f in fragments]
        return self._merge_sql_statements(all_sql)

    def _merge_sql_statements(self, sql_list: List[str]) -> str:
        """
        合併多個 SQL 語句

        這是一個簡化版本，實際應用需要更複雜的 SQL 解析和合併邏輯
        """
        if len(sql_list) == 1:
            return sql_list[0]

        # TODO: 實作智能 SQL 合併
        # 目前簡單地使用 CTE (Common Table Expression) 方式組合
        cte_parts = []
        for i, sql in enumerate(sql_list[:-1]):
            cte_name = f"sub_{i}"
            # 移除結尾的分號
            clean_sql = sql.rstrip(";").strip()
            cte_parts.append(f"{cte_name} AS (\n  {clean_sql}\n)")

        main_sql = sql_list[-1].rstrip(";").strip()

        if cte_parts:
            return f"WITH {', '.join(cte_parts)}\n{main_sql};"
        else:
            return f"{main_sql};"

    def validate_sql(self, sql: str) -> Dict[str, Any]:
        """
        驗證 SQL 語法

        Args:
            sql: SQL 查詢

        Returns:
            驗證結果
        """
        result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
        }

        # 基本檢查
        sql_upper = sql.upper()

        # 檢查是否有 SELECT
        if "SELECT" not in sql_upper:
            result["is_valid"] = False
            result["errors"].append("Missing SELECT clause")

        # 檢查括號匹配
        if sql.count("(") != sql.count(")"):
            result["is_valid"] = False
            result["errors"].append("Unmatched parentheses")

        # 檢查引號匹配
        if sql.count("'") % 2 != 0:
            result["is_valid"] = False
            result["errors"].append("Unmatched quotes")

        # 警告：可能的問題
        if "SELECT *" in sql_upper:
            result["warnings"].append("Using SELECT * may impact performance")

        if "WHERE" not in sql_upper and "LIMIT" not in sql_upper:
            result["warnings"].append("Query without WHERE or LIMIT may return large result set")

        return result


# 測試代碼
if __name__ == "__main__":
    schema = {
        "tables": ["employees", "departments"],
    }

    composer = SQLComposer(schema)

    # 測試單一片段
    fragment1 = SQLFragment(
        id="main",
        sql="SELECT * FROM employees",
        tables_used=["employees"],
        columns_used=["*"],
    )
    result = composer.compose([fragment1])
    print(f"單一片段組合結果：\n{result}\n")

    # 測試多片段組合
    fragment2 = SQLFragment(
        id="sub1",
        sql="SELECT dept_id, AVG(salary) as avg_salary FROM employees GROUP BY dept_id",
        tables_used=["employees"],
        columns_used=["dept_id", "salary"],
        is_subquery=True,
    )
    fragment3 = SQLFragment(
        id="main",
        sql="SELECT d.name, s.avg_salary FROM departments d JOIN sub1 s ON d.id = s.dept_id ORDER BY s.avg_salary DESC LIMIT 1",
        tables_used=["departments", "sub1"],
        columns_used=["name", "avg_salary"],
    )
    result = composer.compose([fragment2, fragment3])
    print(f"多片段組合結果：\n{result}\n")

    # 測試驗證
    validation = composer.validate_sql(result)
    print(f"驗證結果：{validation}")
