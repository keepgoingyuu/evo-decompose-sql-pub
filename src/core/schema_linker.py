"""
SchemaLinker - Schema 連結器
將自然語言中的實體連結到資料庫 schema
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class SchemaMatch:
    """Schema 匹配結果"""
    mention: str              # 自然語言中的提及
    table: str               # 匹配的表名
    column: str = None       # 匹配的欄位名（可選）
    confidence: float = 0.0  # 匹配信心度


class SchemaLinker:
    """
    Schema 連結器

    負責將自然語言查詢中的實體連結到資料庫 schema 中的表和欄位。
    這是 Text-to-SQL 中最容易出錯的部分（佔 33% 的錯誤）。
    """

    def __init__(self, schema: Dict[str, Any]):
        """
        初始化 Schema 連結器

        Args:
            schema: 資料庫 schema 資訊，格式如下：
                {
                    "tables": {
                        "table_name": {
                            "columns": ["col1", "col2", ...],
                            "primary_key": "id",
                            "foreign_keys": [{"column": "fk_col", "references": "other_table.id"}],
                            "description": "表描述",
                            "column_descriptions": {"col1": "欄位描述", ...}
                        }
                    }
                }
        """
        self.schema = schema
        self._build_index()

    def _build_index(self):
        """建立搜尋索引"""
        self.table_index = {}
        self.column_index = {}

        tables = self.schema.get("tables", {})
        for table_name, table_info in tables.items():
            # 索引表名
            self.table_index[table_name.lower()] = table_name

            # 索引欄位
            columns = table_info.get("columns", [])
            for col in columns:
                key = col.lower()
                if key not in self.column_index:
                    self.column_index[key] = []
                self.column_index[key].append((table_name, col))

    def link(self, question: str) -> List[SchemaMatch]:
        """
        執行 schema 連結

        Args:
            question: 自然語言問題

        Returns:
            匹配結果列表
        """
        matches = []
        question_lower = question.lower()

        # 嘗試匹配表名
        for table_key, table_name in self.table_index.items():
            if table_key in question_lower:
                matches.append(SchemaMatch(
                    mention=table_key,
                    table=table_name,
                    confidence=0.9 if table_key == table_name.lower() else 0.7,
                ))

        # 嘗試匹配欄位名
        for col_key, col_refs in self.column_index.items():
            if col_key in question_lower:
                for table_name, col_name in col_refs:
                    matches.append(SchemaMatch(
                        mention=col_key,
                        table=table_name,
                        column=col_name,
                        confidence=0.8,
                    ))

        # 使用同義詞匹配
        synonym_matches = self._match_synonyms(question)
        matches.extend(synonym_matches)

        # 去重和排序
        matches = self._deduplicate_matches(matches)
        matches.sort(key=lambda x: x.confidence, reverse=True)

        return matches

    def _match_synonyms(self, question: str) -> List[SchemaMatch]:
        """使用同義詞匹配"""
        # 常見的業務術語到 schema 的映射
        synonyms = {
            # 員工相關
            "員工": [("employees", None)],
            "人員": [("employees", None)],
            "職員": [("employees", None)],
            "薪水": [("salaries", "amount"), ("employees", "salary")],
            "薪資": [("salaries", "amount"), ("employees", "salary")],
            "工資": [("salaries", "amount"), ("employees", "salary")],

            # 部門相關
            "部門": [("departments", None)],
            "單位": [("departments", None)],

            # 產品相關
            "商品": [("products", None)],
            "產品": [("products", None)],
            "品項": [("products", None)],
            "價格": [("products", "price"), ("products", "unit_price")],
            "售價": [("products", "price"), ("products", "unit_price")],

            # 交易相關
            "交易": [("transactions", None)],
            "訂單": [("orders", None), ("transactions", None)],
            "銷售": [("sales", None), ("transactions", None)],
            "營業額": [("transactions", "total_amount"), ("sales", "amount")],

            # 客戶相關
            "客戶": [("customers", None)],
            "顧客": [("customers", None)],
            "會員": [("customers", None), ("members", None)],

            # 庫存相關
            "庫存": [("inventory", None), ("products", "stock")],
            "存貨": [("inventory", None)],
        }

        matches = []
        question_lower = question.lower()

        for term, mappings in synonyms.items():
            if term in question_lower:
                for table, column in mappings:
                    # 檢查表是否存在於 schema
                    if table in self.table_index.values():
                        matches.append(SchemaMatch(
                            mention=term,
                            table=table,
                            column=column,
                            confidence=0.7,
                        ))

        return matches

    def _deduplicate_matches(self, matches: List[SchemaMatch]) -> List[SchemaMatch]:
        """去除重複匹配"""
        seen = set()
        unique_matches = []

        for match in matches:
            key = (match.table, match.column)
            if key not in seen:
                seen.add(key)
                unique_matches.append(match)

        return unique_matches

    def get_relevant_tables(self, question: str) -> List[str]:
        """
        取得相關的表名列表

        Args:
            question: 自然語言問題

        Returns:
            相關表名列表
        """
        matches = self.link(question)
        tables = list(set(m.table for m in matches))
        return tables

    def get_relevant_columns(self, question: str, table: str = None) -> List[Tuple[str, str]]:
        """
        取得相關的欄位列表

        Args:
            question: 自然語言問題
            table: 限定的表名（可選）

        Returns:
            (表名, 欄位名) 元組列表
        """
        matches = self.link(question)
        columns = [
            (m.table, m.column)
            for m in matches
            if m.column is not None and (table is None or m.table == table)
        ]
        return list(set(columns))


# 測試代碼
if __name__ == "__main__":
    # 模擬 schema
    schema = {
        "tables": {
            "employees": {
                "columns": ["id", "name", "dept_id", "salary", "hire_date"],
                "primary_key": "id",
            },
            "departments": {
                "columns": ["id", "name", "manager_id"],
                "primary_key": "id",
            },
            "salaries": {
                "columns": ["id", "employee_id", "amount", "effective_date"],
                "primary_key": "id",
            },
        }
    }

    linker = SchemaLinker(schema)

    # 測試案例
    test_questions = [
        "哪個部門的員工平均薪資最高？",
        "列出所有員工的名字和薪水",
        "找出薪水超過 50000 的員工",
        "統計每個部門的人數",
    ]

    for question in test_questions:
        print(f"\n問題：{question}")
        matches = linker.link(question)
        print("匹配結果：")
        for m in matches:
            print(f"  - {m.mention} → {m.table}.{m.column or '*'} (信心度: {m.confidence:.2f})")

        tables = linker.get_relevant_tables(question)
        print(f"相關表：{tables}")
