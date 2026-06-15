"""
FitnessEvaluator - 適應度評估器
評估分解策略的效果
"""

from typing import List, Dict, Any, Callable
from dataclasses import dataclass
import asyncio


@dataclass
class EvaluationResult:
    """評估結果"""
    strategy_id: str
    execution_accuracy: float     # 執行準確率
    exact_match: float           # 完全匹配率
    partial_match: float         # 部分匹配率
    avg_execution_time: float    # 平均執行時間
    total_queries: int           # 總查詢數
    correct_queries: int         # 正確查詢數
    error_count: int             # 錯誤數
    fitness_score: float         # 最終適應度分數


class FitnessEvaluator:
    """
    適應度評估器

    評估查詢分解策略在驗證集上的表現。
    這是進化演算法的核心組件，決定哪些策略會被保留。
    """

    def __init__(
        self,
        validation_set: List[Dict[str, Any]],
        llm_client: Any,
        db_executor: Callable,
        weights: Dict[str, float] = None,
    ):
        """
        初始化評估器

        Args:
            validation_set: 驗證集，格式 [{"question": str, "gold_sql": str, "db_id": str}, ...]
            llm_client: LLM 客戶端（用於生成 SQL）
            db_executor: 資料庫執行器函數
            weights: 評估指標權重
        """
        self.validation_set = validation_set
        self.llm_client = llm_client
        self.db_executor = db_executor
        self.weights = weights or {
            "execution_accuracy": 0.6,
            "exact_match": 0.2,
            "partial_match": 0.1,
            "speed_bonus": 0.1,
        }

    def evaluate(self, strategy: Any, decomposer_class: type, composer_class: type) -> EvaluationResult:
        """
        評估策略

        Args:
            strategy: 分解策略
            decomposer_class: 分解器類別
            composer_class: 組合器類別

        Returns:
            評估結果
        """
        correct = 0
        exact_matches = 0
        partial_matches = 0
        errors = 0
        total_time = 0

        for item in self.validation_set:
            question = item["question"]
            gold_sql = item["gold_sql"]
            db_id = item.get("db_id", "default")
            schema = item.get("schema", {})

            try:
                import time
                start_time = time.time()

                # 1. 使用策略分解問題
                decomposer = decomposer_class(strategy, schema)
                sub_queries = decomposer.decompose(question)

                # 2. 為每個子查詢生成 SQL
                sub_sqls = []
                for sq in sub_queries:
                    # 使用 LLM 生成 SQL
                    generated_sql = self._generate_sql(sq.question, schema)
                    sub_sqls.append(generated_sql)

                # 3. 組合子 SQL
                composer = composer_class(schema, strategy.composition_order)
                from ..core.composer import SQLFragment
                fragments = [
                    SQLFragment(id=f"sub_{i}", sql=sql, tables_used=[], columns_used=[])
                    for i, sql in enumerate(sub_sqls)
                ]
                final_sql = composer.compose(fragments)

                end_time = time.time()
                total_time += (end_time - start_time)

                # 4. 評估結果
                is_correct, match_type = self._evaluate_sql(final_sql, gold_sql, db_id)

                if is_correct:
                    correct += 1
                if match_type == "exact":
                    exact_matches += 1
                elif match_type == "partial":
                    partial_matches += 1

            except Exception as e:
                errors += 1
                print(f"Error evaluating query: {e}")

        # 計算指標
        total = len(self.validation_set)
        execution_accuracy = correct / total if total > 0 else 0
        exact_match_rate = exact_matches / total if total > 0 else 0
        partial_match_rate = partial_matches / total if total > 0 else 0
        avg_time = total_time / total if total > 0 else 0

        # 計算最終適應度分數
        fitness_score = (
            self.weights["execution_accuracy"] * execution_accuracy +
            self.weights["exact_match"] * exact_match_rate +
            self.weights["partial_match"] * partial_match_rate +
            self.weights["speed_bonus"] * max(0, 1 - avg_time / 10)  # 10秒以內有速度獎勵
        )

        return EvaluationResult(
            strategy_id=getattr(strategy, "id", "unknown"),
            execution_accuracy=execution_accuracy,
            exact_match=exact_match_rate,
            partial_match=partial_match_rate,
            avg_execution_time=avg_time,
            total_queries=total,
            correct_queries=correct,
            error_count=errors,
            fitness_score=fitness_score,
        )

    def _generate_sql(self, question: str, schema: Dict[str, Any]) -> str:
        """
        使用 LLM 生成 SQL

        Args:
            question: 自然語言問題
            schema: 資料庫 schema

        Returns:
            生成的 SQL
        """
        # TODO: 實作 LLM 調用
        # 這裡需要實際調用 OpenAI 或其他 LLM API

        prompt = f"""Given the following database schema:
{schema}

Generate a SQL query for the following question:
{question}

Return only the SQL query, no explanation."""

        # 模擬 LLM 回應（實際應用需要真正調用 API）
        if self.llm_client:
            # response = self.llm_client.generate(prompt)
            # return response
            pass

        return "SELECT * FROM table"  # 佔位符

    def _evaluate_sql(self, generated_sql: str, gold_sql: str, db_id: str) -> tuple:
        """
        評估生成的 SQL

        Args:
            generated_sql: 生成的 SQL
            gold_sql: 標準答案 SQL
            db_id: 資料庫 ID

        Returns:
            (是否正確, 匹配類型)
        """
        # 標準化 SQL
        gen_normalized = self._normalize_sql(generated_sql)
        gold_normalized = self._normalize_sql(gold_sql)

        # 完全匹配
        if gen_normalized == gold_normalized:
            return True, "exact"

        # 執行匹配（比較執行結果）
        try:
            gen_result = self.db_executor(generated_sql, db_id)
            gold_result = self.db_executor(gold_sql, db_id)

            if gen_result == gold_result:
                return True, "execution"

            # 部分匹配（結果有重疊）
            if self._partial_match(gen_result, gold_result):
                return False, "partial"

        except Exception:
            pass

        return False, "none"

    def _normalize_sql(self, sql: str) -> str:
        """標準化 SQL 字串"""
        # 轉小寫
        sql = sql.lower()
        # 移除多餘空白
        sql = " ".join(sql.split())
        # 移除結尾分號
        sql = sql.rstrip(";")
        return sql

    def _partial_match(self, result1: Any, result2: Any) -> bool:
        """檢查部分匹配"""
        if not result1 or not result2:
            return False

        # 如果是列表，檢查是否有交集
        if isinstance(result1, list) and isinstance(result2, list):
            set1 = set(map(str, result1))
            set2 = set(map(str, result2))
            intersection = set1 & set2
            return len(intersection) > 0.5 * max(len(set1), len(set2))

        return False


class ComplexityStratifiedEvaluator(FitnessEvaluator):
    """
    複雜度分層評估器

    按查詢複雜度（JOIN 數量）分層評估，
    特別關注複雜查詢的表現。
    """

    def evaluate_stratified(self, strategy: Any, decomposer_class: type, composer_class: type) -> Dict[str, EvaluationResult]:
        """
        分層評估

        Returns:
            按複雜度分層的評估結果
        """
        # 將驗證集按複雜度分組
        stratified_sets = {
            "simple": [],      # 1-2 JOINs
            "medium": [],      # 3-4 JOINs
            "complex": [],     # 5+ JOINs
        }

        for item in self.validation_set:
            join_count = item.get("join_count", self._estimate_joins(item["gold_sql"]))

            if join_count <= 2:
                stratified_sets["simple"].append(item)
            elif join_count <= 4:
                stratified_sets["medium"].append(item)
            else:
                stratified_sets["complex"].append(item)

        # 對每個分組評估
        results = {}
        for complexity, subset in stratified_sets.items():
            if subset:
                original_set = self.validation_set
                self.validation_set = subset
                results[complexity] = self.evaluate(strategy, decomposer_class, composer_class)
                self.validation_set = original_set

        return results

    def _estimate_joins(self, sql: str) -> int:
        """估計 SQL 中的 JOIN 數量"""
        sql_upper = sql.upper()
        join_count = sql_upper.count(" JOIN ")
        return join_count


# 測試代碼
if __name__ == "__main__":
    # 模擬驗證集
    validation_set = [
        {
            "question": "列出所有員工",
            "gold_sql": "SELECT * FROM employees",
            "db_id": "test",
            "join_count": 0,
        },
        {
            "question": "哪個部門的員工最多",
            "gold_sql": "SELECT d.name, COUNT(*) FROM departments d JOIN employees e ON d.id = e.dept_id GROUP BY d.id ORDER BY COUNT(*) DESC LIMIT 1",
            "db_id": "test",
            "join_count": 1,
        },
    ]

    # 模擬執行器
    def mock_executor(sql, db_id):
        return [("result",)]

    evaluator = FitnessEvaluator(
        validation_set=validation_set,
        llm_client=None,
        db_executor=mock_executor,
    )

    print("適應度評估器已初始化")
    print(f"驗證集大小: {len(validation_set)}")
