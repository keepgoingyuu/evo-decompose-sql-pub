"""
StrategyGenome - 策略基因組表示
定義可進化的策略結構
"""

import uuid
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import json


@dataclass
class StrategyGenome:
    """
    策略基因組

    定義查詢分解策略的完整基因結構，
    包含所有可進化的參數。
    """

    # 唯一識別碼
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # ===== 分解規則（可進化）=====
    # 每個規則的啟用權重（0-1）
    rules: Dict[str, float] = field(default_factory=lambda: {
        "split_by_join": 0.8,           # 按 JOIN 拆分
        "split_by_aggregation": 0.6,     # 按聚合拆分
        "extract_subquery": 0.7,         # 提取子查詢
        "filter_first": 0.9,             # 先處理篩選
        "compute_then_join": 0.5,        # 先計算再 JOIN
    })

    # ===== 組合順序（可進化）=====
    composition_order: str = "filter_join_agg_select"

    # ===== 數值參數（可進化）=====
    max_decomposition_depth: int = 3      # 最大分解深度 (1-5)
    join_threshold: int = 2               # JOIN 閾值 (1-5)
    agg_threshold: int = 1                # 聚合閾值 (1-3)

    # ===== 布林參數（可進化）=====
    subquery_extraction: bool = True      # 是否提取子查詢
    parallel_subqueries: bool = False     # 是否並行處理子查詢

    # ===== 進化元數據 =====
    fitness_score: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)

    @classmethod
    def random(cls) -> "StrategyGenome":
        """生成隨機策略"""
        return cls(
            rules={
                "split_by_join": random.random(),
                "split_by_aggregation": random.random(),
                "extract_subquery": random.random(),
                "filter_first": random.random(),
                "compute_then_join": random.random(),
            },
            composition_order=random.choice([
                "filter_join_agg_select",
                "join_filter_agg_select",
                "agg_first",
                "subquery_first",
            ]),
            max_decomposition_depth=random.randint(1, 5),
            join_threshold=random.randint(1, 5),
            agg_threshold=random.randint(1, 3),
            subquery_extraction=random.choice([True, False]),
            parallel_subqueries=random.choice([True, False]),
        )

    def to_dict(self) -> Dict[str, Any]:
        """序列化為字典"""
        return {
            "id": self.id,
            "rules": self.rules,
            "composition_order": self.composition_order,
            "max_decomposition_depth": self.max_decomposition_depth,
            "join_threshold": self.join_threshold,
            "agg_threshold": self.agg_threshold,
            "subquery_extraction": self.subquery_extraction,
            "parallel_subqueries": self.parallel_subqueries,
            "fitness_score": self.fitness_score,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyGenome":
        """從字典反序列化"""
        return cls(**data)

    def to_json(self) -> str:
        """序列化為 JSON"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "StrategyGenome":
        """從 JSON 反序列化"""
        return cls.from_dict(json.loads(json_str))

    def clone(self) -> "StrategyGenome":
        """複製策略"""
        new_strategy = StrategyGenome.from_dict(self.to_dict())
        new_strategy.id = str(uuid.uuid4())[:8]
        new_strategy.parent_ids = [self.id]
        return new_strategy


class StrategyPopulation:
    """
    策略種群

    管理一組策略基因組，支援進化操作。
    """

    def __init__(self, size: int = 50, initial_strategies: List[StrategyGenome] = None):
        """
        初始化種群

        Args:
            size: 種群大小
            initial_strategies: 初始策略（如果沒有，則隨機生成）
        """
        self.size = size
        self.generation = 0

        if initial_strategies:
            self.strategies = initial_strategies[:size]
            # 如果不夠，用隨機策略補充
            while len(self.strategies) < size:
                self.strategies.append(StrategyGenome.random())
        else:
            self.strategies = [StrategyGenome.random() for _ in range(size)]

        # 歷史記錄
        self.history: List[Dict[str, Any]] = []
        self.best_strategy: Optional[StrategyGenome] = None
        self.best_fitness: float = 0.0

    def evaluate(self, fitness_fn) -> List[float]:
        """
        評估所有策略的適應度

        Args:
            fitness_fn: 適應度函數，接受 StrategyGenome，返回 float

        Returns:
            適應度分數列表
        """
        fitness_scores = []

        for strategy in self.strategies:
            score = fitness_fn(strategy)
            strategy.fitness_score = score
            fitness_scores.append(score)

            # 更新最佳策略
            if score > self.best_fitness:
                self.best_fitness = score
                self.best_strategy = strategy.clone()

        return fitness_scores

    def evolve(self, mutation_op, crossover_op, selection_op, fitness_scores: List[float]) -> None:
        """
        進行一代進化

        Args:
            mutation_op: 變異算子
            crossover_op: 交叉算子
            selection_op: 選擇算子
            fitness_scores: 當前適應度分數
        """
        # 記錄當前代的統計
        self._record_generation(fitness_scores)

        # 選擇
        selected = selection_op.select(self.strategies, fitness_scores, self.size // 2)

        # 交叉
        offspring = crossover_op.crossover_population(selected * 2)[:self.size]

        # 變異
        mutated = mutation_op.mutate_population(offspring)

        # 保留最佳個體（精英主義）
        if self.best_strategy:
            mutated[0] = self.best_strategy.clone()

        # 更新種群
        self.strategies = mutated
        self.generation += 1

        # 更新代數
        for strategy in self.strategies:
            strategy.generation = self.generation

    def _record_generation(self, fitness_scores: List[float]) -> None:
        """記錄當前代的統計資訊"""
        record = {
            "generation": self.generation,
            "best_fitness": max(fitness_scores),
            "avg_fitness": sum(fitness_scores) / len(fitness_scores),
            "min_fitness": min(fitness_scores),
            "best_strategy_id": self.best_strategy.id if self.best_strategy else None,
        }
        self.history.append(record)

    def get_top_strategies(self, n: int = 5) -> List[StrategyGenome]:
        """取得前 n 個最佳策略"""
        sorted_strategies = sorted(self.strategies, key=lambda s: s.fitness_score, reverse=True)
        return sorted_strategies[:n]

    def save(self, filepath: str) -> None:
        """儲存種群到檔案"""
        data = {
            "size": self.size,
            "generation": self.generation,
            "strategies": [s.to_dict() for s in self.strategies],
            "history": self.history,
            "best_strategy": self.best_strategy.to_dict() if self.best_strategy else None,
            "best_fitness": self.best_fitness,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> "StrategyPopulation":
        """從檔案載入種群"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        population = cls(
            size=data["size"],
            initial_strategies=[StrategyGenome.from_dict(s) for s in data["strategies"]],
        )
        population.generation = data["generation"]
        population.history = data["history"]
        population.best_fitness = data["best_fitness"]
        if data["best_strategy"]:
            population.best_strategy = StrategyGenome.from_dict(data["best_strategy"])

        return population


# 測試代碼
if __name__ == "__main__":
    # 建立隨機策略
    print("=== 策略基因組測試 ===")
    strategy = StrategyGenome.random()
    print(f"隨機策略：")
    print(strategy.to_json())

    # 複製策略
    cloned = strategy.clone()
    print(f"\n複製策略 ID: {cloned.id} (原始: {strategy.id})")

    # 建立種群
    print("\n=== 種群測試 ===")
    population = StrategyPopulation(size=10)
    print(f"種群大小: {len(population.strategies)}")
    print(f"代數: {population.generation}")

    # 模擬適應度評估
    def mock_fitness(strategy):
        # 模擬適應度函數
        return random.random()

    fitness_scores = population.evaluate(mock_fitness)
    print(f"適應度分數: {[f'{s:.3f}' for s in fitness_scores]}")

    # 取得最佳策略
    top = population.get_top_strategies(3)
    print(f"\n前 3 名策略:")
    for i, s in enumerate(top):
        print(f"  {i+1}. {s.id}: {s.fitness_score:.3f}")

    # 儲存和載入
    population.save("/tmp/test_population.json")
    loaded = StrategyPopulation.load("/tmp/test_population.json")
    print(f"\n載入後種群大小: {len(loaded.strategies)}")
