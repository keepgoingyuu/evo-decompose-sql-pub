"""
進化算子 - 變異和交叉操作
"""

import random
from typing import List, Tuple, Any
from copy import deepcopy


class MutationOperator:
    """
    變異算子

    對策略進行隨機變異，引入新的變異。
    """

    def __init__(self, mutation_rate: float = 0.1):
        """
        初始化變異算子

        Args:
            mutation_rate: 變異率（0-1）
        """
        self.mutation_rate = mutation_rate

    def mutate(self, strategy: Any) -> Any:
        """
        對策略進行變異

        Args:
            strategy: 原始策略

        Returns:
            變異後的策略
        """
        mutated = deepcopy(strategy)

        # 變異規則權重
        if random.random() < self.mutation_rate:
            rule_key = random.choice(list(mutated.rules.keys()))
            # 在當前值附近進行小幅變異
            current = mutated.rules[rule_key]
            delta = random.gauss(0, 0.1)  # 標準差 0.1 的高斯變異
            mutated.rules[rule_key] = max(0, min(1, current + delta))

        # 變異組合順序
        if random.random() < self.mutation_rate:
            orders = [
                "filter_join_agg_select",
                "join_filter_agg_select",
                "agg_first",
                "subquery_first",
            ]
            mutated.composition_order = random.choice(orders)

        # 變異數值參數
        if random.random() < self.mutation_rate:
            param = random.choice(["max_decomposition_depth", "join_threshold", "agg_threshold"])
            if param == "max_decomposition_depth":
                mutated.max_decomposition_depth = random.randint(1, 5)
            elif param == "join_threshold":
                mutated.join_threshold = random.randint(1, 4)
            elif param == "agg_threshold":
                mutated.agg_threshold = random.randint(1, 3)

        # 變異布林參數
        if random.random() < self.mutation_rate:
            mutated.subquery_extraction = not mutated.subquery_extraction

        return mutated

    def mutate_population(self, population: List[Any]) -> List[Any]:
        """
        對整個種群進行變異

        Args:
            population: 策略種群

        Returns:
            變異後的種群
        """
        return [self.mutate(strategy) for strategy in population]


class CrossoverOperator:
    """
    交叉算子

    將兩個策略的基因組合，產生子代。
    """

    def __init__(self, crossover_rate: float = 0.7):
        """
        初始化交叉算子

        Args:
            crossover_rate: 交叉率（0-1）
        """
        self.crossover_rate = crossover_rate

    def crossover(self, parent1: Any, parent2: Any) -> Tuple[Any, Any]:
        """
        交叉兩個父代

        Args:
            parent1: 父代 1
            parent2: 父代 2

        Returns:
            兩個子代
        """
        if random.random() > self.crossover_rate:
            return deepcopy(parent1), deepcopy(parent2)

        child1 = deepcopy(parent1)
        child2 = deepcopy(parent2)

        # 交叉規則權重（均勻交叉）
        for rule_key in child1.rules.keys():
            if random.random() < 0.5:
                child1.rules[rule_key], child2.rules[rule_key] = \
                    child2.rules[rule_key], child1.rules[rule_key]

        # 交叉組合順序
        if random.random() < 0.5:
            child1.composition_order, child2.composition_order = \
                child2.composition_order, child1.composition_order

        # 交叉數值參數（算術交叉）
        if random.random() < 0.5:
            alpha = random.random()
            child1.max_decomposition_depth = int(
                alpha * parent1.max_decomposition_depth +
                (1 - alpha) * parent2.max_decomposition_depth
            )
            child2.max_decomposition_depth = int(
                (1 - alpha) * parent1.max_decomposition_depth +
                alpha * parent2.max_decomposition_depth
            )

        if random.random() < 0.5:
            alpha = random.random()
            child1.join_threshold = int(
                alpha * parent1.join_threshold +
                (1 - alpha) * parent2.join_threshold
            )
            child2.join_threshold = int(
                (1 - alpha) * parent1.join_threshold +
                alpha * parent2.join_threshold
            )

        return child1, child2

    def crossover_population(self, population: List[Any]) -> List[Any]:
        """
        對種群進行交叉

        Args:
            population: 策略種群

        Returns:
            交叉後的種群
        """
        # 隨機配對
        random.shuffle(population)
        new_population = []

        for i in range(0, len(population) - 1, 2):
            child1, child2 = self.crossover(population[i], population[i + 1])
            new_population.extend([child1, child2])

        # 如果是奇數，保留最後一個
        if len(population) % 2 == 1:
            new_population.append(deepcopy(population[-1]))

        return new_population


class SelectionOperator:
    """
    選擇算子

    根據適應度選擇存活的個體。
    """

    def __init__(self, selection_type: str = "tournament", tournament_size: int = 3):
        """
        初始化選擇算子

        Args:
            selection_type: 選擇類型 (tournament, roulette, elitism)
            tournament_size: 錦標賽大小
        """
        self.selection_type = selection_type
        self.tournament_size = tournament_size

    def select(self, population: List[Any], fitness_scores: List[float], num_select: int) -> List[Any]:
        """
        選擇個體

        Args:
            population: 策略種群
            fitness_scores: 對應的適應度分數
            num_select: 要選擇的數量

        Returns:
            被選中的個體
        """
        if self.selection_type == "tournament":
            return self._tournament_selection(population, fitness_scores, num_select)
        elif self.selection_type == "roulette":
            return self._roulette_selection(population, fitness_scores, num_select)
        elif self.selection_type == "elitism":
            return self._elitism_selection(population, fitness_scores, num_select)
        else:
            raise ValueError(f"Unknown selection type: {self.selection_type}")

    def _tournament_selection(self, population: List[Any], fitness_scores: List[float], num_select: int) -> List[Any]:
        """錦標賽選擇"""
        selected = []

        for _ in range(num_select):
            # 隨機選擇 tournament_size 個個體
            tournament_idx = random.sample(range(len(population)), min(self.tournament_size, len(population)))
            # 選擇最佳的
            best_idx = max(tournament_idx, key=lambda i: fitness_scores[i])
            selected.append(deepcopy(population[best_idx]))

        return selected

    def _roulette_selection(self, population: List[Any], fitness_scores: List[float], num_select: int) -> List[Any]:
        """輪盤選擇"""
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            return random.choices(population, k=num_select)

        probabilities = [f / total_fitness for f in fitness_scores]
        selected_indices = random.choices(range(len(population)), weights=probabilities, k=num_select)

        return [deepcopy(population[i]) for i in selected_indices]

    def _elitism_selection(self, population: List[Any], fitness_scores: List[float], num_select: int) -> List[Any]:
        """菁英選擇"""
        # 按適應度排序，選擇前 num_select 個
        sorted_pairs = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
        return [deepcopy(p) for p, _ in sorted_pairs[:num_select]]


# 測試代碼
if __name__ == "__main__":
    from dataclasses import dataclass, field
    from typing import Dict

    @dataclass
    class MockStrategy:
        rules: Dict[str, float] = field(default_factory=lambda: {
            "split_by_join": 0.8,
            "split_by_agg": 0.6,
        })
        composition_order: str = "filter_join_agg_select"
        max_decomposition_depth: int = 3
        join_threshold: int = 2
        agg_threshold: int = 1
        subquery_extraction: bool = True

    # 測試變異
    mutation = MutationOperator(mutation_rate=0.5)
    original = MockStrategy()
    mutated = mutation.mutate(original)

    print("原始策略：")
    print(f"  rules: {original.rules}")
    print(f"  composition_order: {original.composition_order}")

    print("\n變異後：")
    print(f"  rules: {mutated.rules}")
    print(f"  composition_order: {mutated.composition_order}")

    # 測試交叉
    crossover = CrossoverOperator(crossover_rate=0.9)
    parent1 = MockStrategy()
    parent2 = MockStrategy(
        rules={"split_by_join": 0.3, "split_by_agg": 0.9},
        composition_order="agg_first",
    )

    child1, child2 = crossover.crossover(parent1, parent2)

    print("\n父代 1：")
    print(f"  rules: {parent1.rules}")
    print("\n父代 2：")
    print(f"  rules: {parent2.rules}")
    print("\n子代 1：")
    print(f"  rules: {child1.rules}")
    print("\n子代 2：")
    print(f"  rules: {child2.rules}")

    # 測試選擇
    selection = SelectionOperator(selection_type="tournament")
    population = [MockStrategy() for _ in range(10)]
    fitness_scores = [random.random() for _ in range(10)]

    selected = selection.select(population, fitness_scores, 5)
    print(f"\n選擇了 {len(selected)} 個個體")
