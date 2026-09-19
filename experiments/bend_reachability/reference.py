"""Independent graph specification: queue-based BFS, matrix permutation."""
from collections import deque


def matrix(graph, n=3):
    return [[(graph // 2 ** (n * i + j)) % 2 for j in range(n)] for i in range(n)]


def distance(graph, start=0, target=2, n=3):
    edges = matrix(graph, n)
    todo, seen = deque([(start, 0)]), {start}
    while todo:
        vertex, depth = todo.popleft()
        if vertex == target:
            return depth
        for neighbor in range(n):
            if edges[vertex][neighbor] and neighbor not in seen:
                seen.add(neighbor)
                todo.append((neighbor, depth + 1))
    return n


def relabel(graph, start, target):
    order = [start] + [v for v in range(3) if v not in (start, target)] + [target]
    edges = matrix(graph)
    return sum(edges[order[i]][order[j]] * 2 ** (3 * i + j)
               for i in range(3) for j in range(3))


def decode(answer):
    return {1: 0, 2: 0, 3: 1}.get(answer, 2)


def predict(scenario, graph):
    if scenario["mode"] == 1:
        return 3
    if scenario["mode"] == 2:
        return 3 if graph == 4 else distance(graph)
    return scenario["encoding"][distance(graph) - 1]


def repair(scenario, answer):
    return scenario["decoder"][answer - 1] if answer in (1, 2, 3) else 0


def expected_rows(cfg):
    pairs = [(s, t) for s in range(3) for t in range(3) if s != t]
    for graph in range(512):
        for s, t in pairs:
            mapped = relabel(graph, s, t)
            # Compare against original-graph BFS, not the relabeled solver.
            d = distance(graph, s, t)
            for answer in cfg["output_classes"]:
                yield dict(kind="point", g=graph, s=s, t=t, answer=answer,
                           mapped=mapped, distance=d, verdict=decode(answer))
    for scenario in cfg["scenarios"]:
        name = scenario["name"]
        for graph in range(512):
            raw = predict(scenario, graph)
            yield dict(kind="useful", name=name, g=graph, raw=raw, repaired=repair(scenario, raw))
        for graph in range(512):
            for s, t in pairs:
                mapped = relabel(graph, s, t)
                raw = predict(scenario, mapped)
                fixed = repair(scenario, raw)
                yield dict(kind="protected", name=name, g=graph, s=s, t=t,
                           mapped=mapped, raw=raw, repaired=fixed,
                           recovered=decode(fixed),
                           live=1 if scenario["live_removed"] else decode(fixed))
