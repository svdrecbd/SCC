"""Exact full-information, binary-perfect-privacy transport certificates."""
from fractions import Fraction as F
import heapq
import json
from pathlib import Path
import sys


def transport(cost, mass=F(1), partial=False):
    m, n = len(cost), len(cost[0])
    scale = mass.denominator
    total = m * n * scale
    target = int(total * mass)
    source, sink = m + n, m + n + 1
    graph = [[] for _ in range(sink + 1)]

    def edge(a, b, cap, c):
        graph[a].append([b, len(graph[b]), cap, c])
        graph[b].append([a, len(graph[a]) - 1, 0, -c])
        return graph[a][-1]

    for i in range(m):
        edge(source, i, n * scale, 0)
    edges = [[edge(i, m + j, total + 1, cost[i][j]) for j in range(n)] for i in range(m)]
    for j in range(n):
        edge(m + j, sink, m * scale, 0)
    potential = [0] * len(graph)
    sent = 0
    while sent < target:
        dist = [None] * len(graph)
        previous = [None] * len(graph)
        dist[source] = 0
        queue = [(0, source)]
        while queue:
            distance, a = heapq.heappop(queue)
            if distance != dist[a]:
                continue
            for eidx, (b, rev, cap, c) in enumerate(graph[a]):
                if not cap:
                    continue
                reduced = c + potential[a] - potential[b]
                assert reduced >= 0
                candidate = distance + reduced
                if dist[b] is None or candidate < dist[b]:
                    dist[b] = candidate
                    previous[b] = (a, eidx)
                    heapq.heappush(queue, (candidate, b))
        assert dist[sink] is not None
        for a, distance in enumerate(dist):
            if distance is not None:
                potential[a] += distance
        amount = target - sent
        b = sink
        while b != source:
            a, eidx = previous[b]
            amount = min(amount, graph[a][eidx][2])
            b = a
        b = sink
        while b != source:
            a, eidx = previous[b]
            e = graph[a][eidx]
            e[2] -= amount
            graph[b][e[1]][2] += amount
            b = a
        sent += amount
    flow = [[i, j, total + 1 - edges[i][j][2]] for i in range(m)
            for j in range(n) if edges[i][j][2] != total + 1]
    if partial:
        return (flow, [min(0, potential[source]-potential[i]) for i in range(m)],
                [min(0, potential[m+j]-potential[sink]) for j in range(n)],
                potential[sink]-potential[source], scale, target)
    return flow, [-potential[i] for i in range(m)], potential[m:m+n]


def cases(cfg):
    for N in cfg['denominators']:
        worlds = [(a, b, c, N-a-b-c) for a in range(N+1)
                  for b in range(N-a+1) for c in range(N-a-b+1)]
        for judge in cfg['judgments']:
            def label(q):
                a, b, c, d = q
                return {'action_risk': 2*(b+d)>N, 'net_harm': b>c,
                        'newly_harmed': 2*b>N, 'any_new_harm': b>0}[judge]
            for utility in cfg['utilities']:
                pos, neg = [], []
                for q in worlds:
                    a, b, c, d = q
                    v = [c+d, b+d] if utility == 'prospective' else list(q)
                    (pos if label(q) else neg).append(v)
                for prior in cfg['priors']:
                    pi = F(1, 2) if prior == 'balanced' else F(len(pos), len(worlds))
                    yield f'N{N}/{judge}/{utility}/{prior}', N, pos, neg, pi
    yield 'control/causal/prospective', 4, [[1,3]], [[1,3]], F(1,2)
    yield 'control/causal/paired', 4, [[0,3,1,0]], [[1,2,0,1]], F(1,2)
    yield 'control/small_margin', 20, [[11]], [[9]], F(1,2)
    yield 'control/target_identity', 1, [[1]], [[0]], F(1,2)
    yield 'control/independent', 2, [[0],[1],[2]], [[0],[1],[2]], F(1,2)


def main(config, output):
    cfg = json.loads(Path(config).read_text())
    with Path(output).open('x') as f:
        for name, N, pos, neg, pi in cases(cfg):
            cost = [[sum((a-b)**2 for a,b in zip(p,q)) for q in neg] for p in pos]
            flow, u, v = transport(cost)
            objective = sum(k*cost[i][j] for i,j,k in flow)
            row = dict(case=name, denominator=N, positive=pos, negative=neg,
                       pi=[pi.numerator, pi.denominator], flow=flow,
                       dual_positive=u, dual_negative=v, objective=objective)
            f.write(json.dumps(row)+'\n')
            f.flush()


if __name__ == '__main__':
    main(*sys.argv[1:])
