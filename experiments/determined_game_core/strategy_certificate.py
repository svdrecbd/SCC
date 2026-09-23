"""Editable winning-policy certificates for finite total parity games."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ParityGame:
    owners: tuple
    priorities: tuple
    edges: tuple

    def __post_init__(self):
        count = len(self.owners)
        if not count or len(self.priorities) != count or len(self.edges) != count:
            raise ValueError('Game arrays must have the same positive length.')
        if any(type(owner) is not int or owner not in (0, 1) for owner in self.owners):
            raise ValueError('Every owner must be 0 or 1.')
        if any(type(priority) is not int or priority < 0 for priority in self.priorities):
            raise ValueError('Priorities must be nonnegative integers.')
        for successors in self.edges:
            if not successors or len(set(successors)) != len(successors):
                raise ValueError('Every vertex must have distinct nonempty successors.')
            if any(type(target) is not int or not 0 <= target < count for target in successors):
                raise ValueError('Invalid edge endpoint.')


def reachable_vertices(edges, sources, permitted, work):
    reached, pending = set(), list(sources)
    while pending:
        vertex = pending.pop()
        work['vertex_visits'] += 1
        if vertex not in permitted or vertex in reached:
            continue
        reached.add(vertex)
        for successor in edges[vertex]:
            work['edge_visits'] += 1
            if successor in permitted and successor not in reached:
                pending.append(successor)
    return reached


def verify_strategy(game, start, player, policy):
    count = len(game.owners)
    work = {'vertex_visits': 0, 'edge_visits': 0, 'cycle_tests': 0}
    if (type(start) is not int or not 0 <= start < count
            or type(player) is not int or player not in (0, 1)
            or not isinstance(policy, (tuple, list))
            or len(policy) != count
            or any(type(target) is not int or not 0 <= target < count for target in policy)):
        return False, work
    if any(policy[vertex] not in game.edges[vertex]
           for vertex in range(count) if game.owners[vertex] == player):
        return False, work
    edges = tuple((policy[vertex],) if game.owners[vertex] == player else successors
                  for vertex, successors in enumerate(game.edges))
    reachable = reachable_vertices(edges, [start], set(range(count)), work)
    for vertex in sorted(reachable):
        priority = game.priorities[vertex]
        if priority % 2 == player:
            continue
        work['cycle_tests'] += 1
        permitted = {other for other in reachable if game.priorities[other] <= priority}
        # Starting at successors requires a positive-length return path.
        returned = reachable_vertices(edges, edges[vertex], permitted, work)
        if vertex in returned:
            return False, work
    return True, work


def disclosure(game, start, player, secret, policy):
    if type(secret) is not bool:
        raise ValueError('The private input must be Boolean.')
    accepted, work = verify_strategy(game, start, player, policy)
    return secret if accepted else False, work


def synthesize_with_assessor(game, start, assessor):
    """Assessor returns the exact winning player, with every call counted."""
    player = assessor(game, start)
    calls, copied_edges = 1, 0
    restricted = game
    policy = [0]*len(game.owners)
    for vertex, owner in enumerate(game.owners):
        if owner != player:
            continue
        selected = False
        for successor in restricted.edges[vertex]:
            edges = list(restricted.edges)
            edges[vertex] = (successor,)
            copied_edges += sum(map(len, edges))
            candidate = ParityGame(game.owners, game.priorities, tuple(edges))
            calls += 1
            if assessor(candidate, start) == player:
                policy[vertex] = successor
                restricted = candidate
                selected = True
                break
        if not selected:
            raise ValueError('Assessor answers violated the winning restriction invariant.')
    accepted, work = verify_strategy(game, start, player, policy)
    if not accepted:
        raise ValueError('Final synthesized policy failed verification.')
    return player, tuple(policy), {'assessor_calls': calls, 'copied_edge_entries': copied_edges,
                                  'verification': work}
