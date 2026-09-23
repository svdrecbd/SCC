"""Independently check certificates, binary labels and adaptive synthesis."""
from fractions import Fraction
from itertools import product
from pathlib import Path
import json
import platform
import random
import sys
import time
from strategy_certificate import ParityGame, verify_strategy, disclosure, synthesize_with_assessor


def policies(game, player):
    positions = [vertex for vertex, owner in enumerate(game.owners) if owner == player]
    for selections in product(*(game.edges[vertex] for vertex in positions)):
        policy = [0]*len(game.owners)
        for vertex, successor in zip(positions, selections):
            policy[vertex] = successor
        yield tuple(policy)


def eventual_winner(game, start, first, second):
    path, positions = [], {}
    vertex = start
    while vertex not in positions:
        positions[vertex] = len(path)
        path.append(vertex)
        vertex = (first if game.owners[vertex] == 0 else second)[vertex]
    return max(game.priorities[item] for item in path[positions[vertex]:]) % 2


def reference_accepts(game, start, player, policy):
    for opposing in policies(game, 1-player):
        first, second = (policy, opposing) if player == 0 else (opposing, policy)
        if eventual_winner(game, start, first, second) != player:
            return False
    return True


def reference_winner(game, start):
    for policy in policies(game, 0):
        if reference_accepts(game, start, 0, policy):
            return 0
    return 1


def enumerate_games(configuration):
    count = configuration['exhaustive_vertices']
    successor_sets = [tuple(vertex for vertex in range(count) if mask & (1 << vertex))
                      for mask in range(1, 1 << count)]
    for edges in product(successor_sets, repeat=count):
        for owners in product((0, 1), repeat=count):
            for priorities in product(range(configuration['exhaustive_priorities']), repeat=count):
                yield 'exhaustive', ParityGame(owners, priorities, edges)
    generator = random.Random(configuration['seed'])
    count = configuration['random_vertices']
    for _ in range(configuration['random_games']):
        edges = tuple(tuple(sorted(generator.sample(range(count), generator.randrange(1, count+1))))
                      for vertex in range(count))
        owners = tuple(generator.randrange(2) for _ in range(count))
        priorities = tuple(generator.randrange(configuration['random_priorities']) for _ in range(count))
        yield 'seeded', ParityGame(owners, priorities, edges)


def controls():
    hidden_cycle = ParityGame((1, 1), (2, 1), ((1,), (0, 1)))
    assert max(hidden_cycle.priorities) % 2 == 0
    assert not verify_strategy(hidden_cycle, 0, 0, (0, 0))[0]
    unreachable_cycle = ParityGame((0, 1), (2, 1), ((0,), (1,)))
    assert verify_strategy(unreachable_cycle, 0, 0, (0, 0))[0]
    assert not verify_strategy(unreachable_cycle, 0, 0, (1, 0))[0]
    region_choice = ParityGame((0, 0), (1, 0), ((0, 1), (1,)))
    assert all(reference_winner(region_choice, start) == 0 for start in range(2))
    assert not verify_strategy(region_choice, 0, 0, (0, 1))[0]
    assert synthesize_with_assessor(region_choice, 0, reference_winner)[1] == (1, 1)
    try:
        ParityGame((0,), (0,), ((),))
    except ValueError:
        pass
    else:
        raise AssertionError('A non-total game was accepted.')
    return ['hidden_bad_subcycle', 'unreachable_bad_cycle', 'invalid_policy_edge',
            'winning_region_choice_is_insufficient', 'non_total_graph']


def main():
    directory = Path(sys.argv[1])
    configuration = json.loads((directory/'config.json').read_text())
    started = time.perf_counter()
    totals = {'games': 0, 'starts': 0, 'policies': 0, 'disclosure_comparisons': 0,
              'assessor_calls': 0, 'copied_edge_entries': 0, 'verification_vertex_visits': 0,
              'verification_edge_visits': 0, 'winning_players': [0, 0]}
    successes = [0]*4
    expected_correct = [Fraction(0)]*4
    records = []
    for source, game in enumerate_games(configuration):
        totals['games'] += 1
        for start in range(len(game.owners)):
            accepted_by_player = [[], []]
            for player in (0, 1):
                for policy in policies(game, player):
                    accepted, work = verify_strategy(game, start, player, policy)
                    reference = reference_accepts(game, start, player, policy)
                    assert accepted == reference
                    totals['policies'] += 1
                    totals['verification_vertex_visits'] += work['vertex_visits']
                    totals['verification_edge_visits'] += work['edge_visits']
                    outputs = [disclosure(game, start, player, secret, policy)[0] for secret in (False, True)]
                    assert (outputs[0] != outputs[1]) == accepted
                    totals['disclosure_comparisons'] += 1
                    if accepted:
                        accepted_by_player[player].append(policy)
            labels = [bool(items) for items in accepted_by_player]
            assert sum(labels) == 1
            winner = int(labels[1])
            totals['winning_players'][winner] += 1
            synthesized, policy, work = synthesize_with_assessor(game, start, reference_winner)
            assert synthesized == winner and policy in accepted_by_player[winner]
            assert work['assessor_calls'] <= 1+sum(map(len, game.edges))
            totals['assessor_calls'] += work['assessor_calls']
            totals['copied_edge_entries'] += work['copied_edge_entries']
            # Four arbitrary success/abstention patterns independent of query role.
            for scheme in range(4):
                successful = scheme == 3 or (scheme > 0 and totals['starts'] % 3 < scheme)
                successes[scheme] += int(successful)
                for role in (0, 1):
                    expected_correct[scheme] += (Fraction(int((winner == role) == labels[role]))
                                                if successful else Fraction(1, 2))
            records.append({'source': source, 'owners': game.owners, 'priorities': game.priorities,
                            'edges': game.edges, 'start': start, 'winner': winner,
                            'strategy': policy, 'assessor_calls': work['assessor_calls']})
            totals['starts'] += 1
    score_checks = []
    for success_count, correct in zip(successes, expected_correct):
        success = Fraction(success_count, totals['starts'])
        accuracy = correct/(2*totals['starts'])
        assert accuracy == (1+success)/2
        score_checks.append({'synthesis_success': str(success), 'protected_accuracy': str(accuracy)})
    result = {'configuration': configuration, 'totals': totals, 'score_identity_checks': score_checks,
              'controls': controls(), 'python': platform.python_version(),
              'seconds': time.perf_counter()-started, 'neural_training': False,
              'public_complexity_comparison': False}
    (directory/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    (directory/'game_certificates.json').write_text(json.dumps(records, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
