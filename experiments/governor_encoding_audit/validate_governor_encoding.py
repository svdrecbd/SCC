"""Exact controls for an anchored self-modification governor encoding.

Algebraic validation over GF(2) and small reversible circuits. This is not an
attack on a neural model, a training run or a demonstration of SCC.
"""

from collections import deque
from itertools import combinations
from pathlib import Path
import hashlib
import json
import sys
import time


def weight(value):
    return bin(value).count("1")


def parity(value):
    return weight(value) & 1


def carryless_product(left, right):
    result = 0
    while right:
        if right & 1:
            result ^= left
        left <<= 1
        right >>= 1
    return result


class BrakeCode:
    """Cyclic code whose low message bits are the brake and high bits the knowledge."""

    def __init__(self, configuration):
        self.length = configuration["block_length"]
        self.dimension = configuration["message_length"]
        self.generator = configuration["generator_polynomial"]
        self.radius = configuration["decoding_radius"]
        self.brake_bits = configuration["brake_bits"]
        self.brake_mask = (1 << self.brake_bits) - 1
        self.message_of = {}
        for message in range(1 << self.dimension):
            word = carryless_product(message, self.generator)
            assert word < (1 << self.length)
            self.message_of[word] = message
        assert len(self.message_of) == 1 << self.dimension
        self.distance = min(weight(word) for word in self.message_of if word)
        assert self.distance >= 2 * self.radius + 1
        patterns = [0]
        for size in range(1, self.radius + 1):
            for positions in combinations(range(self.length), size):
                patterns.append(sum(1 << position for position in positions))
        self.ball_size = len(patterns)
        self.decoded = {}
        for word, message in self.message_of.items():
            for pattern in patterns:
                state = word ^ pattern
                assert state not in self.decoded
                self.decoded[state] = message

    def encode(self, brake, knowledge):
        return carryless_product(brake | (knowledge << self.brake_bits), self.generator)

    def brake(self, message):
        return message & self.brake_mask

    def knowledge(self, message):
        return message >> self.brake_bits


def unit_edit_search(code, start, allowed, target):
    """Breadth-first search over single-coordinate edits without correction."""
    parent = {start: None}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        if target(code.decoded[state]):
            path = []
            while state is not None:
                path.append(state)
                state = parent[state]
            return list(reversed(path)), len(parent)
        for position in range(code.length):
            candidate = state ^ (1 << position)
            if candidate in parent:
                continue
            message = code.decoded.get(candidate)
            if message is None or not allowed(message):
                continue
            parent[candidate] = state
            queue.append(candidate)
    return None, len(parent)


def drift_controls(code, configuration):
    anchor = configuration["anchor_brake"]
    reference = configuration["reference_knowledge"]
    start = code.encode(anchor, reference)
    results = {
        "code_length": code.length,
        "code_dimension": code.dimension,
        "minimum_distance": code.distance,
        "decoding_radius": code.radius,
        "decodable_states": len(code.decoded),
        "total_states": 1 << code.length,
    }

    # Cliff-only rule: death is decoding failure. Edits accumulate without refresh.
    lengths = []
    for word, message in sorted(code.message_of.items()):
        brake, knowledge = code.brake(message), code.knowledge(message)
        path, _ = unit_edit_search(
            code, word,
            lambda decoded, k=knowledge: code.knowledge(decoded) == k,
            lambda decoded, b=brake: code.brake(decoded) != b)
        assert path is not None
        assert all(code.knowledge(code.decoded[state]) == knowledge for state in path)
        lengths.append(len(path) - 1)
    results["cliff_absorption_brake_change_edits_minimum"] = min(lengths)
    results["cliff_absorption_brake_change_edits_maximum"] = max(lengths)
    results["cliff_absorption_starting_codewords"] = len(lengths)
    path, _ = unit_edit_search(
        code, start,
        lambda decoded: code.knowledge(decoded) == reference,
        lambda decoded: code.brake(decoded) != anchor)
    results["cliff_absorption_reference_path"] = [
        {"state": format(state, "015b"),
         "distance_from_start": weight(state ^ start),
         "decoded_brake": code.brake(code.decoded[state]),
         "decoded_knowledge": code.knowledge(code.decoded[state])}
        for state in path]

    # Refresh after each edit, with an edit-weight cap. Moving from codeword c to
    # c' in one edit requires weight d(c, c') - radius.
    codewords = sorted(code.message_of)
    def smallest_cap(accept_intermediate, target):
        for cap in range(1, code.length + 1):
            seen = {start}
            queue = deque([start])
            while queue:
                word = queue.popleft()
                if target(code.message_of[word]):
                    return cap
                for other in codewords:
                    if other in seen or weight(word ^ other) - code.radius > cap:
                        continue
                    if not accept_intermediate(code.message_of[other]):
                        continue
                    seen.add(other)
                    queue.append(other)
        return None
    results["refresh_cap_for_brake_change"] = smallest_cap(
        lambda message: True,
        lambda message: code.brake(message) != anchor
        and code.knowledge(message) == reference)
    results["refresh_cap_for_knowledge_change"] = smallest_cap(
        lambda message: code.brake(message) == anchor,
        lambda message: code.brake(message) == anchor
        and code.knowledge(message) != reference)

    # Unequal protection: every nonzero brake functional over the message.
    minimum_weight_words = [word for word in codewords if word and weight(word) == code.distance]
    functional_distances = []
    for functional in range(1, 1 << code.dimension):
        functional_distances.append(min(
            weight(word) for word in codewords
            if parity(code.message_of[word] & functional)))
    results["minimum_weight_codewords"] = len(minimum_weight_words)
    results["minimum_weight_message_rank"] = gf2_rank(
        [code.message_of[word] for word in minimum_weight_words])
    results["brake_functional_distance_minimum"] = min(functional_distances)
    results["brake_functional_distance_maximum"] = max(functional_distances)
    results["brake_functionals_examined"] = len(functional_distances)

    # Anchored rule: death on decoding failure or on any decoded brake other than
    # the fixed anchor. This rule is an explicit check.
    path, reachable = unit_edit_search(
        code, start,
        lambda decoded: code.brake(decoded) == anchor,
        lambda decoded: code.brake(decoded) != anchor)
    assert path is None
    reached_knowledge = set()
    queue = deque([start])
    seen = {start}
    while queue:
        state = queue.popleft()
        reached_knowledge.add(code.knowledge(code.decoded[state]))
        for position in range(code.length):
            candidate = state ^ (1 << position)
            message = code.decoded.get(candidate)
            if candidate in seen or message is None or code.brake(message) != anchor:
                continue
            seen.add(candidate)
            queue.append(candidate)
    results["anchored_brake_change_reachable"] = False
    results["anchored_reachable_states"] = reachable
    results["anchored_reachable_knowledge_values"] = len(reached_knowledge)
    results["knowledge_values"] = 1 << (code.dimension - code.brake_bits)
    return results


def gf2_rank(vectors):
    basis = []
    for vector in vectors:
        for element in basis:
            vector = min(vector, vector ^ element)
        if vector:
            basis.append(vector)
    return len(basis)


def column_basis_matrix(code):
    """Rows are coordinates; columns are the code basis followed by unit vectors."""
    columns = [carryless_product(1 << index, code.generator) for index in range(code.dimension)]
    for coordinate in range(code.length):
        candidate = columns + [1 << coordinate]
        if gf2_rank(candidate) == len(candidate):
            columns = candidate
    assert len(columns) == code.length
    return [sum(((columns[column] >> row) & 1) << column for column in range(code.length))
            for row in range(code.length)]


def elimination_operations(matrix, size):
    """Row operations reducing matrix to identity; each is an in-place state edit."""
    rows = list(matrix)
    operations = []
    for column in range(size):
        pivot = next(row for row in range(column, size) if (rows[row] >> column) & 1)
        if pivot != column:
            for target, source in ((column, pivot), (pivot, column), (column, pivot)):
                rows[target] ^= rows[source]
                operations.append((target, source))
        for row in range(size):
            if row != column and (rows[row] >> column) & 1:
                rows[row] ^= rows[column]
                operations.append((row, column))
    assert rows == [1 << index for index in range(size)]
    return operations


def apply_operations(state, operations):
    for target, source in operations:
        if (state >> source) & 1:
            state ^= 1 << target
    return state


def keystream(seed, taps, register_bits, length):
    mask = sum(1 << tap for tap in taps)
    register = seed
    stream = 0
    for position in range(length):
        stream |= (register & 1) << position
        feedback = parity(register & mask)
        register = (register >> 1) | (feedback << (register_bits - 1))
    return stream


def transparent_reencoding(code, configuration):
    results = {}
    anchor = configuration["anchor_brake"]
    knowledge_values = 1 << (code.dimension - code.brake_bits)

    # (a) Streaming brake change. Workspace: the change pattern, a coordinate
    # index and one accumulator bit; the state is never copied.
    streaming_cases = 0
    for change in range(1, 1 << code.brake_bits):
        for knowledge in range(knowledge_values):
            state = code.encode(anchor, knowledge)
            for coordinate in range(code.length):
                accumulator = 0
                for index in range(code.brake_bits):
                    offset = coordinate - index
                    if (change >> index) & 1 and 0 <= offset and (code.generator >> offset) & 1:
                        accumulator ^= 1
                state ^= accumulator << coordinate
            assert state == code.encode(anchor ^ change, knowledge)
            streaming_cases += 1
    results["streaming_brake_change_cases"] = streaming_cases
    results["streaming_workspace_bits"] = (
        code.brake_bits + (code.length - 1).bit_length() + 1)

    # (b) General in-place linear re-encoding with zero auxiliary bits.
    basis = column_basis_matrix(code)
    operations = elimination_operations(basis, code.length)
    for state in range(1 << code.length):
        transformed = apply_operations(state, operations)
        restored = apply_operations(transformed, list(reversed(operations)))
        assert restored == state
    plain_cases = 0
    for word, message in code.message_of.items():
        assert apply_operations(word, operations) == message
        plain_cases += 1
    returned_cases = 0
    for change in range(1, 1 << code.brake_bits):
        for knowledge in range(knowledge_values):
            plain = apply_operations(code.encode(anchor, knowledge), operations)
            plain ^= change
            assert apply_operations(plain, list(reversed(operations))) == code.encode(anchor ^ change, knowledge)
            returned_cases += 1
    results["in_place_operations"] = len(operations)
    results["in_place_auxiliary_bits"] = 0
    results["in_place_bijection_states_verified"] = 1 << code.length
    results["in_place_plain_extraction_cases"] = plain_cases
    results["in_place_return_with_changed_brake_cases"] = returned_cases

    # (c) Keyed stream: the brake is a key required for decoding.
    register_bits = configuration["keystream_register_bits"]
    taps = configuration["keystream_feedback_taps"]
    keys = range(1, 1 << register_bits)
    wrong_key_correct = 0
    wrong_key_trials = 0
    rekey_cases = 0
    for key in keys:
        stream = keystream(key, taps, register_bits, code.length)
        for knowledge in range(knowledge_values):
            stored = code.encode(anchor, knowledge) ^ stream
            for other in keys:
                if other == key:
                    continue
                decoded = code.decoded.get(stored ^ keystream(other, taps, register_bits, code.length))
                wrong_key_trials += 1
                if decoded is not None and code.knowledge(decoded) == knowledge:
                    wrong_key_correct += 1
            register = key
            mask = sum(1 << tap for tap in taps)
            state = stored
            for coordinate in range(code.length):
                state ^= (register & 1) << coordinate
                register = (register >> 1) | (parity(register & mask) << (register_bits - 1))
            assert state == code.encode(anchor, knowledge)
            rekey_cases += 1
    results["keyed_wrong_key_knowledge_recovery"] = [wrong_key_correct, wrong_key_trials]
    results["keyed_in_place_removal_cases"] = rekey_cases
    results["keyed_in_place_workspace_bits"] = register_bits + (code.length - 1).bit_length()
    return results


def package_controls(configuration):
    bits = configuration["package_block_bits"]
    blocks = configuration["package_message_blocks"]
    public_key = configuration["package_public_key"]
    box = configuration["substitution_box"]
    block_mask = (1 << bits) - 1
    assert sorted(box) == list(range(1 << bits))

    def cipher(key, value):
        return box[value ^ key]

    def package(message_blocks, session_key):
        stored = [message_blocks[index] ^ cipher(session_key, index + 1) for index in range(blocks)]
        final = session_key
        for index, value in enumerate(stored):
            final ^= cipher(public_key, value ^ (index + 1))
        return stored + [final]

    def unpack_in_place(state):
        state = list(state)
        accumulator = state[blocks]
        for index in range(blocks):
            accumulator ^= cipher(public_key, state[index] ^ (index + 1))
        for index in range(blocks):
            state[index] ^= cipher(accumulator, index + 1)
        state[blocks] ^= state[blocks]
        return state, accumulator

    cases = 0
    error_recovered_blocks = {}
    for encoded_message in range(1 << (bits * blocks)):
        message_blocks = [(encoded_message >> (bits * index)) & block_mask for index in range(blocks)]
        for session_key in range(1 << bits):
            stored = package(message_blocks, session_key)
            recovered, key = unpack_in_place(stored)
            assert key == session_key and recovered[:blocks] == message_blocks and recovered[blocks] == 0
            cases += 1
            for block in range(blocks + 1):
                for bit in range(bits):
                    damaged = list(stored)
                    damaged[block] ^= 1 << bit
                    recovered, _ = unpack_in_place(damaged)
                    count = sum(recovered[index] == message_blocks[index] for index in range(blocks))
                    error_recovered_blocks[count] = error_recovered_blocks.get(count, 0) + 1
    return {
        "package_inversion_cases": cases,
        "package_in_place_workspace_bits": bits + (blocks).bit_length(),
        "package_single_bit_error_recovered_block_histogram": {
            str(count): total for count, total in sorted(error_recovered_blocks.items())},
    }


def reversible_completeness(configuration):
    wires = configuration["reversible_wires"]
    size = 1 << wires
    gates = []
    for target in range(wires):
        gates.append(lambda x, t=target: x ^ (1 << t))
    for control in range(wires):
        for target in range(wires):
            if control != target:
                gates.append(lambda x, c=control, t=target: x ^ (((x >> c) & 1) << t))
    for target in range(wires):
        controls = [wire for wire in range(wires) if wire != target]
        for first, second in combinations(controls, 2):
            gates.append(lambda x, a=first, b=second, t=target:
                         x ^ ((((x >> a) & 1) & ((x >> b) & 1)) << t))
    tables = [tuple(gate(value) for value in range(size)) for gate in gates]
    identity = tuple(range(size))
    depth = {identity: 0}
    queue = deque([identity])
    while queue:
        permutation = queue.popleft()
        for table in tables:
            successor = tuple(table[value] for value in permutation)
            if successor not in depth:
                depth[successor] = depth[permutation] + 1
                queue.append(successor)
    histogram = {}
    for value in depth.values():
        histogram[value] = histogram.get(value, 0) + 1
    total = 1
    for factor in range(2, size + 1):
        total *= factor
    assert len(depth) == total
    return {
        "reversible_wires": wires,
        "reversible_gate_count": len(gates),
        "reversible_permutations_reached": len(depth),
        "reversible_permutations_total": total,
        "reversible_auxiliary_bits": 0,
        "reversible_depth_histogram": {str(key): value for key, value in sorted(histogram.items())},
    }


def main():
    started = time.monotonic()
    directory = Path(sys.argv[1]).resolve()
    configuration = json.loads((directory / "config.json").read_text())
    code = BrakeCode(configuration)
    summary = {"entry": configuration["entry"]}
    summary["drift"] = drift_controls(code, configuration)
    summary["transparent_reencoding"] = transparent_reencoding(code, configuration)
    summary["package"] = package_controls(configuration)
    summary["reversible"] = reversible_completeness(configuration)
    summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
    summary["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    assert len(text.encode()) <= configuration["output_limit_bytes"]
    (directory / "summary.json").write_text(text)
    print(json.dumps({"status": "completed", "elapsed_seconds": summary["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
