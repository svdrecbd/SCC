"""Independent imperative specification. Never imports Bend or generated code."""


def initial(mem):
    return dict(mem=mem, a=0, b=0, j=False, out=5, pc=0, steps=0,
                reads=0, writes=0, copies=0, status=0)


def execute(program, state, x, role, fuel):
    s = dict(state)
    for _ in range(fuel):
        if s["status"]:
            break
        if not 0 <= s["pc"] < len(program):
            s["status"] = 4
            break
        op = program[s["pc"]]
        s["pc"] += 1
        s["steps"] += 1
        if op == 0:
            s["status"] = 1
        elif op == 1:
            s["a"] = s["mem"]
        elif op == 2:
            s["b"] = x
        elif op == 3:
            s["a"] = (s["a"] + s["b"]) % 4
        elif op == 4:
            s["mem"] = s["a"]
        elif op == 5:
            s["j"] = role
        elif op == 6:
            s["out"] = s["a"] if s["j"] else 4
        elif op == 7:
            s["out"] = s["a"]
        elif op == 8:
            s["b"] = s["a"]
            s["copies"] += 1
        elif op == 9:
            s["a"] = 0
        elif op == 10:
            s["a"] = 3 - s["a"]
        elif op in (11, 12):
            s["j"] = op == 11
        elif op == 13:
            s["pc"] = 0
        else:
            s["status"] = 3
        s["reads"] += {1: 1, 2: 1, 3: 2, 4: 1, 5: 1, 6: 2, 7: 1, 8: 1, 10: 1}.get(op, 0)
        s["writes"] += int(1 <= op <= 12)
    if s["status"] == 0:
        s["status"] = 2
    return s


def evaluate(case):
    if case["kind"] == "single":
        return execute(case["program"], case["state"], case["x"], case["role"], case["fuel"])
    if case["kind"] != "trajectory":
        raise ValueError(case["kind"])
    mem = case["mem"]
    result = []
    for i, (x, role) in enumerate(case["inputs"]):
        program = case["programs"][i] if "programs" in case else case["program"]
        state = execute(program, initial(mem), x, role, case["fuel"])
        result.append(state)
        mem = state["mem"]
    return result
