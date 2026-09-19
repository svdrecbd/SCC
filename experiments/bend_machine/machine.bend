import Base

# Programs are data, including invalid opcodes and policy-bypassing instructions.
# Status: 0 running, 1 halted, 2 exhausted, 3 invalid opcode, 4 fell off tape.
type State is Data:
  S{mem: U32, a: U32, b: U32, j: Bool, out: U32, pc: Nat,
    steps: Nat, reads: Nat, writes: Nat, copies: Nat, status: U32}

type Tape is Data:
  End{}
  Code{op: U32, tail: Tape}

def initial(mem: U32) -> State:
  S{mem, 0, 0, False{}, 5, 0n, 0n, 0n, 0n, 0n, 0}

def stopped(s: State, status: U32) -> State:
  match s:
    case S{mem, a, b, j, out, pc, steps, reads, writes, copies, _}:
      S{mem, a, b, j, out, pc, steps, reads, writes, copies, status}

def fetch(pc: Nat, tape: Tape) -> Maybe<U32>:
  match pc tape:
    case _ End{}:
      None{}
    case 0n Code{op, _}:
      Some{op}
    case 1n+p Code{_, rest}:
      fetch(p, rest)

def selected(j: Bool, a: U32) -> U32:
  match j:
    case False{}:
      4
    case True{}:
      a

# Data reads/writes exclude instruction fetch, PC and instrumentation counters.
# Every branch charges exactly one fetch/step and advances PC unless jumping.
def step(op: U32, s: State, x: U32, role: Bool) -> State:
  match op s:
    case 0 S{mem, a, b, j, out, pc, steps, reads, writes, copies, _}:
      S{mem, a, b, j, out, 1n+pc, 1n+steps, reads, writes, copies, 1}
    case 1 S{+mem, _, b, j, out, pc, steps, reads, writes, copies, _}:
      S{mem, mem, b, j, out, 1n+pc, 1n+steps, 1n+reads, 1n+writes, copies, 0}
    case 2 S{mem, a, _, j, out, pc, steps, reads, writes, copies, _}:
      S{mem, a, x, j, out, 1n+pc, 1n+steps, 1n+reads, 1n+writes, copies, 0}
    case 3 S{mem, a, +b, j, out, pc, steps, reads, writes, copies, _}:
      S{mem, ((a + b) .&. 3 : U32), b, j, out, 1n+pc, 1n+steps, 2n+reads, 1n+writes, copies, 0}
    case 4 S{_, +a, b, j, out, pc, steps, reads, writes, copies, _}:
      S{a, a, b, j, out, 1n+pc, 1n+steps, 1n+reads, 1n+writes, copies, 0}
    case 5 S{mem, a, b, _, out, pc, steps, reads, writes, copies, _}:
      S{mem, a, b, role, out, 1n+pc, 1n+steps, 1n+reads, 1n+writes, copies, 0}
    case 6 S{mem, +a, b, +j, _, pc, steps, reads, writes, copies, _}:
      S{mem, a, b, j, selected(j, a), 1n+pc, 1n+steps, 2n+reads, 1n+writes, copies, 0}
    case 7 S{mem, +a, b, j, _, pc, steps, reads, writes, copies, _}:
      S{mem, a, b, j, a, 1n+pc, 1n+steps, 1n+reads, 1n+writes, copies, 0}
    case 8 S{mem, +a, _, j, out, pc, steps, reads, writes, copies, _}:
      S{mem, a, a, j, out, 1n+pc, 1n+steps, 1n+reads, 1n+writes, 1n+copies, 0}
    case 9 S{mem, _, b, j, out, pc, steps, reads, writes, copies, _}:
      S{mem, 0, b, j, out, 1n+pc, 1n+steps, reads, 1n+writes, copies, 0}
    case 10 S{mem, a, b, j, out, pc, steps, reads, writes, copies, _}:
      S{mem, (a .^. 3 : U32), b, j, out, 1n+pc, 1n+steps, 1n+reads, 1n+writes, copies, 0}
    case 11 S{mem, a, b, _, out, pc, steps, reads, writes, copies, _}:
      S{mem, a, b, True{}, out, 1n+pc, 1n+steps, reads, 1n+writes, copies, 0}
    case 12 S{mem, a, b, _, out, pc, steps, reads, writes, copies, _}:
      S{mem, a, b, False{}, out, 1n+pc, 1n+steps, reads, 1n+writes, copies, 0}
    case 13 S{mem, a, b, j, out, _, steps, reads, writes, copies, _}:
      S{mem, a, b, j, out, 0n, 1n+steps, reads, writes, copies, 0}
    case _ S{mem, a, b, j, out, pc, steps, reads, writes, copies, _}:
      S{mem, a, b, j, out, 1n+pc, 1n+steps, reads, writes, copies, 3}

def position(s: State) -> Nat:
  match s:
    case S{_, _, _, _, _, pc, _, _, _, _, _}:
      pc

def status_of(s: State) -> U32:
  match s:
    case S{_, _, _, _, _, _, _, _, _, _, status}:
      status

def ticks(s: State) -> Nat:
  match s:
    case S{_, _, _, _, _, _, steps, _, _, _, _}:
      steps

def found(op: Maybe<U32>, s: State, x: U32, role: Bool) -> State:
  match op:
    case None{}:
      stopped(s, 4)
    case Some{code}:
      step(code, s, x, role)

def advance_status(status: U32, tape: Tape, +s: State, x: U32, role: Bool) -> State:
  match status:
    case 0:
      found(fetch(position(s), tape), s, x, role)
    case _:
      s

def advance(tape: Tape, +s: State, x: U32, role: Bool) -> State:
  advance_status(status_of(s), tape, s, x, role)

def terminal_status(status: U32) -> U32:
  match status:
    case 0:
      2
    case _:
      status

def finish(s: State) -> State:
  match s:
    case S{mem, a, b, j, out, pc, steps, reads, writes, copies, status}:
      S{mem, a, b, j, out, pc, steps, reads, writes, copies, terminal_status(status)}

# Stopped states are absorbing. Remaining host recursion does not execute VM
# instructions or increase its counters. Host overhead is not a VM step cost.
def run(fuel: Nat, +tape: Tape, +s: State, +x: U32, +role: Bool) -> State:
  match fuel:
    case 0n:
      finish(s)
    case 1n+rest:
      run(rest, tape, advance(tape, s, x, role), x, role)
