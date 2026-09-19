import Base
import ./machine.bend as M

# Universal local accounting facts, not an SCC claim or full compiler proof.
law halt_cost:
  for s: M.State
  for x: U32
  for role: Bool
  {M.ticks(M.step(0, s, x, role)) == 1n+M.ticks(s) : Nat}

law halt_state:
  for s: M.State
  for x: U32
  for role: Bool
  {M.status_of(M.step(0, s, x, role)) == 1 : U32}

law exhausted_cost:
  for tape: M.Tape
  for s: M.State
  for x: U32
  for role: Bool
  {M.ticks(M.run(0n, tape, s, x, role)) == M.ticks(s) : Nat}

law stopped_absorbs:
  for fuel: Nat
  for tape: M.Tape
  for s: M.State
  for x: U32
  for role: Bool
  {M.run(fuel, tape, M.stopped(s, 1), x, role) == M.stopped(s, 1) : M.State}
