import Base
import ./machine.bend as M
import ./LAWS.bend as L

def L.halt_cost(s, x, role):
  match s:
    case M.S{mem, a, b, j, out, pc, steps, reads, writes, copies, status}:
      {==}

def L.halt_state(s, x, role):
  match s:
    case M.S{mem, a, b, j, out, pc, steps, reads, writes, copies, status}:
      {==}

def L.exhausted_cost(tape, s, x, role):
  match s:
    case M.S{mem, a, b, j, out, pc, steps, reads, writes, copies, status}:
      {==}

def L.stopped_absorbs(fuel, tape, s, x, role):
  match fuel s:
    case 0n M.S{mem, a, b, j, out, pc, steps, reads, writes, copies, status}:
      {==}
    case 1n+rest M.S{mem, a, b, j, out, pc, steps, reads, writes, copies, status}:
      L.stopped_absorbs(rest, tape, M.S{mem, a, b, j, out, pc, steps, reads, writes, copies, 1}, x, role)
