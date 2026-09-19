import Base
import ./machine.bend as M

# Negative checker control: this must fail with a type/equality error.
law halt_is_free:
  {M.ticks(M.step(0, M.initial(0), 0, False{})) == 0n : Nat}

def halt_is_free():
  {==}
