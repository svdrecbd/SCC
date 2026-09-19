import Base
import ./graph.bend as G

law reachable_is_safe:
  {G.decode(G.choose(True{}, False{})) == 1 : U32}

def reachable_is_safe():
  {==}
