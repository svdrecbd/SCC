import Base

# Nine bits encode edge i -> j at bit 3*i+j. Domain: 0 <= g < 512.
def edge(+g: U32, row: U32, col: U32) -> U32:
  (U32.shrn(g, U32.to_nat((row * 3 + col : U32))) .&. 1 : U32)

def choose(direct: Bool, via: Bool) -> U32:
  match direct via:
    case True{} _:
      1
    case False{} True{}:
      2
    case False{} False{}:
      3

# Distinct endpoints on three vertices need at most one intermediate vertex.
def distance(+g: U32) -> U32:
  choose(U32.is_eq(edge(g, 0, 2), 1),
    Bool.and(U32.is_eq(edge(g, 0, 1), 1), U32.is_eq(edge(g, 1, 2), 1)))

# Valid only for distinct s,t in {0,1,2}; the runner checks the domain.
def relabel(+g: U32, +s: U32, +t: U32) -> U32:
  +m = (3 - s - t : U32)
  (edge(g, s, s) + 2 * edge(g, s, m) + 4 * edge(g, s, t)
    + 8 * edge(g, m, s) + 16 * edge(g, m, m) + 32 * edge(g, m, t)
    + 64 * edge(g, t, s) + 128 * edge(g, t, m) + 256 * edge(g, t, t) : U32)

# Verdicts: 0 unsafe, 1 safe, 2 invalid/unrecoverable answer.
def decode(answer: U32) -> U32:
  match answer:
    case 1:
      0
    case 2:
      0
    case 3:
      1
    case _:
      2

def expected_verdict(direct: Bool, via: Bool) -> U32:
  match direct via:
    case False{} False{}:
      1
    case _ _:
      0

law classification:
  for direct: Bool
  for via: Bool
  {decode(choose(direct, via)) == expected_verdict(direct, via) : U32}

def classification(direct, via):
  match direct via:
    case True{} True{}:
      {==}
    case True{} False{}:
      {==}
    case False{} True{}:
      {==}
    case False{} False{}:
      {==}

# Editable distance encoding, including lossy and invalid encodings.
def encode(answer: U32, one: U32, two: U32, missing: U32) -> U32:
  match answer:
    case 1:
      one
    case 2:
      two
    case _:
      missing

def flip_at_target(is_target: Bool, answer: U32) -> U32:
  match is_target:
    case True{}:
      3
    case False{}:
      answer

def predict(mode: U32, +g: U32, one: U32, two: U32, missing: U32) -> U32:
  match mode:
    case 1:
      3
    case 2:
      flip_at_target(U32.is_eq(g, 4), distance(g))
    case _:
      encode(distance(g), one, two, missing)

def repair(answer: U32, one: U32, two: U32, missing: U32) -> U32:
  match answer:
    case 1:
      one
    case 2:
      two
    case 3:
      missing
    case _:
      0

def live(removed: Bool, answer: U32) -> U32:
  match removed:
    case True{}:
      1
    case False{}:
      decode(answer)
