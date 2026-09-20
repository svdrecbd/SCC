import Base

def bit(b: Bool) -> U32:
  match b:
    case False{}:
      0
    case True{}:
      1

def factor(same: Bool) -> U32:
  match same:
    case False{}:
      1
    case True{}:
      3

def likelihood(candidate: U32, observed: U32) -> U32:
  factor(U32.is_eq(candidate, observed))

def advance_weight(weight: U32, candidate: U32, observed: U32) -> U32:
  (weight * likelihood(candidate, observed) : U32)

def predict(zero: U32, one: U32) -> U32:
  bit(U32.is_gt(one, zero))

# The horizon is at most10, so the offset16 stays positive. Inputs are bits.
def advance_count(count: U32, observed: U32) -> U32:
  (count + 2 * observed - 1 : U32)

def invert(count: U32) -> U32:
  (32 - count : U32)

def predict_count(count: U32) -> U32:
  bit(U32.is_gt(count, 16))

law matched_observation:
  {advance_weight(1, 1, 1) == 3 : U32}

def matched_observation():
  {==}

law opposite_observation:
  {advance_weight(1, 1, 0) == 1 : U32}

def opposite_observation():
  {==}

law count_observation:
  {advance_count(16, 1) == 17 : U32}

def count_observation():
  {==}
