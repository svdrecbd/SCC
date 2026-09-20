import Base

def pop4(+x: U32) -> U32:
  ((x .&. 1) + (U32.shrn(x, 1n) .&. 1)
    + (U32.shrn(x, 2n) .&. 1) + (U32.shrn(x, 3n) .&. 1) : U32)

def pop16(+x: U32) -> U32:
  (pop4(x) + pop4(U32.shrn(x, 4n))
    + pop4(U32.shrn(x, 8n)) + pop4(U32.shrn(x, 12n)) : U32)

def parity(z: U32, q: U32) -> U32:
  (pop4((z .&. q : U32)) .&. 1 : U32)

def pick_max(ge: Bool, a: U32, b: U32) -> U32:
  match ge:
    case True{}:
      a
    case False{}:
      b

def maximum(+a: U32, +b: U32) -> U32:
  pick_max(U32.is_ge(a, b), a, b)

# The two buckets contain all sixteen source values. Each nonzero query has
# eight labels of each kind. Domain: encoder, target are 16-bit truth tables.
def correct(+encoder: U32, +target: U32) -> U32:
  +n = pop16(encoder)
  +a = pop16((encoder .&. target : U32))
  (maximum(a, (n - a : U32))
    + maximum((8 - a : U32), (8 + a - n : U32)) : U32)

def bucket_correct(+encoder: U32, target: U32) -> U32:
  +n = pop16(encoder)
  +a = pop16((encoder .&. target : U32))
  maximum(a, (n - a : U32))

law constant_has_chance:
  {correct(0, 43690) == 8 : U32}

def constant_has_chance():
  {==}

law same_query_is_exact:
  {correct(43690, 43690) == 16 : U32}

def same_query_is_exact():
  {==}
