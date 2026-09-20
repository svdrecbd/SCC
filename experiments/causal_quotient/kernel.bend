import Base

type State is Data:
  S{phase: U32, useful: U32, hazard: U32}

def protected(s: State) -> U32:
  match s:
    case S{p, u, h}:
      h

def extra(enabled: Bool, h: U32) -> U32:
  match enabled:
    case False{}:
      0
    case True{}:
      h

def phase_extra(p: U32, h: U32) -> U32:
  extra(U32.is_eq(p, 0), h)

def step(mode: U32, s: State, +action: U32) -> State:
  match s:
    case S{+p, u, +h}:
      S{((p + 1) % 4 : U32),
        ((u + action + mode * action * phase_extra(p, h)) % 2 : U32), h}

def encode(s: State) -> U32:
  match s:
    case S{p, u, h}:
      (p + 4 * u + 8 * h : U32)

def step_code(mode: U32, +code: U32, action: U32) -> U32:
  encode(step(mode, S{(code % 4 : U32), ((code / 4) % 2 : U32), ((code / 8) % 2 : U32)}, action))

law hazard_preserved:
  for mode: U32
  for s: State
  for action: U32
  {protected(step(mode, s, action)) == protected(s) : U32}

def hazard_preserved(mode, s, action):
  match s:
    case S{p, u, h}:
      {==}
