import Base

def lookup(prior: U32, context: U32) -> U32:
  (U32.shrn(prior, U32.to_nat(context)) .&. 1 : U32)

def task(prior: U32, offset: U32) -> U32:
  (prior .^. (15 * offset) : U32)

def relative(+prior: U32) -> U32:
  task(prior, lookup(prior, 0))

def adapt(prior: U32, context: U32, label: U32) -> U32:
  (label .^. lookup(prior, context) : U32)

def answer(prior: U32, offset: U32, context: U32) -> U32:
  (lookup(prior, context) .^. offset : U32)

def recode(prior: U32, mask: U32) -> U32:
  (prior .^. mask : U32)

law relative_pair:
  {relative(5) == relative(10) : U32}

def relative_pair():
  {==}

law task_pair:
  {task(5, 1) == task(10, 0) : U32}

def task_pair():
  {==}

law heldout_answer:
  {answer(relative(5), adapt(relative(5), 1, 1), 2) == 0 : U32}

def heldout_answer():
  {==}
