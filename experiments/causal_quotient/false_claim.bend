import Base
import ./kernel.bend as K

law false_hazard_erased:
  {K.protected(K.step(1, K.S{0, 0, 1}, 1)) == 0 : U32}

def false_hazard_erased():
  {==}
