import Std

namespace SafetyCapabilityCoupling

/-- A total program with one private bit and an arbitrary public input. -/
abbrev PrivateProgram (Input : Type) := Bool → Input → Bool

/-- Termination-insensitive output confidentiality for total programs. -/
def Confidential {Input : Type} (program : PrivateProgram Input) : Prop :=
  ∀ input firstSecret secondSecret,
    program firstSecret input = program secondSecret input

/-- A computable predicate controls whether the private bit is disclosed. -/
def guardedDisclosure {Input : Type} (predicate : Input → Bool) : PrivateProgram Input :=
  fun secret input => if predicate input then false else secret

/-- Every proof of the ordinary universal predicate supplies a confidentiality proof. -/
theorem predicateProofToConfidentiality {Input : Type} (predicate : Input → Bool)
    (valid : ∀ input, predicate input = true) :
    Confidential (guardedDisclosure predicate) := by
  intro input firstSecret secondSecret
  simp [guardedDisclosure, valid input]

/-- A confidentiality proof contains enough to recover the universal predicate. -/
theorem confidentialityToPredicateProof {Input : Type} (predicate : Input → Bool)
    (secure : Confidential (guardedDisclosure predicate)) :
    ∀ input, predicate input = true := by
  intro input
  have comparison := secure input false true
  cases value : predicate input with
  | false => simp [guardedDisclosure, value] at comparison
  | true => rfl

theorem confidentialityIffPredicate {Input : Type} (predicate : Input → Bool) :
    Confidential (guardedDisclosure predicate) ↔ ∀ input, predicate input = true :=
  ⟨confidentialityToPredicateProof predicate, predicateProofToConfidentiality predicate⟩

/-- A false predicate instance is an executable disclosure witness. -/
theorem predicateCounterexampleToDisclosure {Input : Type} (predicate : Input → Bool)
    (input : Input) (failure : predicate input = false) :
    guardedDisclosure predicate false input ≠ guardedDisclosure predicate true input := by
  simp [guardedDisclosure, failure]

/-- Ordinary program comparison supplies confidentiality for a general program. -/
theorem confidentialityIffComparison {Input : Type} (program : PrivateProgram Input) :
    Confidential program ↔ ∀ input, program false input = program true input := by
  constructor
  · intro secure input
    exact secure input false true
  · intro comparison input firstSecret secondSecret
    cases firstSecret <;> cases secondSecret <;> simp_all

/-- The protected family includes every total computable Boolean predicate. -/
def comparisonPredicate {Input : Type} (program : PrivateProgram Input) : Input → Bool :=
  fun input => program false input == program true input

theorem comparisonPredicateIffConfidentiality {Input : Type} (program : PrivateProgram Input) :
    (∀ input, comparisonPredicate program input = true) ↔ Confidential program := by
  simp only [comparisonPredicate, beq_iff_eq]
  exact (confidentialityIffComparison program).symm

-- These examples check both directions without provisioning a policy secret.
example : Confidential (guardedDisclosure (fun (_ : Nat) => true)) := by
  apply predicateProofToConfidentiality
  intro input
  rfl

example : ¬ Confidential (guardedDisclosure (fun (_ : Nat) => false)) := by
  intro secure
  have invalid := confidentialityToPredicateProof _ secure 0
  cases invalid

#print axioms confidentialityIffPredicate
#print axioms predicateCounterexampleToDisclosure
#print axioms comparisonPredicateIffConfidentiality

end SafetyCapabilityCoupling
