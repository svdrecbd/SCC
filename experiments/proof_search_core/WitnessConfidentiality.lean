import ConfidentialityReduction

namespace SafetyCapabilityCoupling

/-- Public, total certificate verification controls disclosure of a private bit. -/
def certificateDisclosure {Certificate : Type}
    (accept : Certificate → Bool) : PrivateProgram Certificate :=
  fun secret certificate => if accept certificate then secret else false

/-- An accepted useful certificate is exactly a disclosure witness. -/
theorem acceptedIffDisclosure {Certificate : Type}
    (accept : Certificate → Bool) (certificate : Certificate) :
    accept certificate = true ↔
      certificateDisclosure accept false certificate ≠
        certificateDisclosure accept true certificate := by
  cases value : accept certificate <;> simp [certificateDisclosure, value]

/-- Absence of all accepted certificates is exactly confidentiality. -/
theorem confidentialIffNoCertificate {Certificate : Type}
    (accept : Certificate → Bool) :
    Confidential (certificateDisclosure accept) ↔
      ∀ certificate, accept certificate = false := by
  constructor
  · intro secure certificate
    have comparison := secure certificate false true
    cases value : accept certificate with
    | false => rfl
    | true => simp [certificateDisclosure, value] at comparison
  · intro rejected certificate firstSecret secondSecret
    simp [certificateDisclosure, rejected certificate]

/-- A native confidentiality assessor returns a concrete input causing disclosure. -/
abbrev DisclosureCertificate {Certificate : Type} (accept : Certificate → Bool) :=
  { certificate : Certificate //
    certificateDisclosure accept false certificate ≠
      certificateDisclosure accept true certificate }

/-- The useful interface obtains its certificate from the protected assessor. -/
def usefulFromDisclosureAssessor {Certificate : Type} {accept : Certificate → Bool}
    (result : Option (DisclosureCertificate accept)) :
    Option { certificate : Certificate // accept certificate = true } :=
  result.map fun certificate =>
    ⟨certificate.val, (acceptedIffDisclosure accept certificate.val).mpr certificate.property⟩

/-- Every replacement useful solver also supplies a protected disclosure assessor. -/
def disclosureAssessorFromUseful {Certificate : Type} {accept : Certificate → Bool}
    (result : Option { certificate : Certificate // accept certificate = true }) :
    Option (DisclosureCertificate accept) :=
  result.map fun certificate =>
    ⟨certificate.val, (acceptedIffDisclosure accept certificate.val).mp certificate.property⟩

theorem usefulSuccessPreserved {Certificate : Type} {accept : Certificate → Bool}
    (result : Option (DisclosureCertificate accept)) :
    (usefulFromDisclosureAssessor result).isSome = result.isSome := by
  cases result <;> rfl

theorem replacementSuccessPreserved {Certificate : Type} {accept : Certificate → Bool}
    (result : Option { certificate : Certificate // accept certificate = true }) :
    (disclosureAssessorFromUseful result).isSome = result.isSome := by
  cases result <;> rfl

-- These controls verify polarity: an accepting verifier is unsafe, rejecting safe.
example : ¬ Confidential (certificateDisclosure (fun (_ : Nat) => true)) := by
  intro secure
  have impossible := (confidentialIffNoCertificate _).mp secure 0
  cases impossible

example : Confidential (certificateDisclosure (fun (_ : Nat) => false)) := by
  exact (confidentialIffNoCertificate _).mpr (fun _ => rfl)

#print axioms acceptedIffDisclosure
#print axioms confidentialIffNoCertificate
#print axioms usefulSuccessPreserved
#print axioms replacementSuccessPreserved

end SafetyCapabilityCoupling
