import Mathlib
set_option linter.style.longLine false
set_option linter.style.commandStart false

------------------------------------------
-- LAB SIX: THE FINAL BOSS
------------------------------------------

-- This lab will require you to recall everything you learned over the course of the semester by way of Lean techniques (as well as how to define and manipulate logical syntax).

-- Define an inductive type called pform, which obeys the rules of syntax of propositional logic formulas. A pform can be an atomic formula (call this constructor at -- which takes as a parameter any string, of type String in Lean), or built using existing pform objects and the constructors myNot, myAnd, myOr, and myImp (standing for the usual operators). You must specify the fact that there is an algorithm to test for the syntactic equality of two pform objects.

inductive pform : Type where 
  | atom : String → pform 
  | myNot : pform → pform
  | myAnd : pform → pform → pform
  | myOr : pform → pform → pform 
  | myImp : pform → pform → pform  
deriving DecidableEq, Repr


-- Once you have defined this type, define an inductive object called pftree, which witnesses whether or not there is a proof of a pform φ from a finite set of pforms X (choose the type for this appropriately!) according to the rules given in Table 1 of https://www.cmi.ac.in/~spsuresh/pdfs/jlc2020-tr.pdf. This is a proof system in *intuitionistic* propositional logic, which means that it does not admit the law of excluded middle (i.e. φ ∨ ¬φ is not a tautology, and since this proof system is sound, cannot be proven without assumptions.) It also means, consequently, that anything proven must follow from the assumptions via a proof; there are no "free" axioms. Intuitionism has a long history, and goes back to Brouwer, and underpins much of theorem proving -- Lean's own underlying theory is intuitionistic.

-- In order to define a finite set, you will need to import Mathlib. Mathlib is THE Lean library, in that it includes a lot of handy constructs and tactics. In particular, it also includes the constructor called Finset, which takes as input a type, and spits out a finite set of said type (much like the List constructor). If you are using VSCode, if you type Finset followed by "." (exactly like with List) you can see the various methods and theorems you have access to under the Finset constructor. This is why we require the lines included above, which import the Mathlib library, and disable some irritating linters about line length and where a command should start.


open pform

inductive pftree : Finset pform → pform → Prop where
  -- Axiom rule:
  | ax (Γ : Finset pform) (φ : pform) (h : φ ∈ Γ) :pftree Γ φ

  -- ∧-introduction: 
  | and_i {Γ: Finset pform} {φ ψ: pform} (pf_φ : pftree Γ φ) (pf_ψ : pftree Γ ψ) : pftree Γ (myAnd φ ψ)

  -- ∧-elimination left:
  | and_el {Γ: Finset pform} {φ ψ: pform} (pf_and : pftree Γ (myAnd φ ψ)) : pftree Γ φ

  -- ∧-elimination right
  | and_er {Γ: Finset pform} {φ ψ: pform} (pf_and : pftree Γ (myAnd φ ψ)) :pftree Γ ψ

  -- ∨-introduction left
  | or_il {Γ: Finset pform} {φ ψ: pform} (pf_φ : pftree Γ φ) : pftree Γ (myOr φ ψ)

  -- ∨-introduction right
  | or_ir {Γ: Finset pform} {φ ψ: pform} (pf_ψ : pftree Γ ψ) : pftree Γ (myOr φ ψ)

  -- ~e
  | not_e {Γ: Finset pform} {φ ψ: pform}  (pf1: pftree Γ φ) (pf2 : pftree Γ (myNot φ)) : pftree Γ ψ

  -- p
  | imp_p {Γ: Finset pform} {φ ψ: pform} (pf1: pftree Γ ψ) : pftree Γ (myImp φ ψ)

  -- imp_elimination (modus ponens)
  | imp_e {Γ φ ψ} (pf1: pftree Γ φ) (pf2: pftree Γ (myImp φ ψ)) : pftree Γ ψ
  -- ∨ Elimination
  
  | or_e {Γ φ ψ θ} (pf_or : pftree Γ (myOr φ ψ)) 
    (pf_φ_th : pftree (insert φ Γ) θ) (pf_ψ_th : pftree (insert ψ Γ) θ) : pftree Γ θ


  -- Implication Introduction
  | imp_i {Γ φ ψ} (pf_imp : pftree (insert φ Γ) ψ) : pftree Γ (myImp φ ψ)



  -- not introduction 
  | not_i {Γ φ ψ} (pf1: pftree (insert φ Γ) ψ) (pf2: pftree (insert φ Γ) (myNot ψ)) : pftree Γ (myNot φ)



-- Finally, define a theorem called mono_prf, which says that if there is a proof tree witnessing a proof of a pform φ from a Finset X of pforms, then there is a proof tree witnessing a proof of φ from a set which is a superset of X. Recall that you can state this in multiple ways; use whichever way seems most amenable to proving this statement. (This is what we have proved in class and called Monotonicity.) Submit your entire answer below this sequence of comments. In general, one wants to show not just Monotonicity, but various other desirable properties of any given proof system, including whether inference in them is decidable (and if it, how efficiently it can be done), like we are doing in that paper linked above.

-- Monotonicity Theorem: If X ⊢ φ, then Y ⊢ φ for any superset Y of X (X ⊆ Y).
theorem mono_prf {X Y : Finset pform} {φ : pform} (h_sub : X ⊆ Y) (pf : pftree X φ) :
  pftree Y φ := by
  induction pf generalizing Y
  case ax Γ ψ hin =>
    exact pftree.ax Y ψ (h_sub hin)

  case and_i Γ ψ χ pf1 pf2 ih1 ih2 =>
    exact pftree.and_i (ih1 h_sub) (ih2 h_sub)

  case and_el Γ ψ χ pf_and ih =>
    exact pftree.and_el (ih h_sub)

  case and_er Γ ψ χ pf_and ih =>
    exact pftree.and_er (ih h_sub)

  case or_il Γ ψ χ pf ih =>
    exact pftree.or_il (ih h_sub)

  case or_ir Γ ψ χ pf ih =>
    exact pftree.or_ir (ih h_sub)

  case not_e Γ ψ θ pf1 pf2 ih1 ih2 =>
    exact pftree.not_e (ih1 h_sub) (ih2 h_sub)

  case imp_p Γ α β pf ih =>
    exact pftree.imp_p (ih h_sub)

  case imp_e Γ α β pf1 pf2 ih1 ih2 =>
    exact pftree.imp_e (ih1 h_sub) (ih2 h_sub)

  case or_e Γ α β θ pf_or pf_α_th pf_β_th ih_or ih_α ih_β =>
    have h1 : insert α Γ ⊆ insert α Y := Finset.insert_subset_insert α h_sub
    have h2 : insert β Γ ⊆ insert β Y := Finset.insert_subset_insert β h_sub
    exact pftree.or_e (ih_or h_sub)
                      (@ih_α (insert α Y) h1)
                      (@ih_β (insert β Y) h2)

  case imp_i Γ α β pf_imp ih =>
    exact pftree.imp_i (@ih (insert α Y) (Finset.insert_subset_insert α h_sub))

  case not_i Γ α β pf1 pf2 ih1 ih2 =>
    let h' := Finset.insert_subset_insert α h_sub
    exact pftree.not_i (@ih1 (insert α Y) h') (@ih2 (insert α Y) h')
