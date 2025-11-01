-- import Mathlib
-- set_option linter.style.longLine false
-- set_option linter.style.commandStart false

-- ------------------------------------------
-- -- LAB SIX: THE FINAL BOSS
-- ------------------------------------------

-- -- This lab will require you to recall everything you learned over the course of the semester by way of Lean techniques (as well as how to define and manipulate logical syntax).

-- -- Define an inductive type called pform, which obeys the rules of syntax of propositional logic formulas. A pform can be an atomic formula (call this constructor at -- which takes as a parameter any string, of type String in Lean), or built using existing pform objects and the constructors myNot, myAnd, myOr, and myImp (standing for the usual operators). You must specify the fact that there is an algorithm to test for the syntactic equality of two pform objects.

-- -- Once you have defined this type, define an inductive object called pftree, which witnesses whether or not there is a proof of a pform φ from a finite set of pforms X (choose the type for this appropriately!) according to the rules given in Table 1 of https://www.cmi.ac.in/~spsuresh/pdfs/jlc2020-tr.pdf. This is a proof system in *intuitionistic* propositional logic, which means that it does not admit the law of excluded middle (i.e. φ ∨ ¬φ is not a tautology, and since this proof system is sound, cannot be proven without assumptions.) It also means, consequently, that anything proven must follow from the assumptions via a proof; there are no "free" axioms. Intuitionism has a long history, and goes back to Brouwer, and underpins much of theorem proving -- Lean's own underlying theory is intuitionistic.

-- -- In order to define a finite set, you will need to import Mathlib. Mathlib is THE Lean library, in that it includes a lot of handy constructs and tactics. In particular, it also includes the constructor called Finset, which takes as input a type, and spits out a finite set of said type (much like the List constructor). If you are using VSCode, if you type Finset followed by "." (exactly like with List) you can see the various methods and theorems you have access to under the Finset constructor. This is why we require the lines included above, which import the Mathlib library, and disable some irritating linters about line length and where a command should start.

-- -- Finally, define a theorem called mono_prf, which says that if there is a proof tree witnessing a proof of a pform φ from a Finset X of pforms, then there is a proof tree witnessing a proof of φ from a set which is a superset of X. Recall that you can state this in multiple ways; use whichever way seems most amenable to proving this statement. (This is what we have proved in class and called Monotonicity.) Submit your entire answer below this sequence of comments. In general, one wants to show not just Monotonicity, but various other desirable properties of any given proof system, including whether inference in them is decidable (and if it, how efficiently it can be done), like we are doing in that paper linked above.


import Mathlib

set_option linter.style.longLine false
set_option linter.style.commandStart false

-- Inductive syntax for propositional formulas (pform)

inductive pform : Type
| at    : String → pform                    -- atomic formula
| myNot : pform → pform                     -- ¬φ
| myAnd : pform → pform → pform             -- φ ∧ ψ
| myOr  : pform → pform → pform             -- φ ∨ ψ
| myImp : pform → pform → pform             -- φ → ψ
deriving DecidableEq, Repr

open pform

-- Intuitionistic proof trees following Table 1 (system IL).

inductive pftree : Finset pform → pform → Type
| ax {X φ} (h : φ ∈ X) : pftree X φ
  
| not_i {X α β} (h₁ : pftree (X ∪ {α}) β) (h₂ : pftree (X ∪ {α}) (myNot β)) : 
    pftree X (myNot α)
  
| not_e {X α β} (h₁ : pftree X β) (h₂ : pftree X (myNot β)) : 
    pftree X α
  
| and_i {X α β} (h₁ : pftree X α) (h₂ : pftree X β) : 
    pftree X (myAnd α β)
  
| and_e {X α₀ α₁} (h : pftree X (myAnd α₀ α₁)) (j : Bool) : 
    pftree X (if j then α₀ else α₁)
  
| or_i {X α₀ α₁} (j : Bool) (h : pftree X (if j then α₀ else α₁)) : 
    pftree X (myOr α₀ α₁)
  
| or_e {X α β γ} (h₁ : pftree X (myOr α β)) 
    (h₂ : pftree (X ∪ {α}) γ) (h₃ : pftree (X ∪ {β}) γ) : 
    pftree X γ
  
| imp_i {X α β} (h : pftree (X ∪ {α}) β) : 
    pftree X (myImp α β)
  
| imp_e {X α β} (h₁ : pftree X (myImp α β)) (h₂ : pftree X α) : 
    pftree X β

open pftree

-- Monotonicity: use match/structural recursion

-- def mono_prf {X Y : Finset pform} {φ : pform} 
--     (h₁ : pftree X φ) (hsub : X ⊆ Y) : pftree Y φ := 
--   match h₁ with
--   | ax h => 
--       ax (hsub h)
--   | @not_i X α _β hβ hNotβ => 
--       not_i 
--         (mono_prf hβ (Finset.union_subset_union_left hsub))
--         (mono_prf hNotβ (Finset.union_subset_union_left hsub))
--   | not_e hβ hNotβ => 
--       not_e (mono_prf hβ hsub) (mono_prf hNotβ hsub)
--   | and_i hα hβ => 
--       and_i (mono_prf hα hsub) (mono_prf hβ hsub)
--   | and_e h j => 
--       and_e (mono_prf h hsub) j
--   | or_i j h => 
--       or_i j (mono_prf h hsub)
--   | @or_e X _α _β γ hOr hα hβ => 
--       or_e (mono_prf hOr hsub)
--         (mono_prf hα (Finset.union_subset_union_left hsub))
--         (mono_prf hβ (Finset.union_subset_union_left hsub))
--   | @imp_i X α β h => 
--       imp_i (mono_prf h (Finset.union_subset_union_left hsub))
--   | imp_e hImp hα => 
--       imp_e (mono_prf hImp hsub) (mono_prf hα hsub)

-- Monotonicity as a theorem (propositional statement)
-- theorem mono_prf {X Y : Finset pform} {φ : pform} 
--     (h₁ : pftree X φ) (hsub : X ⊆ Y) : Nonempty (pftree Y φ) := 
--   ⟨match h₁ with
--   | ax h => 
--       ax (hsub h)
--   | @not_i X α _β hβ hNotβ => 
--       not_i 
--         (mono_prf hβ (Finset.union_subset_union_left hsub) |>.some)
--         (mono_prf hNotβ (Finset.union_subset_union_left hsub) |>.some)
--   | not_e hβ hNotβ => 
--       not_e (mono_prf hβ hsub |>.some) (mono_prf hNotβ hsub |>.some)
--   | and_i hα hβ => 
--       and_i (mono_prf hα hsub |>.some) (mono_prf hβ hsub |>.some)
--   | and_e h j => 
--       and_e (mono_prf h hsub |>.some) j
--   | or_i j h => 
--       or_i j (mono_prf h hsub |>.some)
--   | @or_e X _α _β γ hOr hα hβ => 
--       or_e (mono_prf hOr hsub |>.some)
--         (mono_prf hα (Finset.union_subset_union_left hsub) |>.some)
--         (mono_prf hβ (Finset.union_subset_union_left hsub) |>.some)
--   | @imp_i X α β h => 
--       imp_i (mono_prf h (Finset.union_subset_union_left hsub) |>.some)
--   | imp_e hImp hα => 
--       imp_e (mono_prf hImp hsub |>.some) (mono_prf hα hsub |>.some)⟩


-- Monotonicity theorem (existence statement)
theorem mono_prf {X Y : Finset pform} {φ : pform} 
    (h₁ : pftree X φ) (hsub : X ⊆ Y) : ∃ (_ : pftree Y φ), True := 
  ⟨mono_prf_fn h₁ hsub, trivial⟩

-- Helper function
def mono_prf_fn {X Y : Finset pform} {φ : pform} 
    (h₁ : pftree X φ) (hsub : X ⊆ Y) : pftree Y φ := 
  match h₁ with
  | ax h => 
      ax (hsub h)
  | @not_i X α _β hβ hNotβ => 
      not_i 
        (mono_prf_fn hβ (Finset.union_subset_union_left hsub))
        (mono_prf_fn hNotβ (Finset.union_subset_union_left hsub))
  | not_e hβ hNotβ => 
      not_e (mono_prf_fn hβ hsub) (mono_prf_fn hNotβ hsub)
  | and_i hα hβ => 
      and_i (mono_prf_fn hα hsub) (mono_prf_fn hβ hsub)
  | and_e h j => 
      and_e (mono_prf_fn h hsub) j
  | or_i j h => 
      or_i j (mono_prf_fn h hsub)
  | @or_e X _α _β γ hOr hα hβ => 
      or_e (mono_prf_fn hOr hsub)
        (mono_prf_fn hα (Finset.union_subset_union_left hsub))
        (mono_prf_fn hβ (Finset.union_subset_union_left hsub))
  | @imp_i X α β h => 
      imp_i (mono_prf_fn h (Finset.union_subset_union_left hsub))
  | imp_e hImp hα => 
      imp_e (mono_prf_fn hImp hsub) (mono_prf_fn hα hsub)