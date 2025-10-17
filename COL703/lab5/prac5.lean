-- So far, we saw ways in which we can manually apply the correct tactic, and get Lean to proceed to the simplified goal state, and keep doing this till there are no proof obligations left. However, this seems no better than writing proofs by hand (in some cases, like with some inductive proofs, writing proofs in Lean is actually harder, because some implicit assumptions one can make in a handwritten proof cannot be made here). At the end of it all, Lean is a proof assistant, and there should be some ways in which using Lean can make proof search easier than doing it painfully by hand. Can we automate any of the theorem proving that we have done so far?

-- Suppose I want to prove the following (very simple) theorem.

theorem le_less_or_eq : ∀ a b : Nat, a ≤ b → a < b ∨ a = b :=
  by
    intro a b h
  /- At this point, simp does not know what to do, even if we say simp [h] (it does not break down the ≤ further). simp certainly does not work on the goal because of the presence of the or operator. However, neither Or introduction is feasible, because we do not know what the exact relationship between a and b actually is. One might, at this point, try to figure out an inductive spec for ≤ and then induct on h, but this would be tiresome and error-prone. And, ultimately, this relationship between ≤ and < seems quite trivial, and an obvious property of the ordering on the naturals; so surely Lean must have a tactic to handle this? Unfortunately, none of the obvious tactics we have seen so far helps, because this is an inequality (rfl is out) and somehow simp is also powerless in the face of needing to break down ≤.

  Enter the tactic "grind". grind refers to a compendium of facts about equality, inequality and logic, and every time it discovers anything new, it adds it to that compendium. grind can therefore handle notions about inequality and ordering better than either rfl or simp can, and is enough to prove this extremely simple fact! In fact grind can handle a lot of linear arithmetic, not just facts about ≤. This also means that grind would succeed even if we used this tactic without even introducing a and b, which makes it very useful indeed. (You can try this yourself by commenting out the line with the intro above, and see if Lean still presents any pending goals.)
  -/
    grind

  /-
  State and prove a theorem called gt_4 which says that for any natural number, if it is greater than 4, then it must be greater than 0, greater than 1, greater than 2, and greater than 3.
  -/
  -- YOUR ANSWER HERE
theorem gt_4 (m : Nat) (h : m > 4) : m > 0 ∧ m > 1 ∧ m > 2 ∧ m > 3 := by
  grind

  /-
  State and prove a theorem called gt_n which says that for any natural number m, if it is greater than n, then it must be greater than all natural numbers from 0 through n.
  -/
-- theorem gt_n (n m : Nat) (h : m > n) : ∀ k : Nat, k ≤ n → m > k := by
--   intro k hk   -- introduce the arbitrary k and the hypothesis k ≤ n
--   grind
  -- YOUR ANSWER HERE

theorem gt_n : ∀ n m : Nat, m > n → ∀ k : Nat, k ≤ n → m > k := by
  grind

  /-
    A word of caution: While grind is powerful, it's not omnipotent; there are still fairly simple statements it will not be able to prove upfront/without help, in which case it is sensible to fall back on more "traditional" methods which break the proof down into an expected set of subgoals. Try -- for your own edification; you need not include this in the final submission -- to prove the statement that any natural number is either zero or the successor to some natural number.
  -/

  /-
    Recall our definitions of the inductive type day and the function nextDay. One can state and prove a theorem as follows.
  -/

inductive day : Type where
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday
  | saturday
  | sunday

def nextDay : day -> day
  | day.monday => .tuesday
  | .tuesday => .wednesday
  | .wednesday => .thursday
  | .thursday => .friday
  | .friday => .saturday
  | .saturday => .sunday
  | .sunday => .monday

theorem nextDays_first_try :
  nextDay (day.monday) = day.tuesday ∧ nextDay (day.tuesday) = day.wednesday ∧ nextDay (day.wednesday) = day.thursday :=
by
  apply And.intro
  {
    rfl
  }
  {
    apply And.intro
    {
      rfl
    }
    rfl
  }

  /-
    Clearly all that was needed here was to prove each conjunct separately, and each conjunct only needed the 'rfl' tactic. Basic automation would suggest breaking the big "and" down into its two constituent conjuncts, and then applying rfl to these. This can be achieved using the tactic "repeat", which applies a tactic repeatedly as long as the tactic succeeds.
  -/

theorem nextDays_second_try :
  nextDay (day.monday) = day.tuesday ∧ nextDay (day.tuesday) = day.wednesday ∧ nextDay (day.wednesday) = day.thursday :=
by
  repeat apply And.intro
  repeat rfl
  -- At this point we are still left with "case right", which is the second and third conjunct connected with an ∧, and we have to deal with that conjunction now.
  repeat apply And.intro
  repeat rfl

  /-
    However, note that since the ∧ operator is right-associative, repeating does not get us very far, since the second goal is no longer of the form a ∧ b for any a and b, and repeat stops right there. What we want is a kind of a while loop of a repeat, which runs on *both subgoals* till it cannot anymore -- this means that we want it to recursively go down the generated subgoals, and apply the given tactic till it cannot anymore. The recursive version of repeat is called "repeat'" (repeat prime), and it allows us to significantly shorten the proof.
  -/

  theorem nextDays_third_try :
    nextDay (day.monday) = day.tuesday ∧ nextDay (day.tuesday) = day.wednesday ∧ nextDay (day.wednesday) = day.thursday :=
  by
    repeat' apply And.intro
    repeat' rfl

  -- Restate the statement of your theorem gt_4, and name this theorem gt_4_repeat. Use the above kind of repetition to prove it instead of whatever you used earlier. (This might end up being a longer proof, but the idea is to get you used to using repetition.)

  -- YOUR ANSWER HERE
  theorem gt_4_repeat (m : Nat) (h : m > 4) :
  m > 0 ∧ m > 1 ∧ m > 2 ∧ m > 3 := by
  repeat' apply And.intro
  repeat' grind


  -- Note that even in the proof of nextDays_third_try, we ended up having to apply all the And introduction rules first, and then all the rfls. Lean allows a binary operator <;> (read "and then") to join two tactics; tac_a <;> tac_b executes tac_a on the current target goal, and tac_b gets executed on every subgoal that is thus generated. However, we cannot naively join the two tactics in our proof using <;>. Uncomment the following code and step through it to see what happens. The braces are to disambiguate exactly which part of that command is to be repeated.

  /- theorem nextDays_third_and_half :
    nextDay (day.monday) = day.tuesday ∧ nextDay (day.tuesday) = day.wednesday ∧ nextDay (day.wednesday) = day.thursday :=
  by
    repeat' {apply And.intro <;> rfl}
  -/

  -- rfl does not work on the right subgoal, since that goal is of the form a ∧ b, and rfl cannot handle the ∧ operator. However, rfl *does* work on the left subgoal. One can specify a sequence of tactics to try in a particular order using the tactic combinator "first" -- the order is important, and specified top to bottom using case notation. Using "try" outside the repetition allows us to account for the tactic not succeeding (i.e. some -- or all -- goals are still left without simplification). This is basically equivalent to writing first | tac_1 | tac_2 | ... | skip, where skip does nothing but succeeds.

  theorem nextDays_fourth_try :
  nextDay (day.monday) = day.tuesday ∧ nextDay (day.tuesday) = day.wednesday ∧ nextDay (day.wednesday) = day.thursday :=
  by
    try repeat
    first
    | rfl
    | apply And.intro

-- What would happen if we reversed the order of the tactics inside first? (Try to copy over the code and run it if you do not have a good intuition for what might happen; ensure that it is removed in the final submission.) Why does this happen?

/-
If you reverse the order then `first` will try `apply And.intro` before `rfl`.
`first` runs the alternatives in sequence until one succeeds; `repeat` keeps
doing that until no alternative makes progress. With the original order
(`rfl` then `apply And.intro`) `rfl` is attempted on whatever goal `repeat`
produces and it immediately closes equality-goals; `And.intro` is only used
when `rfl` fails (i.e. on a conjunction), so the proof alternates splitting
a conjunction and immediately closing any equality-leaves.
With the reversed order `apply And.intro` will split conjunctions first,
producing all the smaller equality-goals, and then `rfl` will close those
leaves. The end result (the theorem is proven) is the same here, only the
sequence of tactic applications and intermediate goal tree differs.
-/


-- Recall that we defined a function nextWorkingDay. We define it again here.

def nextWorkingDay : day → day
  | day.monday => .tuesday
  | .tuesday => .wednesday
  | .wednesday => .thursday
  | .thursday => .friday
  | .friday => .monday
  | .saturday => .monday
  | .sunday => .monday

-- State and prove a theorem which says that the nextWorkingDay after Friday, Saturday, as well as Sunday is Monday. Try to provide as short a proof as possible, using the techniques from this practical.

-- YOUR ANSWER GOES HERE
theorem nextWorkingDay_Monday :
  nextWorkingDay day.friday = .monday ∧ nextWorkingDay day.saturday = .monday ∧ nextWorkingDay day.sunday = .monday := by
  repeat' apply And.intro
  repeat' rfl
-- For some other useful combinators, read Section 8.1 in the Hitchhiker's Guide to Logical Verification.
