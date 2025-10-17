/-
Recall that we defined inductive types (the type "day" that we started our Lean sessions with was defined inductively, although trivially so; the myList datatype used induction much more substantially). We can also define inductive predicates over inductive datatypes. A simple inductive predicate is BelongsTo, which denotes membership, and we can define as follows.
-/

inductive BelongsTo {α : Type} : α → List α → Prop
where
  | isHead (h : α) (tl : List α) : BelongsTo h (h :: tl)
  | inTail (h y : α) (tl : List α) : BelongsTo h tl → BelongsTo h (y :: tl)

/-
  This predicate is defined for any type α, over an object of type α and an object of type List α. There are two possibilities, each named by a constructor (isHead and inTail). The isHead constructor deals with the case when the object h appears at the head of the list tl, and therefore we can state that h belongs to the list h::tl. The inTail constructor deals with the case when the object h appears somewhere "inside" the list tl, but not at its head. So the constructor takes as arguments not just h, but also y (another object of type α), and the list tl, and states that if h belongs to the list tl, then h also belongs to the list y::tl.
-/


/-
We can now state a theorem regarding what it means for an object x of a type to belong to a list l of that type, namely, that there must exist two lists (either -- or both -- could be empty) bookending that object x inside the list l.
-/

theorem in_surround : ∀ α : Type, ∀ x : α, ∀ l : List α,
  BelongsTo x l → ∃ front back : List α, l = front ++ [x] ++ back :=
  by
    intro α x l h
    /-
      At this point, in a pen and paper proof, one would proceed by doing a case analysis on what exactly causes BelongsTo x l to be true (i.e. does x appear at the head of the list, or somewhere in the "middle"). One could try to do this by applying the "cases" tactic to h. Uncomment the following block of code to see why this will not work. Note that when we need to instantiate an existentially quantified variable, we use the keyword "exists", followed by the appropriate value for that variable which would make the statement true.
    -/
    /-
    cases h with
    | isHead tl =>
                  exists []
                  exists tl
    | inTail y tl htl =>
                  exists [y]
                  cases htl with ...
    -/
    /-
      We get stuck in the inductive step, because it is unclear exactly *where* x is when it is not the precise head of the list. It might be the case that the "front" list is just [y], or that it is [y]++[some prefix of tl] if x happens to be in the middle of the list tl. We would need to keep doing cases on the hypotheses of the form BelongsTo x tl, for smaller and smaller tl. But we start with an arbitrary l, so how do we know how long we must apply cases for? We do not, upfront! So we turn to the usual technique we know when we have an inductively defined object of arbitrary size.
    -/
    induction h with
    | isHead tl =>
                exists []
                exists tl
    | inTail y tl htl Ih =>
    /-
        At this point we have an existentially quantified induction hypothesis statement (which we are calling Ih). A simple pattern matching tells us that the values for "front" and "back" that will satisfy the goal depend on the values for "front" and "back" that witness the truth of Ih (i.e. can be assigned to the quantified variables to make the qf body of Ih true). In order to finish up the proof with exists, however, we need access to the *names* of these values for "front" and "back" in Ih. In order to obtain these names, we turn to the tactic 'cases' (which we already saw earlier, in a different context, to simplify a hypothesis of the form f x = f y to give us x = y), which can help break down an existentially quantified hypothesis. Writing "cases Ih" will break down the statement by stripping off the first existential quantification at the head, giving the variable a name, and giving the rest of the statement another name -- however, these names are not accessible for further use in proofs! In order to enforce names of our choosing, we need to follow "cases Ih" by "case intro" and then give a name to the quantified variable as well as the statement that follows. If, as in our current case, there are more than one quantification, we have to do cases followed by case intro as many times as there are existentially quantified variables. This yields the following structure.
    -/
                            -- cases Ih
                            -- case intro fr Ihf =>
                            --   cases Ihf
                            --   case intro bk Ihfb =>
    /-
        Obviously this becomes a pain to write if one has a deep sequence of existential quantification, so an alternative is to write the following. obtain allows us to specify the structure of the existentially quantified statement by giving names to witnesses for each of the quantified variables and the qf body all in one go, using appropriate nestings of ⟨ (\langle) and ⟩ (\rangle) to indicate scope. (If you wish, you can comment out the above four lines with cases etc, and uncomment the following line with obtain. The effect is the same, and the remaining proof obligation remains exactly the same.)

    -/
    obtain ⟨fr, ⟨bk, Ihfb⟩⟩ := Ih
    /-  Once we are done with this, the rest of the proof becomes easy. The exact position of x inside tl is given by the bookends fr and bk, and so the new bookend values become y prepended to fr, and bk itself. Remove the following sorry and finish the proof. -/
    exists (y :: fr)
    exists bk
    show y :: tl = (y :: fr) ++ [x] ++ bk
    rw [Ihfb]
    simp

/-
  State and prove a theorem called in_append that says that for any type, any object x of said type, and any lists l1 and l2 of said type, if x belongs to l1, then x must belong to l1 ++ l2.

  If you wish to apply a constructor for an inductive predicate to break down a target expression, you may use the "apply <predicate name>.<constructor name>" form. For example, if I wanted to convert a goal of the form BelongsTo x (hd :: tl) to BelongsTo x tl, I would appeal to the inTail constructor of BelongsTo, and I would write "apply BelongsTo.inTail".
-/

-- YOUR ANSWER GOES HERE
theorem in_append : ∀ α : Type, ∀ x : α, ∀ l1 l2 : List α,
  BelongsTo x l1 → BelongsTo x (l1 ++ l2) :=
  by
    intro α x l1 l2 h
    induction h with
    | isHead tl =>
      apply BelongsTo.isHead
    | inTail y tl htl Ih =>
      apply BelongsTo.inTail
      exact Ih

/-
  State and prove another theorem called in_append2 that says that for any type, any object x of said type, and any lists l1 and l2 of said type, if x belongs to *l2*, then x must belong to l1 ++ l2. Can such a small difference in the statement make a difference in the proofs?
-/

-- YOUR ANSWER GOES HERE
theorem in_append2 : ∀ α : Type, ∀ x : α, ∀ l1 l2 : List α,
  BelongsTo x l2 → BelongsTo x (l1 ++ l2) :=
  by
    intro α x l1 l2 h
    induction l1 with
    | nil =>
      exact h
    | cons y tl Ih =>
      -- (y :: tl) ++ l2 = y :: (tl ++ l2)
      apply BelongsTo.inTail
      exact Ih

/-
  Recall that we defined a predicate called uniq, which took as argument a list (of arbitrary type) and returned true if the elements of the list were all distinct, and false otherwise. One possible definition for uniq is given below.

  If you have seen OCaml, the higher order functions foldl and map should be familiar to you -- in fact, they also exist in Python! For those of you to whom these are new, such higher-order functions (functions which can take other functions as inputs) are the mainstay of functional programming languages like Haskell, OCaml etc. A useful place to start, which has some common, useful higher-order functions is https://en.wikipedia.org/wiki/Higher-order_function. (A natural follow-up question to ask is why one might need two folds!)
-/

def uniq {α : Type} [BEq α] : List α → Bool
  | [] => true
  | (hd :: tl) => List.foldl (Bool.and) (uniq tl) (List.map (λx ↦ (if x == hd then false else true)) tl)

/-
  Instead of such a functional definition, define an inductive predicate called uniq_ind instead, along the lines of BelongsTo.
-/

-- YOUR ANSWER GOES HERE

inductive uniq_ind {α : Type} : List α → Prop
where
  | nil : uniq_ind []
  | cons (h : α) (tl : List α) : ¬(BelongsTo h tl) → uniq_ind tl → uniq_ind (h :: tl)

/-
  State and prove a theorem called uniq_sublists which uses the uniq_ind predicate and says that if a list (of any arbitrary type) is composed of distinct elements, it can be split into two sublists, each of which is composed of distinct elements. (Is the converse true?) You might need a helper theorem; if you do, feel free to name it something appropriate and state and prove it before using it.
-/

-- YOUR ANSWER GOES HERE

theorem uniq_sublists {α : Type} :
  ∀ (l : List α), uniq_ind l → ∃ l1 l2 : List α, l = l1 ++ l2 ∧ uniq_ind l1 ∧ uniq_ind l2 := by
  intro l h
  exists l, []
  constructor
  · simp
  constructor
  · exact h
  · apply uniq_ind.nil

/-Converse is False l1=[1] l2=[1], l=l1++l2=[1,1]-/
/-
  For more examples of inductive predicates and inductive proofs using them, go through Sections 6.3 and 6.6 of the Hitchhiker's Guide to Logical Verification.
-/
