-- Using everything we have learned so far, define a function called myRev which takes as input a list of any type, and reverses it.

-- YOUR ANSWER GOES HERE

def myRev (α : Type) : List α → List α
  | []      => []
  | x :: xs => (myRev α xs) ++ [x]

/-
#eval myRev Nat [1, 2, 3]

theorem myRev_example :
  myRev Nat [1, 2, 3] = [3, 2, 1] := by rfl
-/


-- State and prove a theorem called myrev_idemp which states that applying myRev to a list twice yields the original list back. You might need to state and prove a helper theorem; feel free to do so. Hint: Recall that the "simp" tactic can take arguments, using which it can try to simplify the goal. These arguments can be any other predefined object that allows one to rewrite. Also, if the goal immediately follows from some constructor as part of an inductively-defined object, you can use the tactic "constructor".

-- YOUR ANSWER GOES HERE

theorem myRev_append {α : Type} : ∀ (l r : List α), myRev α (l ++ r) = myRev α r ++ myRev α l := by
  intro l
  induction l with
  | nil =>
    intro r
    simp [myRev]
  | cons x xs ih =>
    intro r
    -- myRev (x :: xs ++ r) = (myRev (xs ++ r)) ++ [x]
    -- apply IH to myRev (xs ++ r), then use associativity of ++ (simp knows it)
    simp [myRev, ih]

theorem myrev_idemp {α : Type} : ∀ (l : List α), myRev α (myRev α l) = l := by
  intro l
  induction l with
  | nil => simp [myRev]
  | cons x xs ih =>
    -- myRev (x :: xs) = myRev xs ++ [x], then use myRev_append and the IH
    simp [myRev, myRev_append, ih]

/-
    If you did this correctly, you would see that you needed to account for one case where myRev yielded an empty list, and one where it returned a non-empty list. Often, you might need to consider multiple possible cases for the structure of an inductively-defined object, and do different things based on the structure. Lean has a keyword called "match" which lets you do that. The syntax is the same as that for the "induction" keyword, except there is no inductive hypothesis (and you will need to write List.nil and List.cons instead of it automatically inferring the List type); "match" merely makes an attempt at a proof by case analysis, not a proof by induction.

    Recall that in a non-empty list of the sort (n :: tl), the element n is the "head" and the rest of the list (tl) is the "tail". What is the head of an empty list? Examine the type of the List.head function.
-/

#check List.head

/-
    Note that this is what is called a "dependent type". The type of the result depends on the value of the argument (in this case the List object whose head is being asked for). Dependent types are very handy; one can use them for defining partial functions, with the partial-ness clearly specified. One could define a division function on integers by specifying that the denominator must not be 0, for example.

    What if we wanted to specify a total function for division on integers? We would be able to specify the function as is for all values except when the denominator was zero -- in that case, we would need to specify a "default" value (often, general programming languages would call this "NaN" -- not a number).

    Suppose we wanted to define a total version of the head function similarly. Let us first try to do this for the List Nat type. We would need to define a default value; let us say that we return 0 if one invokes this function on an empty list, and the actual head otherwise. We can define this as follows.
-/

def total_head_nat : List Nat → Nat
  | [] => 0
  | (n :: _) => n

/-
  Ideally we want such a function for *any* list of *any* type.
  We try to define it as follows. The function is defined with an *implicit* type α (with braces around it rather than parentheses, like we used in myLength, where we intended the type to be *explicit*), where it takes an object of List α as the input, and an *explicit* default argument of type α, which is to be returned in the case of the input being an empty list.
-/

def total_head {α : Type} (default : α) : List α → α
  | [] => default
  | (h :: _) => h

/-
  Think: Is there a situation where such a function definition would fail? Some particular type? If there is some such, what more would we need to enforce that it did not fail in that case?

  If we want to enforce conditions on types, we put them not inside braces or parentheses like we do for implicit/explicit types, but inside square brackets. Check the type of List.contains (which we already saw) to recall that this requires the type to admit equality checking (you can compare any two elements of this type to see if they are equal, and return a Boolean value of true or false), given by [BEq α].

  Write a function match_head which takes as input a list of some type, and a target value of the same type, and returns true (of type Bool) if the head of the list matches the target value, and false (of type Bool) otherwise (including the cse where the list does not have a head element).
-/

def match_head {α : Type} [BEq α] : List α → α → Bool
  | [], _ => false
  | (h :: _), n => if (h == n) then true else false

/-
  In general, if we had a total_head function, we would like to write a "pseudocode" function as follows

  match_head(l, n) = (n == total_head l default)

  Let us try to write this for the Nat type in Lean. In order to use l and n as names for the arguments, one would need to write something called an "anonymous function", which we specify using the keyword "fun" and the ↦ operator (written as \mapsto). This is a standard notion in functional programming that comes from lambda calculus, and allows one to treat functions as first-class citizens, not unlike objects of any other type.
-/

def match_head_pc : List Nat → Nat → Bool :=
  fun l n ↦ (n == total_head 0 l)

-- Read up more on functions like map, filter, and zip, f you are unused to anonymous and/or higher-order functions (these show up even in Python, for example).

#check List.map
#check List.filter
#check List.zip

/-
  But what if I accidentally specify a *different* default value here while I'm calling the total_head function? Suppose I intended 0 to be the "default" value to return if the list is empty, but while calling the total_head function in my match_head function, I make a typo and write 9 instead. Now, while I might have enforced in my application that lists do not start with 0 and so such a confusion does not arise, I would get a true value even if the list begins with 9 instead.

  To avoid such accidents and unforeseen confusion, we would like to "bake in" the default value into the total_head function, instead of having to specify it every time one calls the function. However, if you had thought about the exercise given above for when total_head might fail, you would realize that this requires some refinement of the type considered.
-/

/-
  In order to ensure that total_head does not fail when the default value is "left implicit", as it were, one needs to ensure that the type is inhabited, i.e. there is at least one object which has said type. One can define a typeclass Inhabited, as follows.

  class Inhabited (α : Type) : Type where
    default : α

  If we wanted to ensure that the default value for lists was the empty list, we could define it as follows.
-/

instance List.Inhabited {α : Type} : Inhabited (List α) :=
  { default := [] }

/-
  To ensure that lists of any type can provide a default head value (even for the empty list), we ensure that as long as the type is inhabited, we just return the default value for said inhabited type.
-/

def total_head_no_def {α : Type} [Inhabited α] : List α → α
  | [] => Inhabited.default
  | (h :: _) => h

#check Inhabited.default
#eval total_head_no_def [1, 2, 3]
#eval total_head_no_def ([] : List Nat)
#eval total_head_no_def ([] : List Char)


/- To learn more about the behaviour of Inhabited (and other type classes), go through 5.6 in the Hitchhiker's Guide to Logical Verification. -/

/-
  State and prove a theorem named eqlist_eqcomp which says that if two non-empty lists are equal, then their heads and tails must be equal as well.

  It might be worthwhile to use a new tactic called "cases". If you have a hypothesis h which is of the form f x = f y, where f is a function known to be injective, "cases h" will simplify the goal by rewriting y as x. (This holds even if f has arity greater than one.)
-/

-- YOUR ANSWER GOES HERE
theorem eqlist_eqcomp {α : Type} {a b : α} {as bs : List α}
  (h : a :: as = b :: bs) : (a = b ∧ as = bs) := by
  cases h    -- turns b into a and bs into as
  constructor
  · rfl
  · rfl

-- Recall that we do not want to add arbitrary axioms because they could poison our whole context, and allow us to prove all sorts of junk. We know that a proposition is a statement that can be assigned a truth value. We will now show that from a patently false formula, one can derive any proposition whatsoever. The tactic used is called "contradiction".

theorem anything_from_false :
  ∀ α : Prop, 0 = 1 → α :=
  by
    intro α
    intro h
    contradiction

/-
  Essentially, we have 0 = 1 as an assumption, and Lean has 0 ≠ 1 built in as part of its in-built Nat/Int types, so from 0 = 1 ∧ 0 ≠ 1 (which is a pair of statements of the form p and ¬p) we can get any α whatsoever. If you hover over "contradiction" in the proof, you will see some examples of situations where it can be applied, in the face of contradictory assumptions.
-/

/-
Recall we stated a theorem called len_sup_long, which states that given a list (say l1), any other list (say l2) which contains all elements of l1 must have length at least as much as that of l1.

This statement is obviously false, since we can have l1 be [1, 1, 1, 1] and l2 can be [1, 2]. l2 contains all the elements of l1, but is clearly less long than l1. Adding a uniqueness clause would fix this.

Define a property named uniq that returns true if a given list of any type has only unique elements, false otherwise. (A property is a relation defined over a particular type; one way to define an n-ary property easily in Lean is to specify it as an n-ary function, the output of which is true when the input tuple belongs to the relation, and false otherwise.)

The #eval test cases given below might be useful to check if your definition is correct.
-/

-- YOUR ANSWER GOES HERE
def uniq {α : Type} [BEq α] : List α → Bool
  | []      => true
  | x :: xs => if xs.contains x then false else uniq xs

/-
#check uniq
#eval uniq [1, 2, 3]
#eval uniq [0, 1, 0]
#eval uniq ([]: List Char)
#eval uniq ["abc", "abc"]
-/

/-

Now state the fixed theorem called len_sup_long_correct (using the uniq definition you stated above). Make sure that it is applicable to any type whatsoever, which means that your theorem should start as follows.

∀ α : Type, [BEq α] → ...

For the purposes of the submission, discharge the proof obligation with a sorry, but try to prove it using the techniques we have seen so far, and see if you can finish the proof.
-/

-- YOUR ANSWER GOES HERE
theorem len_sup_long_correct {α : Type} [BEq α] :
  ∀ (l1 l2 : List α),
    uniq l1 = true →
    (∀ x, x ∈ l1 → x ∈ l2) →
    l2.length ≥ l1.length := by
  intro l1 l2 huniq hsub
  -- For submission: discharge the (expected) proof obligation with `sorry`
  sorry
