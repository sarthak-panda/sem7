-- One can do long, multiline comments in Lean by enclosing them within a forward slash followed by a hyphen and a hyphen followed by a forward slash, as shown below.

/-
Much like we used the double colon operator for prepending an element to a list, we can use two pluses (++) as an infix operator for list appending. List.append l1 l2 can be written more concisely as l1 ++ l2. You can see how this works via the following test. You have to be careful that for (::) the first operand must be an *element* and the second a *List*, but for (++) both operands must be Lists, even if one happens to be a singleton list. (Types are not to be taken lightly in programming... we are no longer operating in a loosey-goosey language like Python, where anything goes, pretty much, and all sorts of weird side effects might happen.)
-/

#eval ([1, 2, 3] ++ [4])

/-
Recall our earlier theorem where we showed that the lengths of the lists [1, 2, 3] and [[], ['a'], ['B']] were equal. It is obvious that the list which is the result of the expression 0::[1, 2, 3] has length longer than [1, 2, 3] (because it has one extra element). One way to state it as the relatively "normal" theorem, as follows.
-/

theorem len_prep_longer_1234 :
    List.length [1, 2, 3] < List.length (0::[1, 2, 3]) :=
    by
/-
    However, here rfl will no longer work! rfl is only for reflexivity, i.e. stating that a particular term is equal to itself, but we are dealing with *in*equalities here. So while the statement is not a difficult one (of course 3 is less than 4), we still need something that will be allow Lean to realize that the natural numbers (which form the range of the List.length function) obey a total order by way of the < relation. One such tactic is called 'simp', short for 'simplify'. simp will do some rewriting and replace terms by "simpler" variants, in order to simplify the current state to reach the goal.
-/
    simp

-- In fact, simp suffices for our use here, and lets us reach the goal without requiring any further tactics. One can also use simp in place of rewrite, by giving it a specific fact it can use to simplify further.

-- Now, there was nothing special about these choices of numbers for us. It is indeed true that if one prepends an existing list with any element, the length increases by one. This is true for any list and any element *of any given type*. In order to make such a statement ("for any type, for any list of said type and any element of said type..."), one needs to use quantification, namely the \forall operator (which renders as ∀ if you have the right extension). One can state a theorem called len_prep_longer to this effect as follows.

theorem len_prep_plus_one :
    ∀ α : Type, ∀ l : List α, ∀ n : α, List.length l = List.length (n :: l) - 1 :=
    by simp
/-
    Interestingly, simp suffices to prove even this more complex-looking statement, but rfl would not, even though the equality is indeed true. rfl cannot "get past" the quantifiers, but simp does (a fair bit) more than what rfl does. simp would continue to work even if we just said List.length l < List.length (n :: l) inside the quantification, which rfl cannot handle, as we just saw. Even more importantly, simp can figure out that the right hand side of the equation will never evaluate to a non-natural number (-1, for example) thanks to the shape of the list passed as input to List.length on the RHS (the length of a list constructed using cons -- or ::, the syntactic sugar for cons -- can never be zero). So simp is powerful enough to sweep a lot under the rug, including how to handle quantification, should we ever find ourselves in a situation where simp would *not* suffice. In general, one cannot prove relatively complex universally quantified statements without doing something about the quantified variables.

    How would you prove a ∀ statement if you had to do it by hand? How would you start?

    One usually starts the proof of such a statement by saying "Consider an arbitrary < object of the appropriate type >" (a type, a list, a natural number, a graph, whatever) and then proceeds to manipulate this object to obtain the desired conclusion. Finally, one concludes by claiming that since the statement held true about an arbitrarily chosen object, the theorem must hold about all such objects. One does a very similar thing in Lean, but instead of saying a long phrase like the above, one just uses a tactic called "intro".
-/

theorem len_prep_plus_one_try1 :
    ∀ α : Type, ∀ l : List α, ∀ n : α, List.length l = List.length (n :: l) - 1 :=
    by
        intro type_alpha
        intro list_l
        intro number_n
        rfl

-- In the above proof, we considered an arbitrary type called type_alpha, an arbitrary list called list_l (and Lean knows, thanks to our type declarations in the theorem statement, that list_l must be a List of type type_alpha), and an arbitrary number called number_n. We could also have used the same names (α, l, n) but this is not strictly necessary. What yields the correspondence is the order in which the variables are introduced; each introduction corresponds to the universal quantification at the outermost level in the current expected goal. What would happen if we added another intro, say intro x, between intro number_n and rfl?

-- As you step through, you will see that each intro statement moves the arbitrary object considered into the *assumptions* available to Lean (the first statement adds an assumption saying that type_alpha is of type Type etc.), and strips away one level of quantification (since one can clearly use ). Once you are done with the three intro statements, you are left with just the statement inside the quantification, which can be handled by rfl directly, since List.length for n::l is defined to be 1 + List.length l (for any n and l of the appropriate types) and rfl does do some basic simplification, as we saw in the last set of programs.

-- One could complicate this definition slightly more, and write the same theorem in the following way, using implication (either \imp or a hyphen followed by the greater than sign) and conjunction (\wedge, or forward slash followed by backward slash).

theorem len_prep_plus_one_try2 :
  ∀ α : Type, ∀ l1 l2 : List α, ∀ n : α, ∀ len1 len2 : Nat,
        (List.length l1 = len1) ∧ (List.length l2 = len2) ∧ (l2 = n :: l1) → len1 = len2 - 1 :=
  by
    -- One can introduce all variables in one go if one is sure about doing the right thing and getting the variable order-name correspondence right.
    intro α l1 l2 n len1 len2
    -- At this point, our expected goal is still of the form φ /\ ψ /\ χ → ξ. How does one prove an implication?
    -- To check if a formula φ implies another ψ, the following suffices: if I assume φ and am able to prove ψ, then φ implies ψ. So we do the same thing we did for universal quantification, and do an introduction. Unlike ∀ though, here we do not introduce an *arbitrary* object; the exact formula to the left of the → operator needs to be introduced as an assumption. When we introduce it, we give it a name so we can manipulate it later.
    intro h  -- we use h, for 'hypothesis'
    -- h is now a three-way conjunction. We want to be able to use each of these conjuncts separately, and then try to achieve our desired goal.
    -- In Lean, conjunction /\ is a binary operator where parentheses are implicitly applied right-first. So our hypothesis h is essentially h1 /\ (h2 /\ h3). If we have a conjunctive hypothesis, we can use either/both of those conjuncts freely in any proof (unlike a disjunctive hypothesis, where we must show that the goal holds under either disjunct, so we essentially do two separate proofs). In order to extract the two conjuncts from a formula with /\ at the outermost level, we use the functions And.left (to get the left conjunct) and And.right (to get the right conjunct).
    have h1 : l1.length = len1 :=
      by
        exact And.left h
    -- Since we want each conjunct to have first-class standing as an assumption (this might not always be necessary; but in this proof it is handy to do so), we use the keyword "have" and give it a name 'h1', and state the statement we are interested in. The proof for h1 is merely the first part of the conjunction h, so we use the keyword 'exact' to indicate that the goal exactly matches what follows. What follows is 'And.left h', which gives us the left conjunction of h, and the proof is done. (If the goal matches some assumption exactly without doing any other manipulation -- like we are doing with And.left here -- we can even just use the tactic 'assumption'.)

    have h2 : l2.length = len2 :=
      by
        -- One can leave a proof in Lean unfinished by apologizing to the compiler with the keyword 'sorry'. This tells Lean to pretend as if the proof is finished and move on with merely a warning, instead of a full-blown error.
        -- ERASE THE FOLLOWING SORRY AND FILL IN THE PROOF
        exact And.left (And.right h)
    have h3 : l2 = n :: l1 :=
      by
        -- ERASE THE FOLLOWING SORRY AND FILL IN THE PROOF
        exact And.right (And.right h)


    -- At this point, we need to show that len1 = len2 - 1. In order to do this, we first have to tell Lean that len1 and len2 have specific meaning (which we can get from h1 and h2). Since h1 and h2 are equalities, we would ideally like to use the rewrite tactic, to tell Lean that these names belong to the lengths of lists (and said lists are connected by h3). However, recall from earlier that the rewrite (or rw) tactic takes each occurrence of the *LHS* of the equality in the expected goal and rewrites it to the *RHS*. The goal does not contain any occurrences of l1.length or l2.length; so how do we proceed? We have to tell Lean to rewrite in the opposite direction, which we do by specifying a left arrow inside the argument to the rw tactic, upon which it rewrites according to the equality specified in the argument, but from right to left instead (i.e. it rewrites every occurrence of the RHS of the equality to the LHS).

    rw [<- h1]
    rw [<- h2]

    -- Once we have done this, we end up with l1.length = l2.length - 1 as the expected goal. rfl is not going to give us the desired result, because l1, l2 etc are not built-in truths. The goal is only obvious upon using the fact that l2 is obtained by prepending l1 with some element, and we have called this fact h3. So while rfl does not work, simp will, as long as it is allowed to invoke h3 and then simplify.

    simp [h3]

-- State and prove a new theorem called len_prep_longer, which states that any prepending operation causes the length to strictly increase (not necessarily by one). Your statement must hold for any type, any list and any number, and must be written in the implies form, like len_prep_plus_one_try2.

-- YOUR ANSWER GOES HERE

theorem len_prep_longer_try0:--to ask the name
  ∀ α : Type, ∀ l1 l2 : List α, ∀ n : α, ∀ len1 len2 : Nat,
    (List.length l1 = len1) ∧ (List.length l2 = len2) ∧ (l2 = n::l1) → len1<len2 :=
by
  intro α l1 l2 n len1 len2
  intro h
  have h1 : l1.length = len1 := by
    exact And.left h
  have h2 : l2.length = len2 := by
    exact And.left (And.right h)
  have h3 : l2 = n :: l1 := by
    exact And.right (And.right h)
  -- rewrite the named lengths back to the list-length expressions
  rw [← h1]
  rw [← h2]
  simp[h3]
-- So we know how to break down a conjunctive hypothesis. What if the conjunction appears on the right hand side, as part of the goal?

theorem len_prep_longer :
  ∀ α, ∀ l1 l2 : List α, ∀ n : α,
    l2 = (n :: l1) -> List.length l2 > 0 ∧ List.length l2 > List.length l1 :=
    by
    intro α
    intro l1 l2
    intro n
    intro h
    -- At this point the goal is of the form φ ∧ ψ. How does one prove a conjunction? One has to prove each of the conjuncts separately. In order to tell Lean that we want to prove each conjunct separately, we use the 'apply' tactic. The tactic 'apply' takes a parameter and tries to match the current goal against the pattern given by the parameter. And.intro is a constructor for ∧, which, given φ and ψ, constructs φ ∧ ψ. This is one situation where we do a "backward proof"; in order to prove φ ∧ ψ using And.intro, we need to prove φ and prove ψ. apply And.intro creates two subgoals, one for each conjunct of the conjunction that is the goal before applying And.intro. The two cases are called "left" (for the left conjunct) and "right" (for the right conjunct). Cases in Lean are separated by enclosing them inside braces.
    apply And.intro
    {
      simp [h]
    }
    {
      simp [h]
    }

-- Lists are one of the most useful structures in Lean. It is good to know what operations are defined on lists, which you can do by typing List. and finding the list of options the autocomplete dropdown gives you (if you have the right VSCode extension installed). One very helpful function is List.contains. One can also use \in (the set membership operator ∈) as shorthand for List.contains.

#check List.contains
#eval List.contains [1, 2, 3] 1
#eval List.contains [] 4
#eval List.contains [[], ['s']] []
#eval 1 ∈ [1, 2, 3]

-- Use List.contains to state a theorem called len_sup_long, which states that given a list (say l1), any other list (say l2) which contains all elements of l1 must have length at least as much as that of l1. Be careful about quantification, and use parentheses to be precise, if in doubt. The lists l1 and l2 must be of type Nat. Leave the proof as a 'sorry' for now. Convince yourself of the truth or falsehood of the statement by hand, be as precise as possible, and leave it as a comment in your final submission.

-- YOUR THEOREM STATEMENT GOES HERE
theorem len_sup_long (l1 l2 : List Nat)
  (hnodup : l1.Nodup) -- l1 should not have duplicate elements
  (h : ∀ x, x ∈ l1 → x ∈ l2) : -- every element of l1 is a member of l2
  l2.length ≥ l1.length := by
  sorry

-- YOUR EXPLANATION GOES HERE
/-The naive statement "if every element of l1 is also in l2 then length l2 ≥ length l1"
is not true for lists in general, because `List.contains` / `∈` only talks about membership,
not multiplicity. Concretely, duplicates in l1 can make l1 longer while all its elements
still appear (possibly fewer times) in l2.
The fix for this requires `l1` to have no duplicates (`List.Nodup l1`). Then it becomes true
-/
#check List.length
#check List.append

theorem append_len:
  ∀ α: Type, ∀ l1 l2 : List α, List.length (List.append l1 l2) = (List.length l1) + (List.length l2) :=
  by
/-
Pro tip: For any statement in Lean, try to prove the statement by hand for yourself first. You will find that this often makes life easier with proof assistants; they are merely assistants, but often rather silly and extremely-single-minded ones, so your commands have to be accurate, precise, and EXTREMELY detailed -- much more so than your handwritten proof might be. And in general, it is good to have an idea of which tactic works before you start trying various tactics at random.

If you do write down a proof by hand, you will see that the way to prove this statement is to induct on the list l1 (given the way List objects are defined, as we saw last time). So we recreate the proof here, using the tactic 'induction'. In order to induct on an object, the object needs to be defined inductively. Each case of the induction will correspond to one constructor of the inductive definition, and we will specify this using 'with' (like we used 'where' in the inductive definition) and the | operator to separate the cases for the various constructors (for the inductive cases, we will first define the constructor with the appropriate parameters, and also a name for the inductive hypothesis/hypotheses, so that we can refer to them in the proof). Much like how we did for function definitions over inductive objects, each constructor name will be followed by '=>', after which we will write the proof for that case.
-/
  intro α
  intro l1 l2
  induction l1 with
  | nil => simp
  | cons n la Ih =>
      simp
      -- Simple rfl will not work here, since we want to convert a + b + 1 to a + 1 + b, and the two sides of that equality are not the same. However, since we, as humans know that + is associative and commutative, we know that these two terms do indeed evaluate to the same. Lean also knows that + is associate and commutative, and it can apply reflexivity on top of associativity and commutativity for operators that enjoy these properties, using the tactic ac_rfl, instead of plain old rfl.
      ac_rfl

-- State and prove a theorem named append_len_greater which states that if a list is obtained by appending two lists together, its length must be at least that of each individual list.
-- YOUR ANSWER GOES HERE1
theorem append_len_greater:
  ∀ α : Type, ∀ l1 l2 : List α,
  List.length (List.append l1 l2) ≥ List.length l1 ∧ List.length (List.append l1 l2) ≥ List.length l2 :=
  by
  intro α l1 l2
  rw [append_len α l1 l2]
  simp
