-- Lean is a powerful proof assistant as well as programming language. In flavour, it is likely different from most programming languages you have used so far, which have probably all been imperative; you tell the system what to do and how to do it. Lean, by contrast, is more in the style of a functional programming language, which means that you define functions and functionality (what something *is*), leaving the system to figure out how to achieve whatever is asked of it. This approach, while it allows more freedom by way of handling *potentially* infinite objects, of which we might only use an unbounded fraction, means that definitions for types and data now look significantly different from what they might have looked like in C or Java or Python.

-- The basic building block for our purposes is going to be an inductively-defined abstract datatype. One tells Lean that one is defining a datatype by appending ": Type" to the definition. In order to tell Lean that a type is inductive, the keyword "inductive" is prepended to the name of the type. Below, we define an inductive datatype called "day", which can take one of seven possible forms, as according to the standard notion of a week.

inductive day : Type where
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday
  | saturday
  | sunday

-- If you have installed Lean correctly and are using VSCode with the Lean 4 extension, as you use your arrow key to come down one line at a time, you will see the type for that particular line show up in the InfoView. The seven forms are constructors (even though they might not look like functions here). Each "|" symbol denotes one possible constructor for the datatype. Since the type is inductive, each constructor could have, in theory, taken on smaller objects of the same type as parameters. For our current purposes, each constructor is a 0-ary function. Note that every constructor need not be a function of the same arity; in particular, the base case for the inductive definition will usually be a 0-ary function, while the inductive case(s) will likely not.

-- Having defined this type, we can now define functions which operate over objects of this type. For example, one function one can write is "nextDay", which tells us what the day after a particular day is. For thi, much like we used the "inductive" keyword to specify to Lean that we were about to define an inductive datatype, we use the keyword "def" to indicate that we are about to define a function. While Lean is powerful enough to infer the type of a function by itself, it is good programming practice to define the type yourself (especially while getting used to writing functional code). The type, as we did for datatypes earlier, is specified by putting a colon after the name of the function, and then following the colon by the type of the function. The nextDay function for us will take as input a day, and output a day (corresponding to the day after the one taken as input). So the type of nextDay becomes day -> day. (For an arbitrary function which takes n parameters as input, the type would be specified as typeParam1 -> typeParam2 -> ... -> typeParamn -> typeOutput.) For each possible constructor (specified using a | symbol before the name of the constructor), we specify what the output might be using the => symbol between the constructor and the output. The strings 'monday', 'tuesday' etc. by themselves have no locus standi in our program, since they exist only in the scope of the datatype day (as constructors), and so one has to write day.monday, day.tuesday etc to avoid scope confusions for Lean. As shorthand to indicate to Lean that these are constructors inside an inductive definition, one can write .monday, and Lean will smartly figure out which inductive datatype they belong to, and appropriately associate the occurrence with the correct constructor.

def nextDay : day -> day
  | day.monday => .tuesday
  | .tuesday => .wednesday
  | .wednesday => .thursday
  | .thursday => .friday
  | .friday => .saturday
  | .saturday => .sunday
  | .sunday => .monday

-- Suppose we were unsure of the type of the function we wrote, perhaps because something later in the programme behaved erratically. How would one check the type of an object, function or otherwise? For this, one can use the command "#check". We can use this to check the type of the function nextDay that we have just defined.
#check nextDay

-- Similarly, to debug errors in actual values (while our current setup is fairly simple, one is often writing much more complex code, where it would be useful to check the values of particular functions applied to particular inputs), one uses the command "#eval".
#eval nextDay (nextDay .thursday)

-- Having set up various datatypes and functions, one usually wants to prove some properties of interest about them. To do this, one uses the keyword "theorem", gives the theorem a name, defines the property that is expected after a colon, and then after a ":= by" writes out a proof that this property actually holds. For example, we might try to prove something fairly trivial, which is that the value of the nextDay function applied to Monday is the same as that of the nextDay function applied to Monday. This is a trivial statement, of the shape x = x, so one way (the shortest, in fact) to prove this is to remind Lean that equality is an equivalence relation, and that it enjoys reflexivity, so x = x is always true when = is defined over the object x in question. This we do by a tactic called "rfl" (which stands for reflexivity).

theorem nd_monday :
  (nextDay .monday = nextDay .monday) :=
  by
    rfl

-- While we defined a type and functions over said type, sometimes there might be properties that objects of that type enjoy which we cannot capture as a function, or as part of the type definition itself. In such a case, if we are VERY sure that this behaviour is something we wish to capture and holds of every object of that type, we could specify this as an "axiom". However, this is a very dangerous strategy, because it can let us specify any old rubbish as ground truth, as we will see now.

axiom wrong_nd_monday : nextDay .monday = .sunday

-- Having defined such a "poisonous" axiom, we will now show that we will become able to prove arbitrary (and often, patently incorrect) statements, as follows.

theorem nd_monday_incorrect :
  nextDay (nextDay .monday) = .monday :=
  by
    rewrite [wrong_nd_monday]
    rfl

-- Here, since we started with nextDay applied to the input obtained by applying nextDay to .monday, we first need to evaluate nextDay .monday. However, since we have a helpful little axiom up there, we will use that to rewrite nextDay .monday to .sunday, which is what the equation defined as part of the wrong_nd_monday axiom tells us. The "rewrite" tactic (sometimes written as "rw"), when given a particular equation as actionable inside square brackets, rewrites the term matching the left side of the equation to whatever the right side of the equation is. Once we have done this rewriting, we get nextDay (.sunday), which is indeed .monday, and reflexivity does the job for us. But nextDay should be a monotonic operation, and it should not be the case that we can cycle back to the same day in two applications of nextDay. So this axiom poisoned our entire knowledge base, in some sense, which is why axioms are extremely dangerous to utilize willy-nilly. In the rest of our endeavours with Lean in this course, we will NOT use any axiom definitions.

------------------------------------------

-- Your job is to define a function, and prove a statement about it. Make sure that the function name and theorem name are exactly as specified, otherwise the autograder will not be able to grade your file.

-- Define a function nextWorkingDay, which takes as input an object of type day, and outputs an object of type day. nextWorkingDay returns the next working day of the week, where Saturday and Sunday are assumed to not be working days.

-- YOUR FUNCTION DEFINITION GOES HERE --

def nextWorkingDay : day -> day
  | day.monday => .tuesday
  | .tuesday => .wednesday
  | .wednesday => .thursday
  | .thursday => .friday
  | .friday => .monday
  | .saturday => .monday
  | .sunday => .monday

-- Now, define and prove a theorem called nwd_fri_sun which states that the next working day after Friday is the same as the next working day after Sunday.

-- YOUR THEOREM DEFINITION GOES HERE --

theorem nwd_fri_sun:
  (nextWorkingDay .friday= nextWorkingDay .sunday) :=
  by
    rfl

------------------------------------------

-- Similar to how we defined the day datatype, we can define a List datatype. However, unlike the day datatype, lists are much more ubiquitous and polymorphic (in the sense that I can have a list of numbers, a list of names, a list of days, a list of lists...). How might we define a list type? Ideally, we want the same definition, no matter what it is a list of (all lists do share a fundamental common structure, just that the individual objects in two different list might be of different types). So we define a list datatype inductively as follows, for any type α. (In fact, this definition is exactly the same as what Lean uses internally to define a datatype called List, which is why we need to use the name myList for this datatype.) A list can either be empty, or it can be constructed by prepending an element of a type \alpha (the head of the resulting list) to any pre-existing list of type \alpha (the tail of the resulting list).

inductive myList (α : Type) where
  | nil : myList α
  | cons : α -> myList α -> myList α

-- Creating a myList object with two elements of type \alpha would need us to write myList.cons (myList.cons (...)), which is painful and error-prone. So, Lean's in-built List type, instead of insisting that one do this every time to define lists, allows some shorthand by way of syntactic sugar. [] is the nil List, of any type. And instead of writing some sequence of cons every time, one can write a List object using square brackets and commas, and specifying the elements therein, as [el1, el2, ..., eln]. Sometimes (especially while specifying the cons case in an inductive proof) one wants to provide the structure of the list, as an element prepended to an existing list. Prepending is indicated by the infix double colon (::) operator.

-- We can see how the double colon operator works by using #eval.
#eval 1::(2::(3::[]))
#eval 1::[2, 3]

-- What happens if we try to run the following command? #eval 1:(2:3)

-- We can now ask for the types of various kinds of lists.
#check List.nil
#check (1::[2, 3])
#check [1, 2]
#check [1, 4.5]
#check [1.432, 2.5]

-- What happens if we try to run the following command? #check ["abc", 1, 2]

-- Having defined an inductive data type, it is only natural that we try to do proofs by induction on these, wherever applicable. Recall the inductive definition of length from class, and reproduce it here in Lean syntax. Call your function myLength. myLength takes as input a type and a list of this type, and returns a natural number of type Nat, which indicates the number of elements in this list of type α. Note that we have to provide the type as a parameter to the definition of the function *before* we can specify its type, since the type of the function (which is from List α to Nat) depends on α.

def myLength (α: Type): List α → Nat
  | List.nil => 0
  | List.cons _ tl => (myLength α tl) + 1

-- Note that (and you should get a warning about this) the 'hd' in that definition was unnecessary, since it is not even used in how the function is computed for the inductive case -- the inductive definition only depends on the actual value of tl. So we can, and should, remove hd from the left to make this definition clearer in the sense of not being dependent on the value of hd. Since the definition is agnostic to the actual value of hd, we can replace it by a placeholder standing for "any value of the right type", and in Lean, this placeholder is the undescore symbol (_). Since Lean contains an in-built definition of the List type, it also contains an in-built definition of length, which is (unimaginatively) called List.length.

-- One can now state and prove a theorem named eq_len_lists, which states that the length of the list [1, 2, 3] is equal to the length of the list [[], ['a'], ['B']].

-- YOUR ANSWER GOES HERE

theorem eq_len_lists:
  (myLength Nat [1, 2, 3] = myLength (List Char) [[], ['a'], ['B']])
  :=
  by
  rfl

-- Think about this: What would happen if instead of [[], ['a'], ['B']], I wanted to use [[], [], []]? Firstly, is this list of the same size as [1, 2, 3]? And if it is, try to state and prove the theorem in Lean, and see what you see as you step through in the InfoView.
theorem eq_len_lists2:
  (myLength Nat [1, 2, 3] = myLength (List Char) [[], [], []])
  :=
  by
  rfl
------------------------------------------
------------------------------------------
