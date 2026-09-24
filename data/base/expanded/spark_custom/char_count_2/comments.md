The proof harness defines its own `Ref_Count` and wraps `Count` in a function
whose postcondition requires it to return `Ref_Count`. `Count`'s postcondition is stated in terms of the solution's `Count_Ghost`,
and GNATprove cannot relate two recursive functions without an inductive lemma,
so the harness proves one that names `String_Utils.Count_Ghost`. A solution that
redefines what `Count_Ghost` computes, or weakens `Count`'s postcondition, fails
the harness as `unproved`. A solution that renames, hides, or restructures
`Count_Ghost` fails it as `error` or `unproved` even if otherwise correct; that
is accepted, since the prompt gives `Count_Ghost` and asks only for the proof.
