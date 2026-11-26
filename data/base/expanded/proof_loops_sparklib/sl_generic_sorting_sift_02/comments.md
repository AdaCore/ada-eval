Loop 2 of `Generic_Sorting.Sift` comes from `src/spark-containers-formal-vectors-impl.adb`, lines 660-676.
The source revision is `0850ed4a8020f6c3a942f5f33ee58fed6d0b1173` in https://github.com/AdaCore/SPARKlib. The dataset records the upstream licence as `Apache-2.0 WITH LLVM-exception`.

The standalone project contains the subprogram and the declarations it uses. It declares some callees with `Import`. Some declarations or contracts have been adapted for proof in isolation.

The reference solution proves. In `base`, removing 3 loop invariants leaves 8 checks unproved.

## Notes moved out of the sources

These explanations were comments in the project sources. They describe how
the sample was built, so they live here instead of in the files the prompt
hands over.

### `src/sorting.ads`

C0 is the value C holds when upstream's sift-down loop -- the sibling loop deleted here -- falls out.  What that loop leaves is exactly its own invariant, C in S .. Max and Is_Anc (S, C), and that is all the Pre below assumes.

Is_Anc is the sift path invariant: it lets the sift-up loop conclude Father = C / 2 >= Root >= 1.

### `src/vectors.ads`

Upstream, Vectors is a generic with formals Index_Type, Element_Type and "="; Generic_Sorting adds the formal "<".  GNATprove does not analyse a generic, only its instances, and instantiating the whole Vectors -> Vectors.Impl -> Generic_Sorting chain would mean copying most of SPARKlib.  So the formals are fixed here: Index_Type to Positive (the commonest actual), Element_Type to a private record, and "<" to an opaque imported function -- opaque because the subject loops never need to know anything about the order, and an opaque function assumes strictly less than a concrete one would.
