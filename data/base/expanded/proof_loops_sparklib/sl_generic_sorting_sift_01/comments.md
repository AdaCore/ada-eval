Loop 1 of `Generic_Sorting.Sift` comes from `src/spark-containers-formal-vectors-impl.adb`, lines 632-658.
The source revision is `0850ed4a8020f6c3a942f5f33ee58fed6d0b1173` in https://github.com/AdaCore/SPARKlib. The dataset records the upstream licence as `Apache-2.0 WITH LLVM-exception`.

The standalone project contains the subprogram and the declarations it uses. It declares some callees with `Import`. Some declarations or contracts have been adapted for proof in isolation.

The reference solution proves. In `base`, removing 2 loop invariants and the ghost function `Is_Anc` leaves 4 checks unproved.

## Notes moved out of the sources

These explanations were comments in the project sources. They describe how
the sample was built, so they live here instead of in the files the prompt
hands over.

### `src/sorting.ads`

Upstream, Sift is nested inside Generic_Sorting.Sort and reaches Container, Max and Temp as up-level references to Sort's parameter and locals.  Here they are parameters of Sift, which leaves the loop text unchanged.  T is Sort's own declaration, copied verbatim.  Sort's ghost function Is_Anc, which `solution` needs for the sift path invariant, is removed with it: no surviving contract refers to it.

### `src/vectors.ads`

Upstream, Vectors is a generic with formals Index_Type, Element_Type and "="; Generic_Sorting adds the formal "<".  GNATprove does not analyse a generic, only its instances, and instantiating the whole Vectors -> Vectors.Impl -> Generic_Sorting chain would mean copying most of SPARKlib.  So the formals are fixed here: Index_Type to Positive (the commonest actual), Element_Type to a private record, and "<" to an opaque imported function -- opaque because the subject loops never need to know anything about the order, and an opaque function assumes strictly less than a concrete one would.
