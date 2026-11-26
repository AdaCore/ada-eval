Loop 1 of `SPARK.Containers.Formal.Vectors.Impl.Generic_Sorting.Merge` comes from `src/spark-containers-formal-vectors-impl.adb`, lines 552-580.
The source revision is `0850ed4a8020f6c3a942f5f33ee58fed6d0b1173` in https://github.com/AdaCore/SPARKlib. The dataset records the upstream licence as `Apache-2.0 WITH LLVM-exception`.

The standalone project contains the subprogram and the declarations it uses. It declares some callees with `Import`. Some declarations or contracts have been adapted for proof in isolation.

The reference solution proves. In `base`, removing 4 loop invariants leaves 7 checks unproved.

## Notes moved out of the sources

These explanations were comments in the project sources. They describe how
the sample was built, so they live here instead of in the files the prompt
hands over.

### `src/vectors.ads`

The used prefix fits in the capacity and is fully initialized. Upstream follows this declaration with a pragma Annotate (GNATprove, False_Positive, "type ""Vector"" is not fully initialized", ...).  No such message arises here -- the example never default-initialises a Vector -- and GNATprove reports the pragma as justifying nothing, so it is dropped.
