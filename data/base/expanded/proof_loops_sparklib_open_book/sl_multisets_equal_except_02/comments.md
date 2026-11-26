Loop 2 of `SPARK.Containers.Functional.Multisets.Equal_Except` comes from `src/spark-containers-functional-multisets.adb`, lines 470-497.
The source revision is `0850ed4a8020f6c3a942f5f33ee58fed6d0b1173` in https://github.com/AdaCore/SPARKlib. The dataset records the upstream licence as `Apache-2.0 WITH LLVM-exception`.

The standalone project contains the subprogram and the declarations it uses. It declares some callees with `Import`. Some declarations or contracts have been adapted for proof in isolation.

The reference solution proves. In `base`, removing 2 loop invariants and 2 assertions leaves 1 check unproved.

This sample is in `proof_loops_sparklib_open_book` because the SPARKlib installed with
GNATprove contains `spark-containers-functional-multisets.adb`, including the proof code that `base` removes.
An agent can read it there, so scores on this dataset measure open-book proof.


## Changes to the upstream source

Both versions use gnatformat, with comment spacing, line breaks, identifier case
and `end` labels adjusted to satisfy GNAT’s build checks. The edits below also apply to both versions.

- SPARKlib enables its `SPARK_Mode => Off --  #BODYMODE` units for its own proofs. The sample removes those markers and enables SPARK_Mode in `main.adc`.
- In `spark-containers-functional-multisets.adb`, `pragma Warnings` suppresses 'procedure "Lemma_Leq_Cardinality" is not referenced', 'unused variable "E"'.
- In `spark-containers-functional-multisets.ads`, `pragma Warnings` suppresses 'unused variable "Element"'.

## Notes moved out of the sources

These explanations were comments in the project sources. They describe how
the sample was built, so they live here instead of in the files the prompt
hands over.

### `src/spark-containers-functional-multisets.ads`

The second of Equal_Except's two loops on its own.  Its precondition is what the first loop leaves behind when it does not return False: that loop's own invariant with the cursor exhausted, restated over Nb_Occurence because a visible declaration cannot name Multiset's private components.
