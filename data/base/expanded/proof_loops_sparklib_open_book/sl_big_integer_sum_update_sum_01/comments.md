Loop 1 of `SPARK.Higher_Order.Fold.Sum.Big_Integer_Sum.Update_Sum` comes from `src/spark-higher_order-fold.adb`, lines 767-778.
The source revision is `0850ed4a8020f6c3a942f5f33ee58fed6d0b1173` in https://github.com/AdaCore/SPARKlib. The dataset records the upstream licence as `Apache-2.0 WITH LLVM-exception`.

The standalone project contains the subprogram and the declarations it uses. It declares some callees with `Import`. Some declarations or contracts have been adapted for proof in isolation.

The reference solution proves. In `base`, removing 1 loop invariant and 1 assertion leaves 1 check unproved.

This sample is in `proof_loops_sparklib_open_book` because the SPARKlib installed with
GNATprove contains `spark-higher_order-fold.adb`, including the proof code that `base` removes.
An agent can read it there, so scores on this dataset measure open-book proof.


## Changes to the upstream source

Both versions use gnatformat, with comment spacing, line breaks, identifier case
and `end` labels adjusted to satisfy GNAT’s build checks. The edits below also apply to both versions.

- SPARKlib enables its `SPARK_Mode => Off --  #BODYMODE` units for its own proofs. The sample removes those markers and enables SPARK_Mode in `main.adc`.
- In `spark-higher_order-fold.ads`, `pragma Warnings` suppresses 'formal object "Zero" is not referenced', 'function "Add" is not referenced', 'function "In_Range" is not referenced'.

## Notes moved out of the sources

These explanations were comments in the project sources. They describe how
the sample was built, so they live here instead of in the files the prompt
hands over.

### `src/spark-higher_order-fold-sum-big_integer_sum-update_sum.adb`

The loop body is the invariant, and the invariant is gone. `null;` is what keeps the stripped loop legal Ada.
