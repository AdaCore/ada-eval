Loop 1 of `SPARK.Containers.Functional.Infinite_Sequences.Higher_Order.Filter` comes from `src/spark-containers-functional-infinite_sequences-higher_order.adb`, lines 141-151.
The source revision is `0850ed4a8020f6c3a942f5f33ee58fed6d0b1173` in https://github.com/AdaCore/SPARKlib. The dataset records the upstream licence as `Apache-2.0 WITH LLVM-exception`.

The standalone project contains the subprogram and the declarations it uses. Some declarations or contracts have been adapted for proof in isolation.

The reference solution proves. In `base`, removing 2 loop invariants leaves 1 check unproved.

This sample is in `proof_loops_sparklib_open_book` because the SPARKlib installed with
GNATprove contains `spark-containers-functional-infinite_sequences-higher_order.adb`, including the proof code that `base` removes.
An agent can read it there, so scores on this dataset measure open-book proof.


## Changes to the upstream source

Both versions use gnatformat, with comment spacing, line breaks, identifier case
and `end` labels adjusted to satisfy GNAT’s build checks. The edits below also apply to both versions.

- SPARKlib enables its `SPARK_Mode => Off --  #BODYMODE` units for its own proofs. The sample removes those markers and enables SPARK_Mode in `main.adc`.
- Moved `with SPARK.Big_Intervals` into the Higher_Order body, its only user.

The editable file `src/spark-containers-functional-infinite_sequences-higher_order-filter_impl.adb` contains `Filter_Impl`, which implements `Filter`. A subunit cannot carry `Refined_Post`, and GNATprove skips that aspect on a body stub. `Filter_Impl` has the precondition of `Filter` and, as its postcondition, the refined postcondition of `Filter`, which keeps it and only calls `Filter_Impl`.
