Loop 1 of `SPARK.Containers.Functional.Vectors.Higher_Order.Sum` comes from `src/spark-containers-functional-vectors-higher_order.adb`, lines 413-416.
The source revision is `0850ed4a8020f6c3a942f5f33ee58fed6d0b1173` in https://github.com/AdaCore/SPARKlib. The dataset records the upstream licence as `Apache-2.0 WITH LLVM-exception`.

The standalone project contains the subprogram and the declarations it uses. Some declarations or contracts have been adapted for proof in isolation.

The reference solution proves. In `base`, removing 1 loop invariant leaves 1 check unproved.

This sample is in `proof_loops_sparklib_open_book` because the SPARKlib installed with
GNATprove contains `spark-containers-functional-vectors-higher_order.adb`, including the proof code that `base` removes.
An agent can read it there, so scores on this dataset measure open-book proof.


## Changes to the upstream source

SPARKlib enables its `SPARK_Mode => Off --  #BODYMODE` units for its own proofs. The sample removes those markers and enables SPARK_Mode in `main.adc`.

The editable file `src/spark-containers-functional-vectors-higher_order-sum_impl.adb` contains `Sum_Impl`, which implements `Sum`. A subunit cannot carry `Refined_Post`, and GNATprove skips that aspect on a body stub. `Sum_Impl` has the precondition of `Sum` and, as its postcondition, the refined postcondition of `Sum`, which keeps it and only calls `Sum_Impl`.

## Notes moved out of the sources

These explanations were comments in the project sources. They describe how
the sample was built, so they live here instead of in the files the prompt
hands over.

### `src/test_sequence_ho.ads`

Instantiation of the generic the subject loop lives in.

Upstream writes this as a procedure body; here it is a package spec, because the sample has no main subprogram to run.
