Loop 1 of `SPARK.Containers.Functional.Vectors.Higher_Order.Count` comes from `src/spark-containers-functional-vectors-higher_order.adb`, lines 72-78.
The source revision is `0850ed4a8020f6c3a942f5f33ee58fed6d0b1173` in https://github.com/AdaCore/SPARKlib. The dataset records the upstream licence as `Apache-2.0 WITH LLVM-exception`.

The standalone project contains the subprogram and the declarations it uses. Some declarations or contracts have been adapted for proof in isolation.

The reference solution proves. In `base`, removing 2 loop invariants leaves 1 check unproved.

This sample is in `proof_loops_sparklib_open_book` because the SPARKlib installed with
GNATprove contains `spark-containers-functional-vectors-higher_order.adb`, including the proof code that `base` removes.
An agent can read it there, so scores on this dataset measure open-book proof.


## Changes to the upstream source

SPARKlib enables its `SPARK_Mode => Off --  #BODYMODE` units for its own proofs. The sample removes those markers and enables SPARK_Mode in `main.adc`.

The editable file `src/spark-containers-functional-vectors-higher_order-count_impl.adb` contains `Count_Impl`, which implements `Count`. A subunit cannot carry `Refined_Post`, and GNATprove skips that aspect on a body stub. `Count_Impl` has the precondition of `Count` and, as its postcondition, the refined postcondition of `Count`, which keeps it and only calls `Count_Impl`.
