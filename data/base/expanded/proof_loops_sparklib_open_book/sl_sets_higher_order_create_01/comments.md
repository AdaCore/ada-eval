Loop 1 of `SPARK.Containers.Functional.Sets.Higher_Order.Create` comes from `src/spark-containers-functional-sets-higher_order.adb`, lines 172-196.
The source revision is `0850ed4a8020f6c3a942f5f33ee58fed6d0b1173` in https://github.com/AdaCore/SPARKlib. The dataset records the upstream licence as `Apache-2.0 WITH LLVM-exception`.

The standalone project contains the subprogram and the declarations it uses.

The reference solution proves. In `base`, removing 5 loop invariants leaves 1 check unproved.

This sample is in `proof_loops_sparklib_open_book` because the SPARKlib installed with
GNATprove contains `spark-containers-functional-sets-higher_order.adb`, including the proof code that `base` removes.
An agent can read it there, so scores on this dataset measure open-book proof.


## Changes to the upstream source

SPARKlib enables its `SPARK_Mode => Off --  #BODYMODE` units for its own proofs. The sample removes those markers and enables SPARK_Mode in `main.adc`.

The editable file `src/spark-containers-functional-sets-higher_order-create_impl.adb` contains `Create_Impl`, which implements `Create`. A subunit cannot carry `Refined_Post`, and GNATprove skips that aspect on a body stub. `Create_Impl` has, as its postcondition, the refined postcondition of `Create`, which keeps it and only calls `Create_Impl`.
