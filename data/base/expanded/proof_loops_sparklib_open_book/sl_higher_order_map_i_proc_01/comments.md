Loop 1 of `SPARK.Higher_Order.Map_I_Proc` comes from `src/spark-higher_order.adb`, lines 60-65.
The source revision is `0850ed4a8020f6c3a942f5f33ee58fed6d0b1173` in https://github.com/AdaCore/SPARKlib. The dataset records the upstream licence as `Apache-2.0 WITH LLVM-exception`.

The standalone project contains the subprogram and the declarations it uses. It declares some callees with `Import`. Some declarations or contracts have been adapted for proof in isolation.

The reference solution proves. In `base`, removing 1 loop invariant leaves 1 check unproved.

This sample is in `proof_loops_sparklib_open_book` because the SPARKlib installed with
GNATprove contains `spark-higher_order.adb`, including the proof code that `base` removes.
An agent can read it there, so scores on this dataset measure open-book proof.
