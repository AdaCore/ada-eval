Loop 1 of `test_resize.Test_Resize_Big` comes from `tests/665__formal_unbounded_vectors_test/test_resize.adb`, lines 15-18.
The source revision is `0850ed4a8020f6c3a942f5f33ee58fed6d0b1173` in https://github.com/AdaCore/SPARKlib. The dataset records the upstream licence as `Apache-2.0 WITH LLVM-exception`.

The standalone project contains the subprogram and the declarations it uses. Some declarations or contracts have been adapted for proof in isolation.

The reference solution proves. In `base`, removing 1 loop invariant leaves 2 checks unproved.
