Loop 1 of `Test_Higher_Order3.Prove_No_Overflows` comes from `tests/R509-011__higher_order_proof/test_higher_order3.adb`, lines 9-27.
The source revision is `0850ed4a8020f6c3a942f5f33ee58fed6d0b1173` in https://github.com/AdaCore/SPARKlib. The dataset records the upstream licence as `Apache-2.0 WITH LLVM-exception`.

The standalone project contains the subprogram and the declarations it uses. It declares some callees with `Import`.

The reference solution proves. In `base`, removing 4 loop invariants leaves 2 checks unproved.

## Changes to the upstream source

Both versions use gnatformat, with comment spacing, line breaks, identifier case
and `end` labels adjusted to satisfy GNAT’s build checks.
