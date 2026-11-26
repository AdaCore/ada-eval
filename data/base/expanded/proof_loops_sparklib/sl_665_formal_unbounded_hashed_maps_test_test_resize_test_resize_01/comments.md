Loop 1 of `test_resize.Test_Resize` comes from `tests/665__formal_unbounded_hashed_maps_test/test_resize.adb`, lines 16-20.
The source revision is `0850ed4a8020f6c3a942f5f33ee58fed6d0b1173` in https://github.com/AdaCore/SPARKlib. The dataset records the upstream licence as `Apache-2.0 WITH LLVM-exception`.

The standalone project contains the subprogram and the declarations it uses.

The reference solution proves. In `base`, removing 2 loop invariants leaves 1 check unproved.

## Changes to the upstream source

Both versions use gnatformat, with comment spacing, line breaks, identifier case
and `end` labels adjusted to satisfy GNAT’s build checks. The unused `with Ada.Text_IO` and the ineffective `use Inst` were removed from `test_resize.adb`.

The editable loop lives in `src/test_resize-insert_elements.adb`. The library procedure `Test_Resize` keeps the trusted instance `Inst` and calls `Insert_Elements`. Its call satisfies the helper’s precondition, `Is_Empty (X)`, because `X` is empty after default initialization.
