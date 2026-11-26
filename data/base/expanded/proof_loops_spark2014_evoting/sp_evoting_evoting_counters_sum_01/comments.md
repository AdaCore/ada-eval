Loop 1 of `eVoting.Counters_Sum` comes from `testsuite/gnatprove/tests/evoting/evoting.adb`, lines 164-167.
The source revision is `deb177c58ac4e7b0e761f00456910468646d9a43` in https://github.com/AdaCore/spark2014. The dataset records the upstream licence as `bsl-1.0`.

The standalone project contains the subprogram and the declarations it uses.

The reference solution proves. In `base`, removing 1 loop invariant leaves 1 check unproved.

## Changes to the upstream source

Both versions use gnatformat, with comment spacing, line breaks, identifier case
and `end` labels adjusted to satisfy GNAT’s build checks.
