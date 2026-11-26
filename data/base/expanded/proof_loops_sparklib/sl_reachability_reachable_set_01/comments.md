Loop 1 of `SPARK.Higher_Order.Reachability.Reachable_Set` comes from `src/spark-higher_order-reachability.adb`, lines 671-701.
The source revision is `0850ed4a8020f6c3a942f5f33ee58fed6d0b1173` in https://github.com/AdaCore/SPARKlib. The dataset records the upstream licence as `Apache-2.0 WITH LLVM-exception`.

The standalone project contains the subprogram and the declarations it uses. It declares some callees with `Import`. Some declarations or contracts have been adapted for proof in isolation.

The reference solution proves. In `base`, removing 5 loop invariants leaves 4 checks unproved.

## Changes to the upstream source

Both versions use gnatformat, with comment spacing, line breaks, identifier case
and `end` labels adjusted to satisfy GNAT’s build checks. The edits below also apply to both versions.

- SPARKlib enables its `SPARK_Mode => Off --  #BODYMODE` units for its own proofs. The sample removes those markers and enables SPARK_Mode in `main.adc`.
- In `reachability.ads`, `pragma Warnings` suppresses 'use clause for package "Memory_Index_Sequences" has no effect'.

The editable file `src/reachability-reachable_set_impl.adb` contains `Reachable_Set_Impl`, which implements `Reachable_Set`. A subunit cannot carry `Refined_Post`, and GNATprove skips that aspect on a body stub. `Reachable_Set_Impl` has the precondition of `Reachable_Set` and, as its postcondition, the refined postcondition of `Reachable_Set`, which keeps it and only calls `Reachable_Set_Impl`.

## Notes moved out of the sources

These explanations were comments in the project sources. They describe how
the sample was built, so they live here instead of in the files the prompt
hands over.

### `src/reach_inst.ads`

Run the proofs on as general an example as possible

### `src/reachability.adb`

All_Set is a sibling of Reachable_Set in the package body upstream; it is the subject of its own sample, so here it is imported with upstream's contract and no body.
