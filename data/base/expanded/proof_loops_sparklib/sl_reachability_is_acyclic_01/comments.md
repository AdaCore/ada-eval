Loop 1 of `SPARK.Higher_Order.Reachability.Is_Acyclic` comes from `src/spark-higher_order-reachability.adb`, lines 176-195.
The source revision is `0850ed4a8020f6c3a942f5f33ee58fed6d0b1173` in https://github.com/AdaCore/SPARKlib. The dataset records the upstream licence as `Apache-2.0 WITH LLVM-exception`.

The standalone project contains the subprogram and the declarations it uses. It declares some callees with `Import`. Some declarations or contracts have been adapted for proof in isolation.

The reference solution proves. In `base`, removing 3 loop invariants leaves 4 checks unproved.

## Changes to the upstream source

Both versions use gnatformat, with comment spacing, line breaks, identifier case
and `end` labels adjusted to satisfy GNAT’s build checks. The edits below also apply to both versions.

- SPARKlib enables its `SPARK_Mode => Off --  #BODYMODE` units for its own proofs. The sample removes those markers and enables SPARK_Mode in `main.adc`.
- In `reachability.adb`, `pragma Warnings` suppresses 'function "Model_Internal" is not referenced'.

The editable file `src/reachability-is_acyclic_impl.adb` contains `Is_Acyclic_Impl`, which implements `Is_Acyclic`. A subunit cannot carry `Refined_Post`, and GNATprove skips that aspect on a body stub. `Is_Acyclic_Impl` has the precondition of `Is_Acyclic` and, as its postcondition, the refined postcondition of `Is_Acyclic`, which keeps it and only calls `Is_Acyclic_Impl`.

## Notes moved out of the sources

These explanations were comments in the project sources. They describe how
the sample was built, so they live here instead of in the files the prompt
hands over.

### `src/reach_inst.ads`

Run the proofs on as general an example as possible

### `src/reachability.adb`

All_Set is a sibling of Is_Acyclic in the package body upstream; it is the subject of its own sample, so here it is imported with upstream's contract and no body.
