Loop 1 of `SPARK.Higher_Order.Reachability.Model` comes from `src/spark-higher_order-reachability.adb`, lines 579-621.
The source revision is `0850ed4a8020f6c3a942f5f33ee58fed6d0b1173` in https://github.com/AdaCore/SPARKlib. The dataset records the upstream licence as `Apache-2.0 WITH LLVM-exception`.

The standalone project contains the subprogram and the declarations it uses. It declares some callees with `Import`. Some declarations or contracts have been adapted for proof in isolation.

The reference solution proves. In `base`, removing 6 loop invariants and the `with` and `use` clauses for `SPARK.Big_Intervals` leaves 4 checks unproved.

## Changes to the upstream source

SPARKlib enables its `SPARK_Mode => Off --  #BODYMODE` units for its own proofs. The sample removes those markers and enables SPARK_Mode in `main.adc`.

## Notes moved out of the sources

These explanations were comments in the project sources. They describe how
the sample was built, so they live here instead of in the files the prompt
hands over.

### `src/reach_inst.ads`

Run the proofs on as general an example as possible

### `src/reachability.ads`

Local functions of the package body upstream, promoted to the visible part here because the extracted subprogram states its postcondition in terms of Model_Internal.  All_Set is the subject of its own sample, so it is imported with upstream's contract and no body; the rest are expression functions, whose bodies are their meaning, and are copied whole.

Is_Acyclic is a sibling of Model in the package body upstream, and the subject of its own sample.  Only its visible contract is imported here: Model's own body sees Is_Acyclic's Refined_Post upstream, and this example does without it.
