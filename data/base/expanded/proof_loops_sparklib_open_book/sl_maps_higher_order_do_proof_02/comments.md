Loop 2 of `SPARK.Containers.Functional.Maps.Higher_Order.Do_Proof` comes from `src/spark-containers-functional-maps-higher_order.adb`, lines 320-324.
The source revision is `0850ed4a8020f6c3a942f5f33ee58fed6d0b1173` in https://github.com/AdaCore/SPARKlib. The dataset records the upstream licence as `Apache-2.0 WITH LLVM-exception`.

The standalone project contains the subprogram and the declarations it uses. It declares some callees with `Import`. Some declarations or contracts have been adapted for proof in isolation.

The reference solution proves. In `base`, removing 1 loop invariant leaves 1 check unproved.

This sample is in `proof_loops_sparklib_open_book` because the SPARKlib installed with
GNATprove contains `spark-containers-functional-maps-higher_order.adb`, including the proof code that `base` removes.
An agent can read it there, so scores on this dataset measure open-book proof.


## Changes to the upstream source

Both versions use gnatformat, with comment spacing, line breaks, identifier case
and `end` labels adjusted to satisfy GNAT’s build checks. The edits below also apply to both versions.

- SPARKlib enables its `SPARK_Mode => Off --  #BODYMODE` units for its own proofs. The sample removes those markers and enables SPARK_Mode in `main.adc`.
- Converted the upstream test procedure `test_map_ho.adb` to the package specification `inst.ads`, which instantiates the generic for GNATprove.
- Removed the unused `with SPARK.Big_Intervals` from a Higher_Order spec.

## Notes moved out of the sources

These explanations were comments in the project sources. They describe how
the sample was built, so they live here instead of in the files the prompt
hands over.

### `src/inst.ads`

Instantiation of the unit under analysis.  GNATprove analyses instances, not generic units, so the extracted loop needs one.  The declarations are upstream's test procedure Test_Map_HO's, as a package spec here so that GNATprove analyses the instance.
