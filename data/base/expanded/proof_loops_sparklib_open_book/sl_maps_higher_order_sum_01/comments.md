Loop 1 of `SPARK.Containers.Functional.Maps.Higher_Order.Sum` comes from `src/spark-containers-functional-maps-higher_order.adb`, lines 472-481.
The source revision is `0850ed4a8020f6c3a942f5f33ee58fed6d0b1173` in https://github.com/AdaCore/SPARKlib. The dataset records the upstream licence as `Apache-2.0 WITH LLVM-exception`.

The standalone project contains the subprogram and the declarations it uses.

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

The editable file `src/spark-containers-functional-maps-higher_order-sum_impl.adb` contains `Sum_Impl`, which implements `Sum`. A subunit cannot carry `Refined_Post`, and GNATprove skips that aspect on a body stub. `Sum_Impl` has the precondition of `Sum` and, as its postcondition, the refined postcondition of `Sum`, which keeps it and only calls `Sum_Impl`.

## Notes moved out of the sources

These explanations were comments in the project sources. They describe how
the sample was built, so they live here instead of in the files the prompt
hands over.

### `src/inst.ads`

Instantiation of the unit under analysis.  GNATprove analyses instances, not generic units, so the extracted loop needs one.  The declarations are upstream's test procedure Test_Map_HO's, as a package spec here so that GNATprove analyses the instance.
