Loop 2 of `SPARK.Containers.Functional.Total_Maps.Equivalent_Maps` comes from `src/spark-containers-functional-total_maps.adb`, lines 245-264.
The source revision is `0850ed4a8020f6c3a942f5f33ee58fed6d0b1173` in https://github.com/AdaCore/SPARKlib. The dataset records the upstream licence as `Apache-2.0 WITH LLVM-exception`.

The standalone project contains the subprogram and the declarations it uses. It declares some callees with `Import`. Some declarations or contracts have been adapted for proof in isolation.

The reference solution proves. In `base`, removing 1 loop invariant leaves 1 check unproved.

## Changes to the upstream source

SPARKlib enables its `SPARK_Mode => Off --  #BODYMODE` units for its own proofs. The sample removes those markers and enables SPARK_Mode in `main.adc`.

## Notes moved out of the sources

These explanations were comments in the project sources. They describe how
the sample was built, so they live here instead of in the files the prompt
hands over.

### `src/spark-containers-functional-total_maps.ads`

The four primitives named by the Iterable aspect are imported rather than copied: the subject loop never calls them, and upstream's bodies make Iter_Has_Element statically False, which makes the two preconditions below inconsistent.  Importing removes the bodies, which is a weakening.
