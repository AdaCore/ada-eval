Loop 2 of `SPARK.Containers.Formal.Doubly_Linked_Lists.Impl.Rebuild_Free_List` comes from `src/spark-containers-formal-doubly_linked_lists-impl.adb`, lines 1109-1129.
The source revision is `0850ed4a8020f6c3a942f5f33ee58fed6d0b1173` in https://github.com/AdaCore/SPARKlib. The dataset records the upstream licence as `Apache-2.0 WITH LLVM-exception`.

The standalone project contains the subprogram and the declarations it uses. Some declarations or contracts have been adapted for proof in isolation.

The reference solution proves. In `base`, removing 4 loop invariants leaves 3 checks unproved.

## Notes moved out of the sources

These explanations were comments in the project sources. They describe how
the sample was built, so they live here instead of in the files the prompt
hands over.

### `src/lists.ads`

Local model of SPARK.Containers.Formal.Doubly_Linked_Lists.Impl holding exactly what the subject loop needs.  Every declaration below is copied verbatim from the upstream unit named against it; only the package nesting differs.  Upstream, Impl is a private generic child of the generic Doubly_Linked_Lists and the representation lives in that parent's private part, so neither can be named -- let alone instantiated -- from outside.
