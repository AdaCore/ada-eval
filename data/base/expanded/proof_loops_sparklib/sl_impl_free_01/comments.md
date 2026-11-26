Loop 1 of `SPARK.Containers.Formal.Doubly_Linked_Lists.Impl.Free` comes from `src/spark-containers-formal-doubly_linked_lists-impl.adb`, lines 1216-1231.
The source revision is `0850ed4a8020f6c3a942f5f33ee58fed6d0b1173` in https://github.com/AdaCore/SPARKlib. The dataset records the upstream licence as `Apache-2.0 WITH LLVM-exception`.

The standalone project contains the subprogram and the declarations it uses. It declares some callees with `Import`. Some declarations or contracts have been adapted for proof in isolation.

The reference solution proves. In `base`, removing 2 loop invariants and 9 assertions leaves 3 checks unproved.

## Notes moved out of the sources

These explanations were comments in the project sources. They describe how
the sample was built, so they live here instead of in the files the prompt
hands over.

### `src/lists.ads`

Local model of SPARK.Containers.Formal.Doubly_Linked_Lists.Impl holding exactly what the subject loop needs.  Every declaration below is copied verbatim from the upstream unit named against it; only the package nesting differs.  Upstream, Impl is a private generic child of the generic Doubly_Linked_Lists and the representation lives in that parent's private part, so neither can be named -- let alone instantiated -- from outside.

Callee, copied as a spec with its upstream contract and marked `with Import`: SPARK is modular, so a callee's body is never needed to prove its caller.  Upstream this is a ghost procedure nested in Free's own declarative part; keeping its body here would put a second outermost loop in the example (that loop is the sample sl_impl_rebuild_free_list_02, which proves it).  Only its indentation and the `Import` moved.

The subject subprogram, promoted from Impl's private part (its "Local Subprograms" section) with its contract unchanged.
