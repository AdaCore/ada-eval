Loop 1 of `SPARK.Containers.Formal.Doubly_Linked_Lists.Impl.Move` comes from `src/spark-containers-formal-doubly_linked_lists-impl.adb`, lines 1879-1891.
The source revision is `0850ed4a8020f6c3a942f5f33ee58fed6d0b1173` in https://github.com/AdaCore/SPARKlib. The dataset records the upstream licence as `Apache-2.0 WITH LLVM-exception`.

The standalone project contains the subprogram and the declarations it uses. It declares some callees with `Import`. Some declarations or contracts have been adapted for proof in isolation.

The reference solution proves. In `base`, removing 3 loop invariants and the ghost constant `Orig` leaves 6 checks unproved.

## Notes moved out of the sources

These explanations were comments in the project sources. They describe how
the sample was built, so they live here instead of in the files the prompt
hands over.

### `src/lists.ads`

Local model of SPARK.Containers.Formal.Doubly_Linked_Lists.Impl holding exactly what the subject loop needs.  Every declaration below is copied verbatim from the upstream unit named against it; only the package nesting differs.  Upstream, Impl is a private generic child of the generic Doubly_Linked_Lists and the representation lives in that parent's private part, so neither can be named -- let alone instantiated -- from outside.

The subject subprogram.  Upstream's declaration is

```
   procedure Move (Target : in out List; Source : in out List)
   with
     Global     => Address_State,
     Pre        =>
       (Static =>
          Structural_Invariant (Target)
          and then Structural_Invariant (Source)),
     Post       =>
       (Static =>
          Structural_Invariant (Target)
          and then Structural_Invariant (Source)),
     Exit_Cases =>
       (Length (Source) > Target.Capacity =>
          (Exception_Raised => Capacity_Error),
        others                            => Normal_Return);
```

Two changes, the same two that the sibling operation Assign needed.

Global: upstream's `Address_State` is the abstract state of SPARK.Containers.Formal.Impl.Address_Space, read by the volatile Same_Object aliasing short-circuit at the top of the body.  That short-circuit is dropped here (see the body), so the Global goes with it -- state the subprogram only ever read, which is the weakening direction.

Exit_Cases: upstream raises Capacity_Error when Source does not fit, and under FSF 16.1 flow then reports the components of the `in out` parameters as "might not be set ... following exceptional exit" (List's Nodes carry Relaxed_Initialization, so flow will not take the entry value for granted).  The Exit_Cases' own `Normal_Return` guard is therefore hoisted into the precondition and the raise removed.  The loop is handed exactly what upstream's guard handed it -- Source fits in Target -- so its proof burden is unchanged.
