Loop 1 of `SPARK.Containers.Formal.Doubly_Linked_Lists.Impl.Splice` comes from `src/spark-containers-formal-doubly_linked_lists-impl.adb`, lines 2487-2504.
The source revision is `0850ed4a8020f6c3a942f5f33ee58fed6d0b1173` in https://github.com/AdaCore/SPARKlib. The dataset records the upstream licence as `Apache-2.0 WITH LLVM-exception`.

The standalone project contains the subprogram and the declarations it uses. It declares some callees with `Import`. Some declarations or contracts have been adapted for proof in isolation.

The reference solution proves. In `base`, removing 5 loop invariants and the ghost variable `Orig_Sum` leaves 6 checks unproved.

## Notes moved out of the sources

These explanations were comments in the project sources. They describe how
the sample was built, so they live here instead of in the files the prompt
hands over.

### `src/lists.ads`

Local model of SPARK.Containers.Formal.Doubly_Linked_Lists.Impl holding exactly what the subject loop needs.  Every declaration below is copied verbatim from the upstream unit named against it; only the package nesting differs.  Upstream, Impl is a private generic child of the generic Doubly_Linked_Lists and the representation lives in that parent's private part, so neither can be named -- let alone instantiated -- from outside.

The subject subprogram.  Upstream's declaration is

```
   procedure Splice
     (Target : in out List; Before : Cursor; Source : in out List)
   with
     Global     =>
       (Input => Address_State, Proof_In => Same_Object_Ghost),
     Pre        =>
       (Static =>
          Structural_Invariant (Target)
          and then Structural_Invariant (Source)),
     Post       =>
       (Static =>
          Structural_Invariant (Target)
          and then Structural_Invariant (Source)),
     Exit_Cases =>
       (not Is_Relevant (Target, Before)
        => (Exception_Raised => Program_Error),
        Is_Relevant (Target, Before)
        and then not Same_Object_Ghost
        and then Exceeds_Count_Type (Length (Target), Length (Source))
        => (Exception_Raised => Constraint_Error),
        Is_Relevant (Target, Before)
        and then not Same_Object_Ghost
        and then
          Exceeds_Capacity
            (Length (Target), Target.Capacity, Length (Source))
        => (Exception_Raised => Capacity_Error),
        others
        => Normal_Return);
```

Two changes, the same two that the sibling operation Assign needed.

Global: upstream's `Address_State` is the abstract state of SPARK.Containers.Formal.Impl.Address_Space, read by the volatile Same_Object aliasing short-circuit at the top of the body, and `Same_Object_Ghost` is the ghost proxy the Exit_Cases name because a Volatile_Function cannot appear in a contract.  That short-circuit is dropped here (see the body), so both go with it -- state the subprogram only ever read, which is the weakening direction.  The guard is unreachable in SPARK anyway: two `in out` parameters cannot denote the same object, which is why the unit's own header says these aliasing guards are "left out of the Exit_Cases".

Exit_Cases: upstream raises Program_Error, Constraint_Error or Capacity_Error from the three guards that head the body, and under FSF 16.1 a subprogram that propagates an exception must leave its outputs initialized on the exceptional exit; List's Nodes carry Relaxed_Initialization, so flow reports the `in out` parameters as possibly unset there, which a clean proof forbids and which upstream's own test.out does not record.  The Exit_Cases' own `others => Normal_Return` case is therefore hoisted into the precondition and the three guards removed.  The loop is handed exactly what upstream's guards handed it, so its proof burden is unchanged.
