Loop 1 of `SPARK.Containers.Formal.Doubly_Linked_Lists.Impl.Insert` comes from `src/spark-containers-formal-doubly_linked_lists-impl.adb`, lines 1483-1500.
The source revision is `0850ed4a8020f6c3a942f5f33ee58fed6d0b1173` in https://github.com/AdaCore/SPARKlib. The dataset records the upstream licence as `Apache-2.0 WITH LLVM-exception`.

The standalone project contains the subprogram and the declarations it uses. It declares some callees with `Import`. Some declarations or contracts have been adapted for proof in isolation.

The reference solution proves. In `base`, removing 4 loop invariants, 1 assertion and the ghost constant `L0` leaves 3 checks unproved.

## Notes moved out of the sources

These explanations were comments in the project sources. They describe how
the sample was built, so they live here instead of in the files the prompt
hands over.

### `src/lists.ads`

Local model of SPARK.Containers.Formal.Doubly_Linked_Lists.Impl holding exactly what the subject loop needs.  Every declaration below is copied verbatim from the upstream unit named against it; only the package nesting differs.  Upstream, Impl is a private generic child of the generic Doubly_Linked_Lists and the representation lives in that parent's private part, so neither can be named -- let alone instantiated -- from outside.

The subject subprogram.  Upstream's declaration is

```
   procedure Insert
     (Container : in out List;
      Before    : Cursor;
      New_Item  : Element_Type;
      Position  : out Cursor;
      Count     : Count_Type)
   with
     Global     => null,
     Pre        => (Static => Structural_Invariant (Container)),
     Post       =>
       (Static =>
          Structural_Invariant (Container)
          and then Length (Container) = Length (Container'Old) + Count
          and then Is_Relevant (Container, Before)),
     Exit_Cases =>
       (not Is_Relevant (Container, Before)                          =>
          (Exception_Raised => Program_Error),
        Is_Relevant (Container, Before)
        and then Exceeds_Count_Type (Length (Container), Count)      =>
          (Exception_Raised => Constraint_Error),
        Is_Relevant (Container, Before)
        and then
          Exceeds_Capacity
            (Length (Container), Container.Capacity, Count)          =>
          (Exception_Raised => Capacity_Error),
        others                                                       =>
          Normal_Return);
```

and its body raises those three exceptions from the guards that head it. Under FSF 16.1 a procedure that propagates an exception must initialize its `out` parameters on the exceptional exit too, so the raises draw `low: "Position" might not be set in "Insert"`, which a clean proof forbids.  It is not a fact about this loop: it reproduces on an eight-line program with no SPARKlib in it, and upstream's own test.out for this unit records the opposite -- `impl.ads:473:07: info: initialization of "Position" proved`.  So the Exit_Cases' own `others => Normal_Return` case is hoisted into the precondition and the three raising guards are removed.  That exchange is exactly neutral for the loop: after upstream's guards the loop is entered in precisely the state the hoisted precondition describes.
