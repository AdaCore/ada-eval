--
--  Copyright (C) 2004-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;

with SPARK.Big_Integers; use SPARK.Big_Integers;
with SPARK.Containers.Functional.Infinite_Sequences;
with SPARK.Containers.Functional.Sets;
with SPARK.Containers.Types; use SPARK.Containers.Types;
with SPARK.Higher_Order.Reachability;

with Element_Type_Defs; use Element_Type_Defs;

package Lists with SPARK_Mode, Always_Terminates
is

   subtype Positive_Count_Type is Count_Type range 1 .. Count_Type'Last;
   --  From Formal_Model in SPARK.Containers.Formal.Doubly_Linked_Lists.

   --  Representation, from the private part of
   --  SPARK.Containers.Formal.Doubly_Linked_Lists, which is
   --  `pragma SPARK_Mode (Off); --  #BODYMODE` in a normal build
   --  and is switched On for the proof test.

   --  The whole node is marked Relaxed_Initialization, so that a node that
   --  has never been allocated (in the never-used tail of the free region)
   --  need have none of its fields initialized. The Element field is aliased
   --  so that Reference and Constant_Reference can hand out an access
   --  designating it.

   type Node_Type is record
      Prev    : Count_Type'Base;
      Next    : Count_Type;
      Element : aliased Element_Type;
   end record
   with Relaxed_Initialization;

   function "=" (L, R : Node_Type) return Boolean is abstract;

   type Node_Array is array (Count_Type range <>) of Node_Type;
   function "=" (L, R : Node_Array) return Boolean is abstract;

   type List (Capacity : Count_Type) is record
      Free   : Count_Type'Base := -1;
      Length : Count_Type := 0;
      First  : Count_Type := 0;
      Last   : Count_Type := 0;
      Nodes  : Node_Array (1 .. Capacity);
   end record
   with
     Ghost_Predicate =>
       (Static =>
          Length <= Capacity
          and then First <= Capacity
          and then Last <= Capacity
          and then Free in -Capacity - 1 .. Capacity
          and then
            (for all I in 1 .. Capacity =>
               (if Free >= 0 or else I <= -(Free + 1)
                then
                  Nodes (I).Prev'Initialized
                  and then Nodes (I).Next'Initialized
                  and then
                    Nodes (I).Prev
                    in -1 .. (if Free >= 0 then Capacity else -(Free + 1))
                  and then
                    Nodes (I).Next
                    <= (if Free >= 0 then Capacity else -(Free + 1))
                  and then
                    (if Nodes (I).Prev /= -1
                     then Nodes (I).Element'Initialized))));
   --  The free-list encoding determines which nodes have their Prev/Next
   --  bookkeeping initialized: when Free is non-negative every node has been
   --  touched (the free list is an explicit chain over all cells); when Free
   --  is negative only the nodes below abs Free have been touched (the cells
   --  from abs Free to Capacity are the never-used tail). Among the touched
   --  nodes, a deallocated one (Prev = -1) has an uninitialized element and an
   --  allocated one has an initialized element. These cheap structural facts
   --  are carried in a Ghost_Predicate rather than a Type_Invariant: a
   --  Type_Invariant would be nicer (it may be transiently broken inside the
   --  package body and is only re-checked at the package boundary), but
   --  GNATprove would then reject instances of Generic_Sorting occurring
   --  outside of Doubly_Linked_Lists. The deep model invariant is gold and
   --  stays out of the Impl child.

   function Length (Container : List) return Count_Type
   is (Container.Length);

   --  Structural (ghost) model, from
   --  SPARK.Containers.Formal.Doubly_Linked_Lists.Impl.

   package Structural_Model
     with Ghost => Static
   is

      package Memory_Index_Sets is new
        SPARK.Containers.Functional.Sets (Positive_Count_Type);

      package Memory_Index_Sequences is new
        SPARK.Containers.Functional.Infinite_Sequences
          (Positive_Count_Type,
           Use_Logical_Equality => True);

      type Nodes_Type_Base is array (Positive_Count_Type range <>) of Node_Type
      with Predicate => (for all N of Nodes_Type_Base => N.Next'Initialized);

      function Next_Link (N : Node_Type) return Count_Type
      is (N.Next)
      with
        Annotate => (GNATprove, Inline_For_Proof),
        Pre      => N.Next'Initialized;

      package Node_Reachability is new
        SPARK.Higher_Order.Reachability
          (Index_Type                            => Positive_Count_Type,
           No_Index                              => 0,
           Cell_Type                             => Node_Type,
           Memory_Type                           => Nodes_Type_Base,
           Next                                  => Next_Link,
           Memory_Index_Sets                     => Memory_Index_Sets,
           Memory_Index_Sequences                => Memory_Index_Sequences,
           Automatically_Instantiate_Definitions => False);

   end Structural_Model;

   use Structural_Model;
   use Structural_Model.Node_Reachability;

   function Memory (Container : List) return Nodes_Type_Base
   is (if Container.Free >= 0
       then Nodes_Type_Base (Container.Nodes)
       else Nodes_Type_Base (Container.Nodes (1 .. -(Container.Free + 1))))
   with Ghost => Static;
   --  The node array as a Positive_Count_Type-indexed memory for the
   --  reachability model. When Free is negative only the touched prefix
   --  (1 .. abs Free - 1) has initialized links, so the never-used tail is
   --  sliced off (Valid_Memory would otherwise read uninitialized Next).


   --  Promoted, with its contract unchanged, from the ghost procedure
   --  nested in Copy's declarative part; only its indentation moved.

   procedure Rebuild_Free_List (L : List; Base, Cont : Count_Type)
   with
     Ghost => Static,
     Pre   =>
       Valid_Memory (Memory (L))
       and then L.Free = Base
       and then Base in 1 .. L.Capacity
       and then Cont in 0 .. Base - 1
       and then Is_Acyclic (Cont, Memory (L))
       and then
         (for all J in Base .. L.Capacity - 1 => L.Nodes (J).Next = J + 1)
       and then L.Nodes (L.Capacity).Next = Cont
       and then (for all J in Base .. L.Capacity => L.Nodes (J).Prev = -1)
       and then
         (for all I of Reachable_Set (Cont, Memory (L)) => I <= Base - 1)
       and then
         (for all I of Reachable_Set (Cont, Memory (L)) =>
            L.Nodes (I).Prev = -1),
     Post  =>
       Is_Acyclic (Base, Memory (L))
       and then
         (for all I of Reachable_Set (Base, Memory (L)) =>
            I in Base .. L.Capacity
            or else
              Memory_Index_Sets.Contains
                (Reachable_Set (Cont, Memory (L)), I))
       and then
         (for all I of Reachable_Set (Base, Memory (L)) =>
            L.Nodes (I).Prev = -1)
       and then
         (for all I in Base .. L.Capacity =>
            Memory_Index_Sets.Contains
              (Reachable_Set (Base, Memory (L)), I))
       and then
         (for all I of Reachable_Set (Cont, Memory (L)) =>
            Memory_Index_Sets.Contains
              (Reachable_Set (Base, Memory (L)), I))
       and then
         Memory_Index_Sets.Length (Reachable_Set (Base, Memory (L)))
         = Big_Conversions.To_Big (L.Capacity - Base + 1)
           + Memory_Index_Sets.Length (Reachable_Set (Cont, Memory (L)));
   --  The tail Base -> Base + 1 -> ... -> Capacity chains into Source's
   --  free head Cont (< Base). The merged free list from Base is then
   --  acyclic, its reachable set is Base .. Capacity plus Cont's chain,
   --  every node deallocated, with (Capacity - Base + 1) + |Cont chain|
   --  nodes. Walk the tail from Capacity down to establish these by
   --  induction. Carried as a local lemma so the induction's loop
   --  invariants do not enlarge Copy's own postcondition checks.

end Lists;
