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

   --  From the visible part of SPARK.Containers.Formal.Doubly_Linked_Lists.

   type Cursor is record
      Node : Count_Type := 0;
   end record;

   No_Element : constant Cursor := Cursor'(Node => 0);

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

   function Active_Set (Container : List) return Memory_Index_Set
   is (Reachable_Set (Container.First, Memory (Container)))
   with
     Ghost => Static,
     Pre   =>
       Valid_Memory (Memory (Container))
       and then Container.First in Memory (Container)'Range | 0;
   --  The set of nodes on the active list (reachable from First via Next)

   function Active_List_Valid
     (Container : List; Count : Count_Type) return Boolean
   is (Is_Acyclic (Container.First, Memory (Container))

       --  The active list from First is acyclic and has exactly Count nodes

       and then
         Memory_Index_Sets.Length (Active_Set (Container))
         = Big_Conversions.To_Big (Count)

       --  Every node reachable from First is allocated (Prev /= -1) and
       --  doubly linked with its neighbours.

       and then
         (for all I of Active_Set (Container) =>
            Container.Nodes (I).Prev /= -1)
       and then
         (for all I of Active_Set (Container) =>
            (if Container.Nodes (I).Next = 0 then I = Container.Last))
       and then
         (for all I of Active_Set (Container) =>
            (if Container.Nodes (I).Prev = 0
             then I = Container.First
             else
               Memory_Index_Sets.Contains
                 (Active_Set (Container), Container.Nodes (I).Prev)
               and then Container.Nodes (Container.Nodes (I).Prev).Next = I))

       --  First and Last are the endpoints of the active list

       and then
         (if Container.First = 0
          then Container.Last = 0 and Count = 0
          else
            Container.Nodes (Container.First).Prev = 0
            and then Container.Last in Memory (Container)'Range
            and then
              Reachable (Container.First, Memory (Container), Container.Last)
            and then Container.Nodes (Container.Last).Next = 0))
   with
     Ghost => Static,
     Pre   =>
       Valid_Memory (Memory (Container))
       and then Container.First in Memory (Container)'Range | 0;
   --  Structural facts about the active list

   function Free_Chain_Valid
     (Container : List; Excluded : Memory_Index_Set) return Boolean
   is (Container.Free < 0
       or else
         (Is_Acyclic (Container.Free, Memory (Container))
          and then
            (for all I of Reachable_Set (Container.Free, Memory (Container)) =>
               not Memory_Index_Sets.Contains (Excluded, I)
               and then Container.Nodes (I).Prev = -1)))
   with
     Ghost => Static,
     Pre   =>
       Valid_Memory (Memory (Container))
       and then Container.First in Memory (Container)'Range | 0;
   --  Structural part of the free list, without its length: when Free is
   --  non-negative the free nodes form an explicit acyclic chain, each
   --  deallocated (Prev = -1) and outside the Excluded set (the used nodes).
   --  When Free is negative the free nodes are the implicit never-used tail,
   --  so there is no chain to constrain. Kept length-free so the limbo state
   --  between Allocate and Insert_Internal (which has a different count and a
   --  larger used set) can reuse it.

   function Free_Count_Correct
     (Container : List; Used : Count_Type) return Boolean
   is (if Container.Free < 0
       then Container.Free = -Used - 1
       else
         Memory_Index_Sets.Length
           (Reachable_Set (Container.Free, Memory (Container)))
         = Big_Conversions.To_Big (Container.Capacity - Used))
   with
     Ghost => Static,
     Pre   =>
       Valid_Memory (Memory (Container))
       and then Container.First in Memory (Container)'Range | 0;
   --  The free nodes number Capacity - Used, where Used is the number of
   --  touched nodes. When Free is negative the touched region is 1 .. Used, so
   --  Free = -Used - 1 encodes the count directly.

   function Covered (Container : List; Extra : Extended_Index) return Boolean
   is (if Container.Free >= 0
       then
         (for all I in 1 .. Container.Capacity =>
            I = Extra
            or else
              (if Container.Nodes (I).Prev = -1
               then Reachable (Container.Free, Memory (Container), I)
               else Memory_Index_Sets.Contains (Active_Set (Container), I)))
       else
         (for all I in Memory (Container)'Range =>
            I = Extra
            or else Memory_Index_Sets.Contains (Active_Set (Container), I)))
   with
     Ghost => Static,
     Pre   =>
       Valid_Memory (Memory (Container))
       and then Container.First in Memory (Container)'Range | 0;
   --  Partition of the touched region: when Free is negative, every touched
   --  node is then active or the exceptional dangling node Extra (a node in
   --  limbo between the active and free lists; 0 means none). Otherwise, it is
   --  active if its Prev field is positive and free otherwise.

   function Structural_Invariant
     (Container : List; Count : Count_Type) return Boolean
   is (Valid_Memory (Memory (Container))
       and then Container.First in Memory (Container)'Range | 0
       and then Active_List_Valid (Container, Count)
       and then Free_Chain_Valid (Container, Active_Set (Container))
       and then Free_Count_Correct (Container, Count)
       and then Covered (Container, 0))
   with Ghost => Static;
   --  The structural invariant with the active-node count given explicitly as
   --  Count rather than read from Container.Length. Used by the Length-neutral
   --  Insert_*_Node primitives, whose Post is Structural_Invariant (Container,
   --  Count + 1) while Container.Length still holds the pre-insert value; the
   --  caller's single Length write then reconciles the field with the count.

   function Structural_Invariant (Container : List) return Boolean
   is (Structural_Invariant (Container, Container.Length))
   with Ghost => Static;
   --  Silver structural invariant. Carried as Static Pre/Post on the
   --  operations and threaded through their loop invariants/variants.


   --  Callees, copied as specs with their upstream contracts and marked
   --  `with Import`: SPARK is modular, so a callee's body is never needed
   --  to prove its caller.

   function Has_Element (Container : List; Position : Cursor) return Boolean
   with
     Import,
     Global => null,
     Pre    => (Static => Structural_Invariant (Container)),
     Post   =>
       (Static =>
          Has_Element'Result
          = (Position /= No_Element
             and then
               Memory_Index_Sets.Contains
                 (Active_Set (Container), Position.Node)));

   --  The following ghost helpers express, without any model, the exact
   --  conditions under which the operations raise an exception. They are used
   --  in the Exit_Cases below so that those contracts state precisely when,
   --  and which, exception is raised.

   function Is_Relevant (Container : List; Position : Cursor) return Boolean
   is (Position = No_Element or else Has_Element (Container, Position))
   with Ghost => Static, Pre => Structural_Invariant (Container);
   --  Position is a cursor which is relevant for Container: it is No_Element
   --  or it designates an element in Container.

   --     function Reverse_Find
   --       (Container : List; Item : Element_Type;
   --        Position : Cursor := No_Element) return Cursor
   --     with
   --       Pre        => (Static => Structural_Invariant (Container)),
   --       Side_Effects,
   --       Global     => null,
   --       Exit_Cases =>
   --         (Is_Relevant (Container, Position) => Normal_Return,
   --          others                            =>
   --            (Exception_Raised => Program_Error));

   --  and its body raises Program_Error when the cursor is not relevant.
   --  Under FSF 16.1 a Side_Effects function that can raise draws the flow
   --  message `function "Reverse_Find" does not return on some paths`
   --  (reproduced on a five-line program with no SPARKlib in it), which a
   --  clean proof forbids.  So the Exit_Cases' own `Normal_Return` guard is
   --  hoisted into the precondition and the raise removed: after upstream's
   --  guard the loop faces exactly the same state either way -- CFirst is
   --  Container.Last, or it is a node Has_Element accepts -- so the loop's
   --  proof burden is unchanged.

   function Reverse_Find
     (Container : List; Item : Element_Type; Position : Cursor := No_Element)
      return Cursor
   with
     Pre        =>
       (Static =>
          Structural_Invariant (Container)
          and then Is_Relevant (Container, Position)),
     Side_Effects,
     Global     => null;


end Lists;
