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


   --  Ghost helpers from the body of Impl (expression functions,
   --  so copied whole); they appear in the callees' contracts.

   function Limbo_State
     (Container : List; New_Node : Count_Type; Count : Count_Type)
      return Boolean
   is (Valid_Memory (Memory (Container))
       and then Container.First in Memory (Container)'Range | 0
       and then New_Node in Memory (Container)'Range
       and then Count < Container.Capacity
       and then Container.Nodes (New_Node).Prev'Initialized
       and then Container.Nodes (New_Node).Next'Initialized
       and then Container.Nodes (New_Node).Element'Initialized
       and then
         not Memory_Index_Sets.Contains (Active_Set (Container), New_Node)
       and then Active_List_Valid (Container, Count)
       and then
         Free_Chain_Valid
           (Container,
            Memory_Index_Sets.Add (Active_Set (Container), New_Node))
       and then Free_Count_Correct (Container, Count + 1)
       and then Covered (Container, New_Node))
   with Ghost => Static;
   --  The "limbo" state of node New_Node: off both the active and free lists
   --  (element set, links possibly stale) but with the active list still valid
   --  and holding Count nodes. Both Allocate/Insert_Internal (Count = Length,
   --  active untouched) and Unlink_*_Node/Free (Count = Length - 1, active
   --  minus the unlinked node) use it; the field Container.Length is
   --  independent of Count while a node is in limbo. New_Node is the Covered
   --  Extra node, so its stale Prev does not misclassify it as active.

   function Is_Add
     (S1, S2 : Memory_Index_Set; E : Positive_Count_Type) return Boolean
   is (not Memory_Index_Sets.Contains (S1, E)
       and then Memory_Index_Sets.Contains (S2, E)
       and then Memory_Index_Sets."<=" (S1, S2)
       and then Memory_Index_Sets.Included_Except (S2, S1, E))
   with Ghost => Static;
   --  Return True if S2 is obtained by adding E to S1


   --  Callees, copied as specs with their upstream contracts and marked
   --  `with Import`: SPARK is modular, so a callee's body is never needed
   --  to prove its caller.

   procedure Clear (Container : in out List)
   with
     Import,
     Global => null,
     Pre    => (Static => Structural_Invariant (Container)),
     Post   =>
       (Static =>
          Structural_Invariant (Container) and then Length (Container) = 0);

   procedure Unlink_First_Node (Container : in out List)
   with
     Import,
     Global => null,
     Pre  =>
       (Static =>
          Structural_Invariant (Container) and then Container.Length >= 2),
     Post =>
       (Static =>
          Limbo_State (Container, Container.First'Old, Container.Length - 1)
          and then Container.Length = Container.Length'Old
          and then Container.Last = Container.Last'Old
          and then Container.First'Old in Memory (Container)'Range
          and then
            Is_Add
              (Active_Set (Container),
               Active_Set (Container'Old),
               Container.First'Old));
   --  Unlink the active head (old First) from the active list, leaving it off
   --  both lists in the limbo state with Length unchanged. Length >= 2 so a
   --  non-empty list remains. Does not touch the free store or Length: the
   --  caller returns the old head to the free store (Clear/Delete_First/Move)
   --  and does the single Length write. Shared by those and Splice.

   procedure Free (Container : in out List; X : Count_Type)
   with
     Import,
     Global => null,
     Pre  =>
       (Static =>
          Valid_Memory (Memory (Container))
          and then Container.First in Memory (Container)'Range | 0
          and then X in Memory (Container)'Range
          and then Container.Length < Container.Capacity
          and then not Memory_Index_Sets.Contains (Active_Set (Container), X)
          and then Active_List_Valid (Container, Container.Length)
          and then
            Free_Chain_Valid
              (Container, Memory_Index_Sets.Add (Active_Set (Container), X))
          and then Free_Count_Correct (Container, Container.Length + 1)
          and then Covered (Container, X)),
     Post =>
       (Static =>
          Structural_Invariant (Container)
          and then Container.Length = Container.Length'Old
          and then Container.First = Container.First'Old
          and then Container.Last = Container.Last'Old
          and then
            Memory_Index_Sets."="
              (Active_Set (Container), Active_Set (Container'Old)));
   --  X is a dangling node: touched, off the active list, and off the free
   --  chain (the limbo state Allocate produces, symmetric to its Post). Free
   --  returns it to the free store, restoring Structural_Invariant with the
   --  same Length/First/Last.

   procedure Delete_First (Container : in out List; Count : Count_Type)
   with
     Global => null,
     Pre    => (Static => Structural_Invariant (Container)),
     Post   => (Static => Structural_Invariant (Container));


end Lists;
