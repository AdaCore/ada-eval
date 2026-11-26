--
--  Copyright (C) 2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;

with SPARK.Big_Integers; use SPARK.Big_Integers;
with SPARK.Containers.Functional.Infinite_Sequences;
with SPARK.Containers.Functional.Sets;

generic
   type Index_Type is range <>;
   No_Index : Index_Type'Base;
   type Cell_Type is private;
   type Memory_Type is array (Index_Type range <>) of Cell_Type;
   with function Next (C : Cell_Type) return Index_Type'Base;

   with package Memory_Index_Sets is new
     SPARK.Containers.Functional.Sets (Index_Type);

   with package Memory_Index_Sequences is new
     SPARK.Containers.Functional.Infinite_Sequences
       (Index_Type,
        Use_Logical_Equality => True);

package Reachability with SPARK_Mode, Always_Terminates
is

   pragma Assert (No_Index not in Index_Type);
   --  No_Index should not be a valid index

   subtype Extended_Index is Index_Type'Base;

   function Valid_Memory (M : Memory_Type) return Boolean
   is (M'First = Index_Type'First
       and then (for all C of M => Next (C) in M'Range | No_Index))
   with Annotate => (GNATprove, Inline_For_Proof);

   package Big_Conversions is
      package Memory_Index_To_Big is new Signed_Conversions (Extended_Index);
      use Memory_Index_To_Big;
      function To_Big (X : Extended_Index) return Big_Integer
      renames Memory_Index_To_Big.To_Big_Integer;
   end Big_Conversions;
   use Big_Conversions;

   use Memory_Index_Sets;
   subtype Memory_Index_Set is Memory_Index_Sets.Set;

   use Memory_Index_Sequences;

   function All_Set (First, Last : Index_Type'Base) return Memory_Index_Set
   with
     Ghost  => Static,
     Import,
     Global => null,
     Pre    =>
       Last < First or else (First in Index_Type and then Last in Index_Type),
     Post   =>
       Length (All_Set'Result)
       = (if Last < First then 0 else To_Big (Last) - To_Big (First) + 1)
       and then (for all I in First .. Last => Contains (All_Set'Result, I))
       and then (for all E of All_Set'Result => E in First .. Last);

   function Reachable_Set_Internal
     (X : Extended_Index; M : Memory_Type; S : Memory_Index_Set)
      return Memory_Index_Set
   is (if X = No_Index or else not Contains (S, X)
       then Empty_Set
       else Add (Reachable_Set_Internal (Next (M (X)), M, Remove (S, X)), X))
   with
     Ghost              => Static,
     Pre                => X in M'Range | No_Index and then Valid_Memory (M),
     Subprogram_Variant => (Decreases => Length (S)),
     Post               =>
       Reachable_Set_Internal'Result <= S
       and then (for all Y of Reachable_Set_Internal'Result => Y in M'Range);

   function Model_Internal
     (X : Extended_Index; M : Memory_Type; S : Memory_Index_Set)
      return Sequence
   is (if X = No_Index or else not Contains (S, X)
       then Empty_Sequence
       else Add (Model_Internal (Next (M (X)), M, Remove (S, X)), X))
   with
     Ghost              => Static,
     Subprogram_Variant => (Decreases => Length (S)),
     Pre                => X in M'Range | No_Index and then Valid_Memory (M),
     Post               =>
       Length (Reachable_Set_Internal (X, M, S))
       = Length (Model_Internal'Result)
       and then
         (for all Y of Reachable_Set_Internal (X, M, S) =>
            Find (Model_Internal'Result, Y) > 0)
       and then
         (for all I in Model_Internal'Result =>
            Contains
              (Reachable_Set_Internal (X, M, S),
               Get (Model_Internal'Result, I)));

   function Model_Internal
     (X : Extended_Index; M : Memory_Type) return Sequence
   is (Model_Internal (X, M, All_Set (M'First, M'Last)))
   with
     Ghost => Static,
     Pre   => X in M'Range | No_Index and then Valid_Memory (M);

   function Is_Acyclic (X : Extended_Index; M : Memory_Type) return Boolean
   with
     Import,
     Global => null,
     Pre    => X in M'Range | No_Index and then Valid_Memory (M),
     Post   => (Static => (if X = No_Index then Is_Acyclic'Result));

   --  The first of Model's two loops: it walks the structure from X following
   --  Next, accumulating the indexes it meets into R_Seq.  R_Seq is therefore
   --  the model of X read backwards, which Model's second loop then reverses.

   function Model_Reversed
     (X : Extended_Index; M : Memory_Type) return Sequence
   with
     Pre  =>
       X in M'Range | No_Index
       and then Valid_Memory (M)
       and then Is_Acyclic (X, M),
     Post =>
       (Static =>
          Length (Model_Reversed'Result) = Length (Model_Internal (X, M))
          and then
            (if X /= No_Index
             then
               In_Range
                 (Length (Model_Reversed'Result),
                  0,
                  To_Big (M'Last) - To_Big (M'First) + 1))
          and then
            (for all I in Model_Reversed'Result =>
               Get (Model_Reversed'Result, I)
               = Get
                   (Model_Internal (X, M),
                    Last (Model_Internal (X, M)) - I + 1)));

end Reachability;
