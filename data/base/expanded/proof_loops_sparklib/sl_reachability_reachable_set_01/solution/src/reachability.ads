--
--  Copyright (C) 2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Warnings (Off, "use clause for package ""Memory_Index_Sequences"" ha*");
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

   --  The set of reachable memory indexes from X in M

   function Reachable_Set
     (X : Extended_Index; M : Memory_Type) return Memory_Index_Set
   with
     Pre  => X in M'Range | No_Index and then Valid_Memory (M),
     Post =>
       (Static =>
          (if X = No_Index
           then
             Is_Empty (Reachable_Set'Result)
             and then Length (Reachable_Set'Result) = 0
           else
             Contains (Reachable_Set'Result, X)
             and then
               Length (Reachable_Set'Result)
               <= To_Big (M'Last) - To_Big (M'First) + 1)
          and then (for all I of Reachable_Set'Result => I in M'Range));

end Reachability;
