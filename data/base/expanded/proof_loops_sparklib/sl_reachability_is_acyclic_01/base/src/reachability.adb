--
--  Copyright (C) 2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--


package body Reachability is
   pragma Warnings (Off, "function ""Model_Internal"" is not referenced");

   --  Local functions

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

   function Is_Acyclic_Internal
     (X : Extended_Index; M : Memory_Type; S : Memory_Index_Set) return Boolean
   is (X = No_Index
       or else
         (Length (Model_Internal (X, M, S)) > 0
          and then Next (M (Get (Model_Internal (X, M, S), 1))) = No_Index))
   with
     Ghost => Static,
     Pre   => X in M'Range | No_Index and then Valid_Memory (M);

   function Model_Internal
     (X : Extended_Index; M : Memory_Type) return Sequence
   is (Model_Internal (X, M, All_Set (M'First, M'Last)))
   with
     Ghost => Static,
     Pre   => X in M'Range | No_Index and then Valid_Memory (M);

   function Is_Acyclic_Internal
     (X : Extended_Index; M : Memory_Type) return Boolean
   is (Is_Acyclic_Internal (X, M, All_Set (M'First, M'Last)))
   with
     Ghost => Static,
     Pre   => X in M'Range | No_Index and then Valid_Memory (M);

   ---------------------
   -- Is_Acyclic_Impl --
   ---------------------

   function Is_Acyclic_Impl
     (X : Extended_Index; M : Memory_Type) return Boolean
   with
     Pre  => X in M'Range | No_Index and then Valid_Memory (M),
     Post =>
       (Static => Is_Acyclic_Impl'Result = Is_Acyclic_Internal (X, M));

   function Is_Acyclic_Impl
     (X : Extended_Index; M : Memory_Type) return Boolean
   is separate;

   ----------------
   -- Is_Acyclic --
   ----------------

   function Is_Acyclic (X : Extended_Index; M : Memory_Type) return Boolean
   with
     Refined_Post => (Static => Is_Acyclic'Result = Is_Acyclic_Internal (X, M))
   is
   begin
      return Is_Acyclic_Impl (X, M);
   end Is_Acyclic;

end Reachability;
