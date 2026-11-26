--
--  Copyright (C) 2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

package body Reachability is

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

   function Reachable_Set_Internal
     (X : Extended_Index; M : Memory_Type) return Memory_Index_Set
   is (Reachable_Set_Internal (X, M, All_Set (M'First, M'Last)))
   with
     Ghost => Static,
     Pre   => X in M'Range | No_Index and then Valid_Memory (M);

   ------------------------
   -- Reachable_Set_Impl --
   ------------------------

   function Reachable_Set_Impl
     (X : Extended_Index; M : Memory_Type) return Memory_Index_Set
   with
     Pre  => X in M'Range | No_Index and then Valid_Memory (M),
     Post =>
       (Static =>
          Reachable_Set_Impl'Result = Reachable_Set_Internal (X, M)
          and then
            Length (Reachable_Set_Impl'Result)
            = Length (Reachable_Set_Internal (X, M))
          and then
            Length (Reachable_Set_Impl'Result)
            <= (if M'Last < M'First
                then 0
                else To_Big (M'Last) - To_Big (M'First) + 1));

   function Reachable_Set_Impl
     (X : Extended_Index; M : Memory_Type) return Memory_Index_Set
   is separate;

   -------------------
   -- Reachable_Set --
   -------------------

   function Reachable_Set
     (X : Extended_Index; M : Memory_Type) return Memory_Index_Set
   with
     Refined_Post =>
       (Static =>
          Reachable_Set'Result = Reachable_Set_Internal (X, M)
          and then
            Length (Reachable_Set'Result)
            = Length (Reachable_Set_Internal (X, M))
          and then
            Length (Reachable_Set'Result)
            <= (if M'Last < M'First
                then 0
                else To_Big (M'Last) - To_Big (M'First) + 1))
   is
   begin
      return Reachable_Set_Impl (X, M);
   end Reachable_Set;

end Reachability;
