--
--  Copyright (C) 2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

separate (Reachability)
function Reachable_Set_Impl
  (X : Extended_Index; M : Memory_Type) return Memory_Index_Set
is
   Not_Seen : Memory_Index_Set := All_Set (M'First, M'Last)
   with Ghost => Static;
   C        : Extended_Index := X;
begin
   return S : Memory_Index_Set do
      while C /= No_Index loop
         pragma Loop_Variant (Static => (Decreases => Length (Not_Seen)));
         pragma Loop_Invariant (Static => C in M'Range);
         pragma
           Loop_Invariant
             (Static =>
                Length (S) + Length (Not_Seen)
                = To_Big (M'Last) - To_Big (M'First) + 1);
         pragma
           Loop_Invariant
             (Static =>
                (for all I in M'Range =>
                   Contains (S, I) /= Contains (Not_Seen, I)));
         pragma
           Loop_Invariant
             (Static =>
                Length (Reachable_Set_Internal (X, M))
                = Length (S)
                  + Length (Reachable_Set_Internal (C, M, Not_Seen)));
         pragma
           Loop_Invariant
             (Static =>
                Reachable_Set_Internal (X, M)
                = Union (S, Reachable_Set_Internal (C, M, Not_Seen)));
         if Contains (S, C) then
            return;
         end if;
         S := Add (S, C);
         Not_Seen := Remove (Not_Seen, C);
         C := Next (M (C));
      end loop;
   end return;
end Reachable_Set_Impl;
