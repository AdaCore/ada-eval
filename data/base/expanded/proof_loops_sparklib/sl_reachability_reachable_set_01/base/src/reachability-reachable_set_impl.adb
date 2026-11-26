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
         if Contains (S, C) then
            return;
         end if;
         S := Add (S, C);
         Not_Seen := Remove (Not_Seen, C);
         C := Next (M (C));
      end loop;
   end return;
end Reachable_Set_Impl;
