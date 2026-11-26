--
--  Copyright (C) 2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

separate (Reachability)
function All_Set (First, Last : Index_Type'Base) return Memory_Index_Set is
begin
   return S : Memory_Index_Set do
      for I in First .. Last loop
         S := Add (S, I);
         pragma
           Loop_Invariant (Length (S) = To_Big (I) - To_Big (First) + 1);
         pragma Loop_Invariant (for all K in First .. I => Contains (S, K));
         pragma Loop_Invariant (for all K of S => K in First .. I);
      end loop;
   end return;
end All_Set;
