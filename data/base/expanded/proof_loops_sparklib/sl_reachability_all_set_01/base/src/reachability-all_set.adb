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
      end loop;
   end return;
end All_Set;
