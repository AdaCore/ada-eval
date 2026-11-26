--
--  Copyright (C) 2018-2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

separate (SPARK.Higher_Order.Fold.Count_2)
procedure Count_Zero (A : Array_Type) is
begin
   if A'Length (2) > 0 then
      for I in A'Range (1) loop
         for J in A'Range (2) loop
            null;
         end loop;
      end loop;
   end if;
end Count_Zero;
