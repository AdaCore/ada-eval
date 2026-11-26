--
--  Copyright (C) 2018-2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

separate (SPARK.Higher_Order.Fold.Sum_2.Big_Integer_Sum)
procedure Sum_Cst (A : Array_Type; C : Element_Out) is
begin
   if A'Length (2) > 0 then
      for I in A'Range (1) loop
         for J in A'Range (2) loop
            if Value (A (I, J)) /= C then
               return;
            end if;
         end loop;
      end loop;
   end if;
end Sum_Cst;
