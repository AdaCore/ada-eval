--
--  Copyright (C) 2018-2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

separate (SPARK.Higher_Order.Fold.Sum.Big_Integer_Sum)
procedure Sum_Cst (A : Array_Type; C : Element_Out) is
begin
   for I in A'Range loop
      if Value (A (I)) /= C then
         return;
      end if;
   end loop;
end Sum_Cst;
