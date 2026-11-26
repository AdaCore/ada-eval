--
--  Copyright (C) 2018-2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

separate (SPARK.Higher_Order.Fold.Sum_2)
function Sum (A : Array_Type) return Element_Out is
   R : Element_Out := Zero;
begin
   Prove_Zero;
   if A'Length (2) = 0 then
      return R;
   end if;

   for I in A'Range (1) loop
      for J in A'Range (2) loop
         Prove_Add (R, Value (A (I, J)));
         R := R + Value (A (I, J));
      end loop;
   end loop;
   return R;
end Sum;
