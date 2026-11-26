--
--  Copyright (C) 2018-2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

separate (SPARK.Higher_Order.Fold.Sum)
function Sum (A : Array_Type) return Element_Out is
   R : Element_Out := Zero;
begin
   Prove_Zero;
   for I in A'Range loop
      Prove_Add (R, Value (A (I)));
      R := R + Value (A (I));
   end loop;
   return R;
end Sum;
