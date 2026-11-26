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
      pragma Loop_Invariant (No_Overflows (A, R, I));
      pragma
        Loop_Invariant
          (if I = A'First
           then To_Big (R) = 0
           else
             To_Big (R)
             = Big_Integer_Sum.Sum_Left.Acc.Fold (A, 0) (I - 1));
      Prove_Add (R, Value (A (I)));
      R := R + Value (A (I));
   end loop;
   return R;
end Sum;
