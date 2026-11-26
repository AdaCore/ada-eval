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
      pragma Loop_Invariant (No_Overflows (A, R, I, A'First (2)));
      pragma
        Loop_Invariant
          (if I = A'First (1)
           then To_Big (R) = 0
           else
             To_Big (R)
             = Big_Integer_Sum.Fold_Sum.Acc.Fold (A, 0)
                 (I - 1, A'Last (2)));
      for J in A'Range (2) loop
         pragma Loop_Invariant (No_Overflows (A, R, I, J));
         pragma
           Loop_Invariant
             (if I = A'First (1) and J = A'First (2)
              then To_Big (R) = 0
              elsif J = A'First (2)
              then
                To_Big (R)
                = Big_Integer_Sum.Fold_Sum.Acc.Fold (A, 0)
                    (I - 1, A'Last (2))
              else
                To_Big (R)
                = Big_Integer_Sum.Fold_Sum.Acc.Fold (A, 0) (I, J - 1));
         Prove_Add (R, Value (A (I, J)));
         R := R + Value (A (I, J));
      end loop;
   end loop;
   return R;
end Sum;
