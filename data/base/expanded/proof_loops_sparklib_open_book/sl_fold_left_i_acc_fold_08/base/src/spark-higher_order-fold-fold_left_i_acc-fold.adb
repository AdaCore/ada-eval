--
--  Copyright (C) 2018-2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

separate (SPARK.Higher_Order.Fold.Fold_Left_I_Acc)
function Fold (A : Array_Type; Init : Element_Out) return Acc_Array is
   Acc : Element_Out := Init;
begin
   return R : Acc_Array (A'First .. A'Last) := [others => Init] do
      for I in A'Range loop
         R (I) := F (A (I), I, Acc);
         if I /= A'Last then
            Prove_Ind (A, Acc, I);
         else
            Prove_Last (A, Acc);
         end if;
         Acc := R (I);
      end loop;
   end return;
end Fold;
