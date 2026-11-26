--
--  Copyright (C) 2018-2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

separate (SPARK.Higher_Order.Fold.Fold_2_Acc)
function Fold (A : Array_Type; Init : Element_Out) return Acc_Array is
   Acc : Element_Out := Init;
begin
   return
      R : Acc_Array (A'Range (1), A'Range (2)) :=
        [others => [others => Init]]
   do
      for I in A'Range (1) loop
         for J in A'Range (2) loop
            R (I, J) := F (A (I, J), Acc);
            if J < A'Last (2) then
               Prove_Ind_Col (A, Acc, I, J);
            elsif I < A'Last (1) then
               Prove_Ind_Row (A, Acc, I);
            else
               Prove_Last (A, Acc);
            end if;
            Acc := R (I, J);
         end loop;
      end loop;
   end return;
end Fold;
