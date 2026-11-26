--
--  Copyright (C) 2018-2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

separate (SPARK.Higher_Order.Fold.Fold_Left)
function Fold (A : Array_Type; Init : Element_Out) return Element_Out is
begin
   return R : Element_Out := Init do
      for I in A'Range loop
         if I /= A'Last then
            Acc.Prove_Ind (A, R, I);
         end if;
         R := F (A (I), R);
      end loop;
   end return;
end Fold;
