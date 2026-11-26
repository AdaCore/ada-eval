--
--  Copyright (C) 2018-2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

separate (SPARK.Higher_Order.Fold.Fold_Right)
function Fold (A : Array_Type; Init : Element_Out) return Element_Out is
begin
   return R : Element_Out := Init do
      for I in reverse A'Range loop
         pragma
           Loop_Invariant
             (Ind_Prop (A, R, I)
              and then F (A (I), R) = Acc.Fold (A, Init) (I));
         if I /= A'First then
            Acc.Prove_Ind (A, R, I);
         end if;
         R := F (A (I), R);
      end loop;
   end return;
end Fold;
