--
--  Copyright (C) 2018-2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

separate (SPARK.Higher_Order.Fold.Fold_Left_Acc)
function Fold (A : Array_Type; Init : Element_Out) return Acc_Array is
   Acc : Element_Out := Init;
begin
   return R : Acc_Array (A'First .. A'Last) := [others => Init] do
      for I in A'Range loop
         pragma Assert (Ind_Prop (A, Acc, I));
         R (I) := F (A (I), Acc);
         pragma
           Loop_Invariant
             (Ind_Prop (A, Init, A'First)
              and then R (A'First) = F (A (A'First), Init));
         pragma
           Loop_Invariant
             (for all K in A'First .. I =>
                (if K > A'First
                 then
                   Ind_Prop (A, R (K - 1), K)
                   and then R (K) = F (A (K), R (K - 1))));
         pragma
           Loop_Invariant
             (if I = A'First then Acc = Init else Acc = R (I - 1));
         if I /= A'Last then
            Prove_Ind (A, Acc, I);
         else
            Prove_Last (A, Acc);
         end if;
         Acc := R (I);
      end loop;
   end return;
end Fold;
