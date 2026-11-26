--
--  Copyright (C) 2018-2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

separate (SPARK.Higher_Order.Fold.Fold_Right_Acc)
function Fold (A : Array_Type; Init : Element_Out) return Acc_Array is
   Acc : Element_Out := Init;
begin
   return R : Acc_Array (A'First .. A'Last) := [others => Init] do
      for I in reverse A'Range loop
         pragma Assert (Ind_Prop (A, Acc, I));
         R (I) := F (A (I), Acc);
         pragma
           Loop_Invariant
             (Ind_Prop (A, Init, A'Last)
              and then R (A'Last) = F (A (A'Last), Init));
         pragma
           Loop_Invariant
             (for all K in I .. A'Last =>
                (if K < A'Last
                 then
                   Ind_Prop (A, R (K + 1), K)
                   and then R (K) = F (A (K), R (K + 1))));
         pragma
           Loop_Invariant
             (if I = A'Last then Acc = Init else Acc = R (I + 1));
         if I /= A'First then
            Prove_Ind (A, Acc, I);
         else
            Prove_Last (A, Acc);
         end if;
         Acc := R (I);
      end loop;
   end return;
end Fold;
