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
         pragma
           Loop_Invariant
             (if I = A'First (1)
              then Acc = Init
              else Acc = R (I - 1, A'Last (2)));
         pragma Loop_Invariant (Ind_Prop (A, Acc, I, A'First (2)));
         pragma
           Loop_Invariant
             (if I > A'First (1)
              then
                Ind_Prop (A, Init, A'First (1), A'First (2))
                and then
                  R (A'First (1), A'First (2))
                  = F (A (A'First (1), A'First (2)), Init));
         pragma
           Loop_Invariant
             (for all K in A'Range (1) =>
                (if K < I and then K > A'First (1)
                 then
                   Ind_Prop (A, R (K - 1, A'Last (2)), K, A'First (2))
                   and then
                     R (K, A'First (2))
                     = F (A (K, A'First (2)), R (K - 1, A'Last (2)))));
         pragma
           Loop_Invariant
             (for all K in A'Range (1) =>
                (for all L in A'Range (2) =>
                   (if K < I and then L > A'First (2)
                    then
                      Ind_Prop (A, R (K, L - 1), K, L)
                      and then R (K, L) = F (A (K, L), R (K, L - 1)))));
         for J in A'Range (2) loop
            pragma
              Loop_Invariant
                (if I > A'First (1) or else J > A'First (2)
                 then
                   Ind_Prop (A, Init, A'First (1), A'First (2))
                   and then
                     R (A'First (1), A'First (2))
                     = F (A (A'First (1), A'First (2)), Init));
            pragma
              Loop_Invariant
                (for all K in A'Range (1) =>
                   (if K > A'First (1)
                      and then
                        (K < I or else (K = I and then J > A'First (2)))
                    then
                      Ind_Prop (A, R (K - 1, A'Last (2)), K, A'First (2))
                      and then
                        R (K, A'First (2))
                        = F
                            (A (K, A'First (2)),
                             R (K - 1, A'Last (2)))));
            pragma
              Loop_Invariant
                (for all K in A'Range (1) =>
                   (for all L in A'Range (2) =>
                      (if L > A'First (2)
                         and then (K < I or else (K = I and then L < J))
                       then
                         Ind_Prop (A, R (K, L - 1), K, L)
                         and then
                           R (K, L) = F (A (K, L), R (K, L - 1)))));
            pragma
              Loop_Invariant
                (if J /= A'First (2)
                 then Acc = R (I, J - 1)
                 elsif I /= A'First (1)
                 then Acc = R (I - 1, A'Last (2))
                 else Acc = Init);
            pragma Loop_Invariant (Ind_Prop (A, Acc, I, J));
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
      pragma
        Assert
          (for all K in A'Range (1) =>
             (if K > A'First (1)
              then
                Ind_Prop (A, R (K - 1, A'Last (2)), K, A'First (2))
                and then
                  R (K, A'First (2))
                  = F (A (K, A'First (2)), R (K - 1, A'Last (2)))));
   end return;
end Fold;
