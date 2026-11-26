--
--  Copyright (C) 2018-2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

separate (SPARK.Higher_Order.Fold.Fold_2)
function Fold (A : Array_Type; Init : Element_Out) return Element_Out is
begin
   return R : Element_Out := Init do
      if A'Length (2) > 0 then
         for I in A'Range (1) loop
            pragma
              Loop_Invariant
                (Ind_Prop (A, R, I, A'First (2))
                 and then
                   F (A (I, A'First (2)), R)
                   = Acc.Fold (A, Init) (I, A'First (2)));
            for J in A'Range (2) loop
               pragma
                 Loop_Invariant
                   (Ind_Prop (A, R, I, J)
                    and then
                      F (A (I, J), R) = Acc.Fold (A, Init) (I, J));
               if J /= A'Last (2) then
                  Acc.Prove_Ind_Col (A, R, I, J);
               elsif I /= A'Last (1) then
                  Acc.Prove_Ind_Row (A, R, I);
               end if;

               R := F (A (I, J), R);
            end loop;
         end loop;
      end if;
   end return;
end Fold;
