--
--  Copyright (C) 2018-2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

separate (SPARK.Higher_Order.Fold.Sum_2.Big_Integer_Sum)
procedure Update_Sum (A1, A2 : Array_Type; I : Index_1; J : Index_2)
is
begin
   for K in A1'Range (1) loop
      pragma
        Loop_Invariant
          (if K < I or else (K = I and then A1'First (2) < J)
           then
             Fold_Sum.Acc.Fold (A1, 0) (K, A1'First (2))
             = Fold_Sum.Acc.Fold (A2, 0) (K, A1'First (2))
           else
             Fold_Sum.Acc.Fold (A1, 0) (K, A1'First (2))
             - To_Big (Value (A1 (I, J)))
             = Fold_Sum.Acc.Fold (A2, 0) (K, A1'First (2))
               - To_Big (Value (A2 (I, J))));
      for L in A1'Range (2) loop
         if K /= I or else L /= J then
            pragma Assert (Value (A1 (K, L)) = Value (A2 (K, L)));
         end if;
         pragma
           Loop_Invariant
             (if K < I or else (K = I and then L < J)
              then
                Fold_Sum.Acc.Fold (A1, 0) (K, L)
                = Fold_Sum.Acc.Fold (A2, 0) (K, L)
              else
                Fold_Sum.Acc.Fold (A1, 0) (K, L)
                - To_Big (Value (A1 (I, J)))
                = Fold_Sum.Acc.Fold (A2, 0) (K, L)
                  - To_Big (Value (A2 (I, J))));
      end loop;
   end loop;

   pragma
     Assert
       (Fold_Sum.Acc.Fold (A1, 0) (A1'Last (1), A1'Last (2))
        - To_Big (Value (A1 (I, J)))
        = Fold_Sum.Acc.Fold (A2, 0) (A1'Last (1), A1'Last (2))
          - To_Big (Value (A2 (I, J))));
end Update_Sum;
