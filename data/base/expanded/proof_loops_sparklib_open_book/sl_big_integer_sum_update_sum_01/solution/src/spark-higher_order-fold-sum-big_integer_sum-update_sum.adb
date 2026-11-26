--
--  Copyright (C) 2018-2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

separate (SPARK.Higher_Order.Fold.Sum.Big_Integer_Sum)
procedure Update_Sum (A1, A2 : Array_Type; I : Index_Type) is

begin
   for K in A1'Range loop
      pragma
        Loop_Invariant
          (if K < I
           then
             Sum_Left.Acc.Fold (A1, 0) (K)
             = Sum_Left.Acc.Fold (A2, 0) (K)
           else
             Sum_Left.Acc.Fold (A1, 0) (K) - To_Big (Value (A1 (I)))
             = Sum_Left.Acc.Fold (A2, 0) (K)
               - To_Big (Value (A2 (I))));
   end loop;

   pragma
     Assert
       (Sum_Left.Acc.Fold (A1, 0) (A1'Last) - To_Big (Value (A1 (I)))
        = Sum_Left.Acc.Fold (A2, 0) (A1'Last)
          - To_Big (Value (A2 (I))));
end Update_Sum;
