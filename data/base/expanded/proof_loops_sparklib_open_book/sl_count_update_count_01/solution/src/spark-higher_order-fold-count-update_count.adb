--
--  Copyright (C) 2018-2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

separate (SPARK.Higher_Order.Fold.Count)
procedure Update_Count (A1, A2 : Array_Type; I : Index_Type) is
   C : constant Integer :=
     (if (Choose (A1 (I)) and Choose (A2 (I)))
        or (not Choose (A1 (I)) and not Choose (A2 (I)))
      then 0
      elsif Choose (A1 (I))
      then 1
      else -1);
begin
   for K in A1'Range loop
      pragma
        Loop_Invariant
          (if K < I
           then
             Count_Left.Acc.Fold (A1, 0) (K)
             = Count_Left.Acc.Fold (A2, 0) (K)
           else
             Count_Left.Acc.Fold (A1, 0) (K)
             = Count_Left.Acc.Fold (A2, 0) (K) + C);
   end loop;
   pragma
     Assert
       (Count_Left.Acc.Fold (A1, 0) (A1'Last)
        = Count_Left.Acc.Fold (A2, 0) (A1'Last) + C);
end Update_Count;
