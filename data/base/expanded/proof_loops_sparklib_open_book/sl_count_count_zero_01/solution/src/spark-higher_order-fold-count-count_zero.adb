--
--  Copyright (C) 2018-2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

separate (SPARK.Higher_Order.Fold.Count)
procedure Count_Zero (A : Array_Type) is
begin
   for I in A'Range loop
      pragma
        Loop_Invariant
          ((Count_Left.Acc.Fold (A, 0) (I) = 0)
           = (for all K in A'First .. I => not Choose (A (K))));
   end loop;
end Count_Zero;
