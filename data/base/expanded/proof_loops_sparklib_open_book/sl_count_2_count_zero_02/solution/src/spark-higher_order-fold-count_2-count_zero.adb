--
--  Copyright (C) 2018-2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

separate (SPARK.Higher_Order.Fold.Count_2)
procedure Count_Zero (A : Array_Type) is
begin
   if A'Length (2) > 0 then
      for I in A'Range (1) loop
         pragma
           Loop_Invariant
             (if I > A'First (1)
              then
                (Fold_Count.Acc.Fold (A, 0) (I - 1, A'Last (2)) = 0)
                = (for all K in A'First (1) .. I - 1 =>
                     (for all L in A'Range (2) =>
                        not Choose (A (K, L)))));
         for J in A'Range (2) loop
            pragma
              Loop_Invariant
                ((Fold_Count.Acc.Fold (A, 0) (I, J) = 0)
                 = ((if I > A'First (1)
                     then
                       (for all K in A'First (1) .. I - 1 =>
                          (for all L in A'Range (2) =>
                             not Choose (A (K, L)))))
                    and
                      (for all L in A'First (2) .. J =>
                         not Choose (A (I, L)))));
         end loop;
      end loop;
   end if;
end Count_Zero;
