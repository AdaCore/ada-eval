--
--  Copyright (C) 2018-2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

separate (SPARK.Higher_Order.Fold.Sum_2.Big_Integer_Sum)
procedure Sum_Cst (A : Array_Type; C : Element_Out) is
   function Sum_Cst (I : Index_1; J : Index_2) return Boolean
   is (Fold_Sum.Acc.Fold (A, 0) (I, J)
       = To_Big (C)
         * ((To_Big_1 (I) - To_Big_1 (A'First (1)))
            * Length_2 (A)
            + (To_Big_2 (J) - To_Big_2 (A'First (2))))
         + To_Big (C))
   with Pre => I in A'Range (1) and then J in A'Range (2);
   --  The property we want to show at position I, J

begin
   if A'Length (2) > 0 then
      for I in A'Range (1) loop
         pragma
           Loop_Invariant
             (I = A'First (1)
              or else
                (for all K in A'First (1) .. I - 1 =>
                   (for all L in A'Range (2) =>
                      Value (A (K, L)) = C)));
         pragma
           Loop_Invariant
             (I = A'First (1) or else Sum_Cst (I - 1, A'Last (2)));
         for J in A'Range (2) loop
            if Value (A (I, J)) /= C then
               return;
            end if;
            pragma Loop_Invariant (Sum_Cst (I, J));
            pragma
              Loop_Invariant
                (for all L in A'First (2) .. J =>
                   Value (A (I, L)) = C);
         end loop;
      end loop;
   end if;
end Sum_Cst;
