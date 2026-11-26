--
--  Copyright (C) 2018-2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

separate (SPARK.Higher_Order.Fold.Count_2)
procedure Count_Length (A : Array_Type) is

   function Count_Length (I : Index_1; J : Index_2) return Boolean
   is
      --  Count_Length up to the position I, J

      ((Fold_Count.Acc.Fold (A, 0) (I, J)
        = Natural (I - A'First (1))
          * A'Length (2)
          + Natural (J - A'First (2))
          + 1)
       = ((if I > A'First (1)
           then
             (for all K in A'First (1) .. I - 1 =>
                (for all L in A'Range (2) => Choose (A (K, L)))))
          and (for all L in A'First (2) .. J => Choose (A (I, L)))))
   with
     Pre  => I in A'Range (1) and then J in A'Range (2),
     Post =>
       (if J = A'Last (2) then Count_Length'Result = Count_Length (I));

   function Count_Length (I : Index_1) return Boolean
   is
      --  Count_Length up to the position I, A'Last (2)

      ((Fold_Count.Acc.Fold (A, 0) (I, A'Last (2))
        = Natural (I - A'First (1) + 1) * A'Length (2))
       = (for all K in A'First (1) .. I =>
            (for all L in A'Range (2) => Choose (A (K, L)))))
   with
     Pre  => I in A'Range (1) and then A'Length (2) > 0,
     Post =>
       (if I = A'Last (1)
        then
          Count_Length'Result
          = ((Count (A) = A'Length (1) * A'Length (2))
             = (for all I in A'Range (1) =>
                  (for all J in A'Range (2) => Choose (A (I, J))))));

   procedure Prove_Next (I : Index_1; J : Index_2)
   with
     --  Prove Count_Length in the next iteration

     Pre  =>
       I in A'Range (1)
       and then J in A'Range (2)
       and then
         (if J > A'First (2)
          then Count_Length (I, J - 1)
          elsif I > A'First (1)
          then Count_Length (I - 1)),
     Post => Count_Length (I, J);

   ----------------
   -- Prove_Next --
   ----------------

   procedure Prove_Next (I : Index_1; J : Index_2) is null;

begin
   if A'Length (2) > 0 then
      for I in A'Range (1) loop
         for J in A'Range (2) loop
            Prove_Next (I, J);
         end loop;
      end loop;
   end if;
end Count_Length;
