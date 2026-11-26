--
--  Copyright (C) 2018-2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

separate (SPARK.Higher_Order.Fold.Count_2)
procedure Update_Count (A1, A2 : Array_Type; I : Index_1; J : Index_2) is
   C : constant Integer :=
     (if Choose (A1 (I, J)) = Choose (A2 (I, J))
      then 0
      elsif Choose (A1 (I, J))
      then 1
      else -1);

   function Update_Count (K : Index_1; L : Index_2) return Boolean
   is
      --  Update_Count up to the position I, J

      (if K < I or else (K = I and then L < J)
       then
         Fold_Count.Acc.Fold (A1, 0) (K, L)
         = Fold_Count.Acc.Fold (A2, 0) (K, L)
       else
         Fold_Count.Acc.Fold (A1, 0) (K, L)
         = Fold_Count.Acc.Fold (A2, 0) (K, L) + C)
   with
     Pre =>
       I in A1'Range (1)
       and then J in A1'Range (2)
       and then K in A1'Range (1)
       and then L in A1'Range (2)
       and then A1'First (1) = A2'First (1)
       and then A1'Last (1) = A2'Last (1)
       and then A1'First (2) = A2'First (2)
       and then A1'Last (2) = A2'Last (2);

   procedure Prove_Next (K : Index_1; L : Index_2)
   with
     --  Prove Update_Count in the next iteration

     Pre  =>
       I in A1'Range (1)
       and then J in A1'Range (2)
       and then K in A1'Range (1)
       and then L in A1'Range (2)
       and then A1'First (1) = A2'First (1)
       and then A1'Last (1) = A2'Last (1)
       and then A1'First (2) = A2'First (2)
       and then A1'Last (2) = A2'Last (2)
       and then
         (for all K in A1'Range (1) =>
            (for all L in A2'Range (2) =>
               (if K /= I or else L /= J then A1 (K, L) = A2 (K, L))))
       and then
         (if L > A1'First (2)
          then Update_Count (K, L - 1)
          elsif K > A1'First (1)
          then Update_Count (K - 1, A1'Last (2))),
     Post => Update_Count (K, L);

   ----------------
   -- Prove_Next --
   ----------------

   procedure Prove_Next (K : Index_1; L : Index_2) is null;

begin
   for K in A1'Range (1) loop
      pragma
        Loop_Invariant
          (if K > A1'First (1) then Update_Count (K - 1, A1'Last (2)));
      for L in A1'Range (2) loop
         Prove_Next (K, L);
         pragma Loop_Invariant (Update_Count (K, L));
      end loop;
   end loop;
end Update_Count;
