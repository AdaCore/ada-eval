--
--  Copyright (C) 2018-2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

separate (SPARK.Higher_Order)
procedure Map_I_Proc (A : in out Array_Type) is
begin
   for I in A'Range loop
      A (I) := F (I, A (I));
      pragma
        Loop_Invariant
          (for all K in A'First .. I => A (K) = F (K, A'Loop_Entry (K)));
   end loop;
end Map_I_Proc;
