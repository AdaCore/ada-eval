--
--  Copyright (C) 2018-2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

separate (SPARK.Higher_Order)
procedure Map_Proc (A : in out Array_Type) is
begin
   for I in A'Range loop
      A (I) := F (A (I));
   end loop;
end Map_Proc;
