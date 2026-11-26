--
--  Copyright (C) 2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

package body Reachability is

   -------------------------
   -- Model_From_Reversed --
   -------------------------

   function Model_From_Reversed
     (X : Extended_Index; M : Memory_Type; R_Seq : Sequence) return Sequence
     is separate;

end Reachability;
