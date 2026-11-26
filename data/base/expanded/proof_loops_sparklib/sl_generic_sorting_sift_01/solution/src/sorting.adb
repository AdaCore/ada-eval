--
--  Copyright (C) 2004-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

package body Sorting with SPARK_Mode is

   ----------
   -- Sift --
   ----------

   procedure Sift
     (Container : in out Vector; S : T; Max : T; Temp : Element_Type)
   is separate;

end Sorting;
