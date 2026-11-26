--
--  Copyright (C) 2004-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;

package body Lists with SPARK_Mode
is

   procedure Delete
     (Container : in out List; Position : in out Cursor; Count : Count_Type)
   is separate;

end Lists;
