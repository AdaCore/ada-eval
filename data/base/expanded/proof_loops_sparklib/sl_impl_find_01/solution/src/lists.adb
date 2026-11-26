--
--  Copyright (C) 2004-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;

package body Lists with SPARK_Mode
is

   function Find
     (Container : List; Item : Element_Type; Position : Cursor := No_Element)
      return Cursor
   is separate;

end Lists;
