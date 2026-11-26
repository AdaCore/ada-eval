--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;

package body SPARK.Containers.Functional.Maps.Higher_Order is

   ------------
   -- Create --
   ------------

   function Create
     (New_Length : Big_Natural;
      New_Key    : not null access function (I : Big_Positive) return Key_Type;
      New_Item   :
        not null access function (I : Big_Positive) return Element_Type)
      return Map
   is separate;

end SPARK.Containers.Functional.Maps.Higher_Order;
