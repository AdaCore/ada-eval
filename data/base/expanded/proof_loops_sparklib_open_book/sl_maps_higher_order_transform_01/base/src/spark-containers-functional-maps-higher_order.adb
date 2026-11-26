--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;

package body SPARK.Containers.Functional.Maps.Higher_Order is

   ---------------
   -- Transform --
   ---------------

   function Transform
     (M              : Map;
      Transform_Key  : not null access function (K : Key_Type) return Key_Type;
      Transform_Item :
        not null access function (E : Element_Type) return Element_Type)
      return Map is separate;

end SPARK.Containers.Functional.Maps.Higher_Order;
