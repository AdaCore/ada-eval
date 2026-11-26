--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;

package body SPARK.Containers.Functional.Maps.Higher_Order is

   ------------
   -- Filter --
   ------------

   function Filter
     (M    : Map;
      Test :
        not null access function
          (K : Key_Type; E : Element_Type) return Boolean) return Map
          is separate;

   -------------------
   -- Eq_Compatible --
   -------------------

   function Eq_Compatible
     (M    : Map;
      Test :
        not null access function
          (K : Key_Type; E : Element_Type) return Boolean) return Boolean
   is (for all K1 of M =>
         (for all K2 of M =>
            (if Equivalent_Keys (K1, K2)
             then Test (K1, Get (M, K1)) = Test (K2, Get (M, K2)))));

end SPARK.Containers.Functional.Maps.Higher_Order;
