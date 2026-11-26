--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;

with SPARK.Big_Integers; use SPARK.Big_Integers;

package body SPARK.Containers.Functional.Sets.Higher_Order is

   -------------------
   -- Eq_Compatible --
   -------------------

   function Eq_Compatible
     (S    : Set;
      Test : not null access function (E : Element_Type) return Boolean)
      return Boolean
   is (for all E1 of S =>
         (for all E2 of S =>
            (if Equivalent_Elements (E1, E2) then Test (E1) = Test (E2))));

   ------------
   -- Filter --
   ------------

   function Filter
     (S    : Set;
      Test : not null access function (E : Element_Type) return Boolean)
      return Set is separate;

end SPARK.Containers.Functional.Sets.Higher_Order;
