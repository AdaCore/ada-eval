--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;

with SPARK.Big_Integers; use SPARK.Big_Integers;

package body SPARK.Containers.Functional.Sets.Higher_Order is

   ------------------------
   -- Transform_Distinct --
   ------------------------

   function Transform_Distinct
     (S              : Set;
      Transform_Item :
        not null access function (E : Element_Type) return Element_Type)
      return Set is separate;

end SPARK.Containers.Functional.Sets.Higher_Order;
