--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;

package body SPARK.Containers.Functional.Vectors.Higher_Order is

   ------------
   -- Create --
   ------------

   function Create
     (New_Last : Extended_Index;
      New_Item : not null access function (I : Index_Type) return Element_Type)
      return Sequence is separate;

end SPARK.Containers.Functional.Vectors.Higher_Order;
