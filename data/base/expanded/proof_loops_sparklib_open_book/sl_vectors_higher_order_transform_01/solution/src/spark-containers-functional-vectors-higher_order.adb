--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;

package body SPARK.Containers.Functional.Vectors.Higher_Order is

   ---------------
   -- Transform --
   ---------------

   function Transform
     (S              : Sequence;
      Transform_Item :
        not null access function (E : Element_Type) return Element_Type)
      return Sequence is separate;

end SPARK.Containers.Functional.Vectors.Higher_Order;
