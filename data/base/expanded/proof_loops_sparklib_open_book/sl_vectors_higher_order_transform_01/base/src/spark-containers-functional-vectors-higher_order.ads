--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;

generic
package SPARK.Containers.Functional.Vectors.Higher_Order with
    SPARK_Mode,
    Always_Terminates
is

   function Transform
     (S              : Sequence;
      Transform_Item :
        not null access function (E : Element_Type) return Element_Type)
      return Sequence
   with
     Global   => null,
     Annotate => (GNATprove, Higher_Order_Specialization),
     Post     =>
       (SPARKlib_Full =>
          Length (Transform'Result) = Length (S)
          and then
            (for all I in Index_Type'First .. Last (S) =>
               Element_Logic_Equal
                 (Get (Transform'Result, I), Transform_Item (Get (S, I)))));
   --  Return a new sequence with the same length as S. Its elements are
   --  obtained using Transform_Item on the elements of S.

end SPARK.Containers.Functional.Vectors.Higher_Order;
