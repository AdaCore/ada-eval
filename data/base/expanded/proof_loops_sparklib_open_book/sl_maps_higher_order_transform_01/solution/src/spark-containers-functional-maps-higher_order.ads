--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;

generic
package SPARK.Containers.Functional.Maps.Higher_Order with
    SPARK_Mode,
    Always_Terminates
is
   use SPARK.Big_Integers;

   function Transform
     (M              : Map;
      Transform_Key  : not null access function (K : Key_Type) return Key_Type;
      Transform_Item :
        not null access function (E : Element_Type) return Element_Type)
      return Map
   with
     Global   => null,
     Annotate => (GNATprove, Higher_Order_Specialization),
     Pre      =>
       (SPARKlib_Full =>
          (for all K1 of M =>
             (for all K2 of M =>
                Equivalent_Keys (Transform_Key (K1), Transform_Key (K2))
                = Equivalent_Keys (K1, K2)))),
     Post     =>
       (SPARKlib_Full =>
          Length (Transform'Result) = Length (M)
          and then
            (for all K of M => Has_Key (Transform'Result, Transform_Key (K)))
          and then
            (for all K of Transform'Result =>
               (for some L of M => Equivalent_Keys (K, Transform_Key (L))))
          and then
            (for all K of M =>
               Element_Logic_Equal
                 (Get (Transform'Result, Transform_Key (K)),
                  Transform_Item (Get (M, K)))));
   --  Return a new map containing a mapping per key in M. The new associations
   --  are obtained using Transform_Key and Transform_Item. Transform_Key
   --  shall not collapse keys together.

end SPARK.Containers.Functional.Maps.Higher_Order;
