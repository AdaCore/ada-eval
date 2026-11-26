--
--  Copyright (C) 2018-2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

package SPARK.Higher_Order
  with SPARK_Mode
is

   generic
      type Index_Type is range <>;
      type Element is private;
      type Array_Type is array (Index_Type range <>) of Element;
      with
        function Init_Prop (I : Index_Type; A : Element) return Boolean
        with Ghost;
      --  Potential additional constraint on values of the array to allow Map

      with function F (I : Index_Type; X : Element) return Element;
      --  Function that should be applied to elements of Array_In

     procedure Map_I_Proc (A : in out Array_Type)
   with
     Pre  => (for all I in A'Range => Init_Prop (I, A (I))),
     Post => (for all I in A'Range => A (I) = F (I, A'Old (I)));

end SPARK.Higher_Order;
