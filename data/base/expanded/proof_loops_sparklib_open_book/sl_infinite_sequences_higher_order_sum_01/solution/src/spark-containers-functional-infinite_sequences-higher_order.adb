--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
with SPARK.Big_Intervals; use SPARK.Big_Intervals;

package body SPARK.Containers.Functional.Infinite_Sequences.Higher_Order is

   function Sum_Rec
     (S     : Sequence;
      Last  : Big_Natural;
      Value : not null access function (E : Element_Type) return Big_Integer)
      return Big_Integer
   with
     Ghost              => SPARKlib_Full,
     Subprogram_Variant => (Decreases => Last),
     Pre                => Last <= Length (S);
   --  Recursive version of Sum

   --------------
   -- Sum_Impl --
   --------------

   function Sum_Impl
     (S     : Sequence;
      Last  : Big_Natural;
      Value : not null access function (E : Element_Type) return Big_Integer)
      return Big_Integer
   with
     Pre  => (SPARKlib_Defensive => Last <= Length (S)),
     Post => (SPARKlib_Full => Sum_Impl'Result = Sum_Rec (S, Last, Value));

   function Sum_Impl
     (S     : Sequence;
      Last  : Big_Natural;
      Value : not null access function (E : Element_Type) return Big_Integer)
      return Big_Integer
   is separate;

   ---------
   -- Sum --
   ---------

   function Sum
     (S     : Sequence;
      Last  : Big_Natural;
      Value : not null access function (E : Element_Type) return Big_Integer)
      return Big_Integer
   with
     Refined_Post => (SPARKlib_Full => Sum'Result = Sum_Rec (S, Last, Value))
   is
   begin
      return Sum_Impl (S, Last, Value);
   end Sum;

   -------------
   -- Sum_Rec --
   -------------

   function Sum_Rec
     (S     : Sequence;
      Last  : Big_Natural;
      Value : not null access function (E : Element_Type) return Big_Integer)
      return Big_Integer
   is (if Last = 0
       then Big_Natural'(0)
       else Sum_Rec (S, Last - 1, Value) + Value (Get (S, Last)));

end SPARK.Containers.Functional.Infinite_Sequences.Higher_Order;
