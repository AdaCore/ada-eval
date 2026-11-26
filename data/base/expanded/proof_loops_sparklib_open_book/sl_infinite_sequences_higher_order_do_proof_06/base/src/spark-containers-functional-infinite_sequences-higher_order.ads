--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;

generic
package SPARK.Containers.Functional.Infinite_Sequences.Higher_Order with
    SPARK_Mode,
    Always_Terminates
is

   function Sum_Rec
     (S     : Sequence;
      Last  : Big_Natural;
      Value : not null access function (E : Element_Type) return Big_Integer)
      return Big_Integer
   is (if Last = 0
       then Big_Natural'(0)
       else Sum_Rec (S, Last - 1, Value) + Value (Get (S, Last)))
   with
     Ghost              => SPARKlib_Full,
     Annotate           => (GNATprove, Higher_Order_Specialization),
     Subprogram_Variant => (Decreases => Last),
     Pre                => Last <= Length (S);
   --  Recursive version of Sum

   function Sum
     (S     : Sequence;
      Last  : Big_Natural;
      Value : not null access function (E : Element_Type) return Big_Integer)
      return Big_Integer
   with
     Global   => null,
     Import,
     Annotate => (GNATprove, Higher_Order_Specialization),
     Pre      => (SPARKlib_Defensive => Last <= Length (S)),
     Post     =>
       (SPARKlib_Full =>
          (if Last = 0 then Sum'Result = 0)
          and then Sum'Result = Sum_Rec (S, Last, Value));
   --  Sum the result of the Value function on all elements of S up to Last

   procedure Lemma_Sum_Eq
     (S1, S2 : Sequence;
      Last   : Big_Natural;
      Value  : not null access function (E : Element_Type) return Big_Integer)
     --  Automatically instantiated lemma:
     --  Sum returns the same value on sequences containing the same elements.

   with
     Ghost    => SPARKlib_Full,
     Global   => null,
     Annotate => (GNATprove, Higher_Order_Specialization),
     Annotate => (GNATprove, Automatic_Instantiation),
     Pre      =>
       Last <= Length (S1)
       and then Last <= Length (S2)
       and then Range_Equal (S1, S2, 1, Last),
     Post     => Sum (S1, Last, Value) = Sum (S2, Last, Value);

end SPARK.Containers.Functional.Infinite_Sequences.Higher_Order;
