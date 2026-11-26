--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
with SPARK.Big_Integers; use SPARK.Big_Integers;

generic
package SPARK.Containers.Functional.Vectors.Higher_Order with
    SPARK_Mode,
    Always_Terminates
is

   function Sum_Rec
     (S     : Sequence;
      Last  : Extended_Index;
      Value : not null access function (E : Element_Type) return Big_Integer)
      return Big_Integer
   with
     Ghost              => SPARKlib_Full,
     Subprogram_Variant => (Decreases => Last),
     Pre                => Last <= Vectors.Last (S);
   --  Recursive version of Sum

   function Sum_Rec
     (S     : Sequence;
      Last  : Extended_Index;
      Value : not null access function (E : Element_Type) return Big_Integer)
      return Big_Integer
   is (if Last = Extended_Index'First
       then Big_Integer'(0)
       else
         Sum_Rec (S, Extended_Index'Pred (Last), Value)
         + Value (Get (S, Last)));

   function Sum
     (S     : Sequence;
      Last  : Extended_Index;
      Value : not null access function (E : Element_Type) return Big_Integer)
      return Big_Integer
   with
     Import,
     Global   => null,
     Pre      => (SPARKlib_Defensive => Last <= Vectors.Last (S)),
     Post     => (SPARKlib_Full => Sum'Result = Sum_Rec (S, Last, Value));
   --  Sum the result of the Value function on all elements of S up to Last

   procedure Lemma_Sum_Eq
     (S1, S2 : Sequence;
      Last   : Extended_Index;
      Value  : not null access function (E : Element_Type) return Big_Integer)
     --  Automatically instantiated lemma:
     --  Sum returns the same value on sequences containing the same elements.

   with
     Ghost    => SPARKlib_Full,
     Global   => null,
     Annotate => (GNATprove, Automatic_Instantiation),
     Pre      =>
       Last <= Vectors.Last (S1)
       and then Last <= Vectors.Last (S2)
       and then Range_Equal (S1, S2, Index_Type'First, Last),
     Post     => Sum (S1, Last, Value) = Sum (S2, Last, Value);

end SPARK.Containers.Functional.Vectors.Higher_Order;
