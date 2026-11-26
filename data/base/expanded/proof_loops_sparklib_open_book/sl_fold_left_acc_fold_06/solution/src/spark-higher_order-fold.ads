--
--  Copyright (C) 2018-2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

package SPARK.Higher_Order.Fold
  with SPARK_Mode
is

   generic
      type Index_Type is range <>;
      type Element_In is private;
      type Array_Type is array (Index_Type range <>) of Element_In;
      type Element_Out is private;
      with
        function Ind_Prop
          (A : Array_Type; X : Element_Out; I : Index_Type) return Boolean
        with Ghost;
      --  Potential inductive property that should be maintained during fold

      with
        function Final_Prop (A : Array_Type; X : Element_Out) return Boolean
        with Ghost;
      --  Potential inductive property at the last iteration

      with function F (X : Element_In; I : Element_Out) return Element_Out;
      --  Function that should be applied to elements of Array_Type

   package Fold_Left_Acc with Always_Terminates
   is

      procedure Prove_Ind (A : Array_Type; X : Element_Out; I : Index_Type)
      with
        Ghost,
        Import,
        Global => null,
        Pre    =>
          I in A'Range and then Ind_Prop (A, X, I) and then I /= A'Last,
        Post   => Ind_Prop (A, F (A (I), X), I + 1);
      --  Axiom: Ind_Prop should be preserved when going to next index

      procedure Prove_Last (A : Array_Type; X : Element_Out)
      with
        Ghost,
        Import,
        Global => null,
        Pre    => A'Length > 0 and then Ind_Prop (A, X, A'Last),
        Post   => Final_Prop (A, F (A (A'Last), X));
      --  Axiom: Final_Prop should be provable at the last iteration from
      --  Ind_Prop.

      type Acc_Array is array (Index_Type range <>) of Element_Out;

      function Fold (A : Array_Type; Init : Element_Out) return Acc_Array
      with
        Pre  => A'Length > 0 and then Ind_Prop (A, Init, A'First),
        Post =>
          Fold'Result'First = A'First
          and then Fold'Result'Last = A'Last
          and then Ind_Prop (A, Init, A'First)
          and then Fold'Result (A'First) = F (A (A'First), Init)
          and then
            (for all I in A'Range =>
               (if I > A'First
                then
                  Ind_Prop (A, Fold'Result (I - 1), I)
                  and then Fold'Result (I) = F (A (I), Fold'Result (I - 1))))
          and then Final_Prop (A, Fold'Result (A'Last));
   end Fold_Left_Acc;

end SPARK.Higher_Order.Fold;
