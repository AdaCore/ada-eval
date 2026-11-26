--
--  Copyright (C) 2018-2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

package SPARK.Higher_Order.Fold
  with SPARK_Mode
is

   generic
      type Index_1 is range <>;
      type Index_2 is range <>;
      type Element_In is private;
      type Array_Type is
        array (Index_1 range <>, Index_2 range <>) of Element_In;
      type Element_Out is private;
      with
        function Ind_Prop
          (A : Array_Type; X : Element_Out; I : Index_1; J : Index_2)
           return Boolean
        with Ghost;
      --  Potential inductive property that should be maintained during fold

      with
        function Final_Prop (A : Array_Type; X : Element_Out) return Boolean
        with Ghost;
      --  Potential inductive property at the last iteration

      with function F (X : Element_In; I : Element_Out) return Element_Out;
      --  Function that should be applied to elements of Array_Type

   package Fold_2_Acc with Always_Terminates
   is

      procedure Prove_Ind_Col
        (A : Array_Type; X : Element_Out; I : Index_1; J : Index_2)
      with
        Ghost,
        Import,
        Global => null,
        Pre  =>
          I in A'Range (1)
          and then J in A'Range (2)
          and then J /= A'Last (2)
          and then Ind_Prop (A, X, I, J),
        Post => Ind_Prop (A, F (A (I, J), X), I, J + 1);
      --  Axiom: Ind_Prop should be preserved when going to next column

      procedure Prove_Ind_Row (A : Array_Type; X : Element_Out; I : Index_1)
      with
        Ghost,
        Import,
        Global => null,
        Pre  =>
          I in A'Range (1)
          and then A'Length (2) > 0
          and then I /= A'Last (1)
          and then Ind_Prop (A, X, I, A'Last (2)),
        Post => Ind_Prop (A, F (A (I, A'Last (2)), X), I + 1, A'First (2));
      --  Axiom: Ind_Prop should be preserved when going to next row

      --  Prove_Last is deleted: Fold_2's Fold does not call it.

      type Acc_Array is
        array (Index_1 range <>, Index_2 range <>) of Element_Out;

      function Fold (A : Array_Type; Init : Element_Out) return Acc_Array
      with
        Import,
        Global => null,
        Pre  =>
          A'Length (1) > 0
          and then A'Length (2) > 0
          and then Ind_Prop (A, Init, A'First (1), A'First (2)),
        Post =>
          Fold'Result'First (1) = A'First (1)
          and then Fold'Result'Last (1) = A'Last (1)
          and then Fold'Result'First (2) = A'First (2)
          and then Fold'Result'Last (2) = A'Last (2)
          and then Ind_Prop (A, Init, A'First (1), A'First (2))
          and then
            Fold'Result (A'First (1), A'First (2))
            = F (A (A'First (1), A'First (2)), Init)
          and then
            (for all I in A'Range (1) =>
               (if I > A'First (1)
                then
                  Ind_Prop (A, Fold'Result (I - 1, A'Last (2)), I, A'First (2))
                  and then
                    Fold'Result (I, A'First (2))
                    = F (A (I, A'First (2)), Fold'Result (I - 1, A'Last (2)))))
          and then
            (for all I in A'Range (1) =>
               (for all J in A'Range (2) =>
                  (if J > A'First (2)
                   then
                     Ind_Prop (A, Fold'Result (I, J - 1), I, J)
                     and then
                       Fold'Result (I, J)
                       = F (A (I, J), Fold'Result (I, J - 1)))))
          and then Final_Prop (A, Fold'Result (A'Last (1), A'Last (2)));
   end Fold_2_Acc;

   generic
      type Index_1 is range <>;
      type Index_2 is range <>;
      type Element_In is private;
      type Array_Type is
        array (Index_1 range <>, Index_2 range <>) of Element_In;
      type Element_Out is private;
      with
        function Ind_Prop
          (A : Array_Type; X : Element_Out; I : Index_1; J : Index_2)
           return Boolean
        with Ghost;
      --  Potential inductive property that should be maintained during fold

      with
        function Final_Prop (A : Array_Type; X : Element_Out) return Boolean
        with Ghost;
      --  Potential inductive property at the last iteration

      with function F (X : Element_In; I : Element_Out) return Element_Out;
      --  Function that should be applied to elements of Array_Type

   package Fold_2 with Always_Terminates
   is

      package Acc is new
        Fold_2_Acc
          (Index_1     => Index_1,
           Index_2     => Index_2,
           Element_In  => Element_In,
           Array_Type  => Array_Type,
           Element_Out => Element_Out,
           Ind_Prop    => Ind_Prop,
           Final_Prop  => Final_Prop,
           F           => F);

      function Fold (A : Array_Type; Init : Element_Out) return Element_Out
      with
        Pre  =>
          A'Length (1) = 0
          or else A'Length (2) = 0
          or else Ind_Prop (A, Init, A'First (1), A'First (2)),
        Post =>
          (if A'Length (1) = 0 or else A'Length (2) = 0
           then Fold'Result = Init
           else
             Fold'Result = Acc.Fold (A, Init) (A'Last (1), A'Last (2))
             and then Final_Prop (A, Fold'Result));
   end Fold_2;

end SPARK.Higher_Order.Fold;
