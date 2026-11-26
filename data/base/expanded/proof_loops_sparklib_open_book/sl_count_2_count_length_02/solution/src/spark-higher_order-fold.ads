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

      type Acc_Array is
        array (Index_1 range <>, Index_2 range <>) of Element_Out;

      function Fold (A : Array_Type; Init : Element_Out) return Acc_Array
      with
        Import,
        Global => null,
        Pre    =>
          A'Length (1) > 0
          and then A'Length (2) > 0
          and then Ind_Prop (A, Init, A'First (1), A'First (2)),
        Post   =>
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
        Import,
        Global => null,
        Pre    =>
          A'Length (1) = 0
          or else A'Length (2) = 0
          or else Ind_Prop (A, Init, A'First (1), A'First (2)),
        Post   =>
          (if A'Length (1) = 0 or else A'Length (2) = 0
           then Fold'Result = Init
           else
             Fold'Result = Acc.Fold (A, Init) (A'Last (1), A'Last (2))
             and then Final_Prop (A, Fold'Result));
   end Fold_2;

   generic
      type Index_1 is range <>;
      type Index_2 is range <>;
      type Element is private;
      type Array_Type is array (Index_1 range <>, Index_2 range <>) of Element;
      with function Choose (X : Element) return Boolean;
      --  Count the number of elements for which Choose return True

   package Count_2 with Always_Terminates
   is

      function In_Range
        (A : Array_Type; X : Natural; I : Index_1; J : Index_2) return Boolean
      is (X
          <= Natural (I - A'First (1))
             * A'Length (2)
             + Natural (J - A'First (2)))
      with
        Ghost,
        Pre  => I in A'Range (1) and then J in A'Range (2),
        Post => (if In_Range'Result then X < Integer'Last);

      function Add_One (E : Element; X : Natural) return Natural
      is (if Choose (E) then X + 1 else X)
      with
        Pre  => X < Integer'Last,
        Post => Add_One'Result = (if Choose (E) then X + 1 else X);
      pragma Annotate (GNATprove, Inline_For_Proof, Entity => Add_One);

      function Result_In_Range (A : Array_Type; X : Natural) return Boolean
      is (X <= A'Length (1) * A'Length (2));

      package Fold_Count is new
        Fold_2
          (Index_1     => Index_1,
           Index_2     => Index_2,
           Element_In  => Element,
           Array_Type  => Array_Type,
           Element_Out => Natural,
           Ind_Prop    => In_Range,
           Final_Prop  => Result_In_Range,
           F           => Add_One);

      function Count (A : Array_Type) return Natural
      is (Fold_Count.Fold (A, 0));

      procedure Count_Length (A : Array_Type)
      with
        Ghost,
        Post =>
          (Count (A) = A'Length (1) * A'Length (2))
          = (for all I in A'Range (1) =>
               (for all J in A'Range (2) => Choose (A (I, J))));
      --  Lemma: Count returns A'Length (1) * A'Length (2) if and only if all
      --  elements are chosen.

   end Count_2;

end SPARK.Higher_Order.Fold;
