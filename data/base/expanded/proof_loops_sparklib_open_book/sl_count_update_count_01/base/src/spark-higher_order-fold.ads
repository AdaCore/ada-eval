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

      type Acc_Array is array (Index_Type range <>) of Element_Out;

      function Fold (A : Array_Type; Init : Element_Out) return Acc_Array
      with
        Import,
        Global => null,
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

   package Fold_Left with Always_Terminates
   is

      package Acc is new
        Fold_Left_Acc
          (Index_Type  => Index_Type,
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
        Pre  => A'Length = 0 or else Ind_Prop (A, Init, A'First),
        Post =>
          Fold'Result
          = (if A'Length = 0 then Init else Acc.Fold (A, Init) (A'Last));
   end Fold_Left;

   generic
      type Index_Type is range <>;
      type Element is private;
      type Array_Type is array (Index_Type range <>) of Element;
      with function Choose (X : Element) return Boolean;
      --  Count the number of elements for which Choose return True

   package Count with Always_Terminates
   is

      function In_Range
        (A : Array_Type; X : Natural; I : Index_Type) return Boolean
      is (X <= Natural (I - A'First))
      with Ghost, Pre => I in A'Range;

      function In_Range_Last (A : Array_Type; X : Natural) return Boolean
      is (X <= A'Length)
      with Ghost;

      function Add_One (E : Element; X : Natural) return Natural
      is (if Choose (E) then X + 1 else X)
      with Pre => X < Integer'Last;
      pragma Annotate (GNATprove, Inline_For_Proof, Entity => Add_One);

      package Count_Left is new
        Fold_Left
          (Index_Type  => Index_Type,
           Element_In  => Element,
           Array_Type  => Array_Type,
           Element_Out => Natural,
           Ind_Prop    => In_Range,
           Final_Prop  => In_Range_Last,
           F           => Add_One);

      function Count (A : Array_Type) return Natural
      is (Count_Left.Fold (A, 0));

      procedure Update_Count (A1, A2 : Array_Type; I : Index_Type)
      with
        Ghost,
        Pre            =>
          I in A1'Range
          and then A1'First = A2'First
          and then A1'Last = A2'Last
          and then (for all K in A1'Range => (if K /= I then A1 (K) = A2 (K))),
        Contract_Cases =>
          (Choose (A1 (I)) = Choose (A2 (I))       => Count (A1) = Count (A2),
           Choose (A1 (I)) and not Choose (A2 (I)) =>
             Count (A1) = Count (A2) + 1,
           not Choose (A1 (I)) and Choose (A2 (I)) =>
             Count (A1) + 1 = Count (A2));
      --  Lemma: Modification of Count after an update to the array
   end Count;

end SPARK.Higher_Order.Fold;
