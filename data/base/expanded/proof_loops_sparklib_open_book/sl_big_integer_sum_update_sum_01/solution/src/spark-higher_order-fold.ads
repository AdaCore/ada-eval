--
--  Copyright (C) 2018-2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Warnings (Off, "formal object ""Zero"" is not referenced");
pragma Warnings (Off, "function ""Add"" is not referenced");
pragma Warnings (Off, "function ""In_Range"" is not referenced");
with SPARK.Big_Integers; use SPARK.Big_Integers;

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
        Pre    => A'Length > 0 and then Ind_Prop (A, Init, A'First),
        Post   =>
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
        Pre    => A'Length = 0 or else Ind_Prop (A, Init, A'First),
        Post   =>
          Fold'Result
          = (if A'Length = 0 then Init else Acc.Fold (A, Init) (A'Last));
   end Fold_Left;

   generic
      type Index_Type is range <>;
      type Element_In is private;
      type Array_Type is array (Index_Type range <>) of Element_In;
      type Element_Out is private;
      with function Add (Left, Right : Element_Out) return Element_Out;
      Zero : Element_Out;
      with
        function To_Big (X : Element_Out) return Big_Integer
        with Ghost;
      with
        function In_Range (X : Big_Integer) return Boolean
        with Ghost;

      with function Value (X : Element_In) return Element_Out;
      --  Sum the value of each element
   package Sum with Always_Terminates is

      package Big_Integer_Sum
        with Ghost
      is
         function Always_True
           (Unused_A : Array_Type;
            Unused_X : Big_Integer;
            Unused_I : Index_Type) return Boolean
         is (True);

         function Always_True
           (Unused_A : Array_Type; Unused_X : Big_Integer) return Boolean
         is (True);

         function Add_Value
           (X : Element_In; Y : Big_Integer) return Big_Integer
         is (To_Big (Value (X)) + Y);
         pragma Annotate (GNATprove, Inline_For_Proof, Entity => Add_Value);

         package Sum_Left is new
           Fold_Left
             (Index_Type  => Index_Type,
              Element_In  => Element_In,
              Array_Type  => Array_Type,
              Element_Out => Big_Integer,
              Ind_Prop    => Always_True,
              Final_Prop  => Always_True,
              F           => Add_Value);

         function Sum (A : Array_Type) return Big_Integer
         is (Sum_Left.Fold (A, 0));

         procedure Update_Sum (A1, A2 : Array_Type; I : Index_Type)
         with
           Pre  =>
             I in A1'Range
             and then A1'First = A2'First
             and then A1'Last = A2'Last
             and then
               (for all K in A1'Range => (if K /= I then A1 (K) = A2 (K))),
           Post =>
             Sum (A2) - To_Big (Value (A2 (I)))
             = Sum (A1) - To_Big (Value (A1 (I)));
         --  Lemma: Modification of Sum after an update to the array

      end Big_Integer_Sum;

   end Sum;

end SPARK.Higher_Order.Fold;
