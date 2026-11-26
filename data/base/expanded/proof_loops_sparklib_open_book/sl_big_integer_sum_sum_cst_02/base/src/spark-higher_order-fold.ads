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
      type Element_In is private;
      type Array_Type is
        array (Index_1 range <>, Index_2 range <>) of Element_In;
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
   package Sum_2 with Always_Terminates is

      package Big_Integer_Sum
        with Ghost
      is
         package Index_Conv_1 is new Signed_Conversions (Index_1);
         function To_Big_1 (X : Index_1) return Big_Integer
         renames Index_Conv_1.To_Big_Integer;
         package Index_Conv_2 is new Signed_Conversions (Index_2);
         function To_Big_2 (X : Index_2) return Big_Integer
         renames Index_Conv_2.To_Big_Integer;

         function Always_True
           (Unused_A : Array_Type;
            Unused_X : Big_Integer;
            Unused_I : Index_1;
            Unused_J : Index_2) return Boolean
         is (True);

         function Always_True
           (Unused_A : Array_Type; Unused_X : Big_Integer) return Boolean
         is (True);

         function Add_Value
           (X : Element_In; Y : Big_Integer) return Big_Integer
         is (To_Big (Value (X)) + Y);
         pragma Annotate (GNATprove, Inline_For_Proof, Entity => Add_Value);

         package Fold_Sum is new
           Fold_2
             (Index_1     => Index_1,
              Index_2     => Index_2,
              Element_In  => Element_In,
              Array_Type  => Array_Type,
              Element_Out => Big_Integer,
              Ind_Prop    => Always_True,
              Final_Prop  => Always_True,
              F           => Add_Value);

         function Sum (A : Array_Type) return Big_Integer
         is (Fold_Sum.Fold (A, 0));

         function Length_1 (A : Array_Type) return Big_Integer
         is (if A'Last (1) < A'First (1)
             then 0
             else To_Big_1 (A'Last (1)) - To_Big_1 (A'First (1)) + 1);

         function Length_2 (A : Array_Type) return Big_Integer
         is (if A'Last (2) < A'First (2)
             then 0
             else To_Big_2 (A'Last (2)) - To_Big_2 (A'First (2)) + 1);

         procedure Sum_Cst (A : Array_Type; C : Element_Out)
         with
           Ghost,
           Post =>
             (if (for all I in A'Range (1) =>
                    (for all J in A'Range (2) => Value (A (I, J)) = C))
              then Sum (A) = To_Big (C) * Length_1 (A) * Length_2 (A));
         --  Lemma: Value of Sum on a constant array

      end Big_Integer_Sum;

   end Sum_2;

end SPARK.Higher_Order.Fold;
