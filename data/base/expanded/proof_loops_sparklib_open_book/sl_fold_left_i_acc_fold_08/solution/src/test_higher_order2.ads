with SPARK.Higher_Order;
with SPARK.Higher_Order.Fold;
with Test_Types; use Test_Types;

package Test_Higher_Order2
  with SPARK_Mode
is

   function Ind_Prop (A : My_Array; X : Integer; I : My_Index) return Boolean
   with Import, Pre => I in A'Range, Global => null;

   --  Ind_Prop cannot have another precondition, or Prove_Ind and Prove_Last
   --  would not be self guarded.

   function Final_Prop (A : My_Array; X : Integer) return Boolean
   with Import, Pre => A'Length > 0, Global => null;

   --  Final_Prop is always called after Prove_Last, which has the same
   --  application of Final_Prop as a post. There can be no error.

   function F (X : Integer; K : My_Index; I : Integer) return Integer
   with Import, Global => null;

   --  F is always called on A (I), I, X when Ind_Prop (A, X, I) is True.
   --  It is called in the same conditions in Prove_Ind and Prove_Last.

   package My_Fold_Left_I is new
     SPARK.Higher_Order.Fold.Fold_Left_I
       (Index_Type  => My_Index,
        Element_In  => Integer,
        Array_Type  => My_Array,
        Element_Out => Integer,
        Ind_Prop    => Ind_Prop,
        Final_Prop  => Final_Prop,
        F           => F);

end Test_Higher_Order2;
