with Ada.Numerics.Big_Numbers.Big_Integers;
use Ada.Numerics.Big_Numbers.Big_Integers;
with SPARK.Higher_Order;
with SPARK.Higher_Order.Fold;
with Test_Types;                            use Test_Types;

package Test_Higher_Order2
  with SPARK_Mode
is

   function Value (X : Integer) return My_Small
   with Import, Global => null;

   --  Value cannot have a pre as it is called from Add_Value without any
   --  constraint.

   function Valid_Integer (Arg : Big_Integer) return Boolean
   is (In_Range
         (Arg,
          Low  => To_Big_Integer (Integer'First),
          High => To_Big_Integer (Integer'Last)));

   package My_Sum_Int is new
     SPARK.Higher_Order.Fold.Sum
       (Index_Type  => Small_Index,
        Element_In  => Integer,
        Array_Type  => Small_Array,
        Element_Out => Integer,
        Zero        => 0,
        Add         => "+",
        To_Big      => To_Big_Integer,
        In_Range    => Valid_Integer,
        Value       => Value);

   procedure Prove_No_Overflows (A : Small_Array)
   with
     Ghost,
     Always_Terminates,
     Global => null,
     Post => My_Sum_Int.No_Overflows (A);

end Test_Higher_Order2;
