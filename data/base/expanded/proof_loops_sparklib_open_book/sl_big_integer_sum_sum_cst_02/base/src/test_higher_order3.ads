with Ada.Numerics.Big_Numbers.Big_Integers;
use Ada.Numerics.Big_Numbers.Big_Integers;
with SPARK.Higher_Order;
with SPARK.Higher_Order.Fold;
with Test_Types;                            use Test_Types;

package Test_Higher_Order3
  with SPARK_Mode
is

   function Value (X : Integer) return My_Small
   with Global => null, Import;

   --  Value cannot have a pre as it is called from Add_Value without any
   --  constraint.

   function In_Range (X : Big_Integer) return Boolean
   with Global => null, Import, Ghost;

   function To_Big (X : Integer) return Big_Integer
   with Global => null, Import, Ghost;

   function Add_Pre (X, Y : Integer) return Boolean
   with Global => null, Import;

   function Add (X, Y : Integer) return Integer
   with Global => null, Import, Pre => Add_Pre (X, Y);

   package My_Sum_2 is new
     SPARK.Higher_Order.Fold.Sum_2
       (Index_1     => Small_Index,
        Index_2     => Small_Index,
        Element_In  => Integer,
        Element_Out => Integer,
        Zero        => 0,
        Add         => Add,
        To_Big      => To_Big,
        In_Range    => In_Range,
        Array_Type  => Small_Matrix,
        Value       => Value);

end Test_Higher_Order3;
